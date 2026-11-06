"""C-FoundationStereo kernels forward on TAO's modules (C-FS-S and C-FFS).

The network stays TAO's module tree; only the leaves change:

- every conv runs in fp16, channels-last, as a fp16-accumulate cuDNN graph (`cnn_kernels`), with the following norm
  (BatchNorm as an fp32 affine; InstanceNorm from fp32 statistics), bias, residual and activation as one `igev`
  epilogue. The composite TAO blocks (`Conv`, the ResNet basic blocks, `ResidualBlock`, `conv -> norm -> act` runs in
  `Sequential`s) hand their norm / activation to the conv they wrap.
- conv outputs whose channel count is not a multiple of 8 (the 28-channel volumes) keep the zero channels of the
  padded weights; the next conv reads the padded tensor directly (its weights get zero input channels too).
- C-FS-S: DINOv2 through `fast_vit`; the disparity transformer (28 channels, 4 heads of 7) through the `cfs` kernels
  around flash SDPA; the GRU lookups through `fs.geo_lookup_s` on the channels-last hourglass output (no transposed
  copy of the volume, level 2 pooled on the fly).
- cost volumes built straight into channels-last 3-D buffers; softmax, regression and upsampling weights in fp32.

TAO runs these blocks under fp16 autocast; here the whole network is fp16 (same rounding points for the convs, fp32
islands where autocast has them).
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from fastforks._core.kernels import linear
from fastforks._core.patch import override
from fastforks.kernels import cfs as cfs_kernels
from fastforks.kernels import fs as fs_kernels
from fastforks.kernels import fs_cutlass
from fastforks.kernels.fs_cutlass import FsEpiArgs
from fastforks.shared import cnn_kernels
from fastforks.shared.depth_anything_v2.fast_vit import use_fast_vit

from ..net.foundation_stereo import utils
from ..net.foundation_stereo.convolution_helper import (
    Conv,
    Conv3dNormActReduced,
    ResnetBasicBlock,
    ResnetBasicBlock3D,
)
from ..net.foundation_stereo.extractor import ResidualBlock
from ..net.foundation_stereo.foundation_stereo import FoundationStereo
from ..portable import prepare_portable
from .ffs import FfsForward, cam

if TYPE_CHECKING:
    from fastforks._core.kernels import CubinModule

Patched = Any  # a module carrying our attributes (`_plan`, `_act`, ...) and a replaced `forward`
CL3 = torch.channels_last_3d
NORMS = (nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d, nn.InstanceNorm3d)
PAD_IN_MIN = 8  # convs with >= 8 input channels, not a multiple of 8, get zero input channels (cuDNN fp16 tiles)
STATS_CHUNK = 1 << 24  # elements per fp32 chunk of the instance-norm statistics
STATS_C = 256  # channels `fs.chan_stats8` takes (8 per thread, 256 threads)
STATS_BLOCKS = 288  # its blocks per sample (8 per SM of a 36-SM GPU; a few hundred rows per thread at Midd-H)
ATT_CHUNK = 32768  # disparity sequences per transformer chunk: a whole strip (8192: +30 ms at Midd-H; same result)
SDPA_CHUNK = 16384  # sequences per flash SDPA call (grid limit)
STRIP_SEQ = ATT_CHUNK  # disparity sequences per strip of the hourglass tail
_DIM3D = 5
C3_MIN_K = (
    32  # 3-D convs with this many (padded) output channels run on the 128 x 32 CUTLASS tile, wider ones on 128 x 64
)
C3_MAX_K = 192
C3_MAX_T = 3  # longer filters along the leading volume dim (the 17-tap disparity convs) run in slice-major tile order
_TC = 28  # the cfs kernels' d_model
HID = 128  # C-FS-S GRU hidden size


def _mf(dim: int) -> torch.memory_format:
    return cnn_kernels.memory_format(dim)


# ------------------------------------------------------------------------------------------------ epilogues
def _row_stride(x: torch.Tensor) -> int | None:
    """Row stride of `x` seen as (pixels, channels) rows (channels-last, possibly a channel slice), else None."""
    if x.dtype != torch.float16:
        return None
    if x.is_contiguous(memory_format=_mf(x.dim())):
        return x.shape[1]
    if x.shape[1] > 1 and x.stride(1) != 1:
        return None
    rows = [(n, s) for n, s in zip((x.shape[0], *x.shape[2:]), (x.stride(0), *x.stride()[2:]), strict=True) if n > 1]
    if not rows:
        return x.shape[1]  # a single pixel
    cs = expect = rows[-1][1]
    if cs < x.shape[1]:
        return None
    for n, s in reversed(rows):
        if s != expect:
            return None
        expect *= n
    return cs


class _Epilogue:
    """`y = act(y * s + t + r)` in place on channels-last fp16 rows (`igev` affine kernels)."""

    def __init__(self, k: CubinModule) -> None:
        self.k = k

    def __call__(
        self,
        y: torch.Tensor,
        s: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        act: int = 0,
        r: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if s is None and t is None and not act and r is None:
            return y
        base = getattr(y, "_ff_padded", None)
        rb = None if r is None else getattr(r, "_ff_padded", None)
        if base is not None and y.shape[1] % 8 and act != cnn_kernels.ACT["sigmoid"] and (r is None or rb is not None):
            # a view of a zero-padded conv output: the vector kernel over all channels (zero scale / shift / residual
            # on the pad channels keeps them zero: act(0) = 0)
            pad = base.shape[1] - y.shape[1]
            if rb is None or rb.shape == base.shape:
                s = None if s is None else F.pad(s, (0, pad))
                t = None if t is None else F.pad(t, (0, pad))
                self(base, s, t, act, rb)
                return y
        ys = _row_stride(y)
        if ys is None:
            msg = f"epilogue needs channels-last fp16 rows, got {tuple(y.shape)} {y.stride()} {y.dtype}"
            raise RuntimeError(msg)
        rs = 0
        if r is not None:
            if r.shape != y.shape:
                msg = f"residual {tuple(r.shape)} != {tuple(y.shape)}"
                raise RuntimeError(msg)
            r = r if r.dtype == torch.float16 else r.half()
            rs = _row_stride(r)
            if rs is None:
                r = r.contiguous(memory_format=_mf(r.dim()))
                rs = r.shape[1]
        c = y.shape[1]
        p = y.numel() // c
        vec = c % 8 == 0 and ys % 8 == 0 and y.data_ptr() % 16 == 0
        vec = vec and (r is None or (rs % 8 == 0 and r.data_ptr() % 16 == 0))
        name = ("affine_act8" if vec else "affine_act") + ("x" if act > cnn_kernels.ACT["relu6"] else "")
        self.k.launch(name, *linear(p * (c // 8) if vec else p * c), y, ys, s, t, r, rs, p, c, act)
        return y


def _instance_stats(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample, per-channel mean and biased variance over the spatial dims, fp32 `[B, C]` (shifted sums).

    Dense channels-last fp16 rows (or a view of a zero-padded conv output): one `fs.chan_stats8` pass (fp32 per
    thread, fp64 across threads); else chunked torch reductions.
    """
    b, c = y.shape[:2]
    dims = tuple(range(2, y.dim()))
    n = math.prod(y.shape[2:])
    x = getattr(y, "_ff_padded", y)
    cb = x.shape[1]
    if x.dtype == torch.float16 and x.is_contiguous(memory_format=_mf(x.dim())) and cb % 8 == 0 and cb <= STATS_C:
        shift = x[:, :, :1].float().mean(dim=dims).contiguous()
        acc = torch.zeros((b, cb, 2), device=y.device, dtype=torch.float64)
        rows = 256 // (cb // 8)
        grid = (min(STATS_BLOCKS, -(-n // rows)), b, 1)
        fs_kernels.load(y.device).launch("chan_stats8", grid, (256, 1, 1), x, n, cb, shift, acc)
        m = acc[..., 0] / n
        var = (acc[..., 1] / n - m * m).clamp_min(0)
        return (m + shift)[:, :c].float(), var[:, :c].float()
    view = (b, c) + (1,) * len(dims)
    shift = y[:, :, :1].float().mean(dim=dims)
    step = max(1, STATS_CHUNK // max(1, b * c * (n // y.shape[2])))
    s1 = torch.zeros((b, c), device=y.device)
    s2 = torch.zeros((b, c), device=y.device)
    for i in range(0, y.shape[2], step):
        v = y[:, :, i : i + step].float() - shift.view(view)
        s1 += v.sum(dim=dims)
        s2 += v.square().sum(dim=dims)
    m = s1 / n
    return m + shift, (s2 / n - m * m).clamp_min(0)


# ------------------------------------------------------------------------------------------------ module plans
def _act_code(m: nn.Module | None) -> int | None:
    if isinstance(m, nn.GELU):
        return cnn_kernels.ACT["gelu"] if m.approximate == "none" else None
    return cnn_kernels.act_code(m)


def _norm_plan(norm: nn.Module | None, conv: Patched) -> tuple | None:
    """Epilogue plan of `norm` after `conv` (fp32, call before the fp16 cast)."""
    if norm is None or (isinstance(norm, nn.Sequential) and len(norm) == 0):
        return None
    if isinstance(norm, (nn.BatchNorm2d, nn.BatchNorm3d)):
        return ("bn", *cnn_kernels.bn_affine(norm, conv))
    if isinstance(norm, (nn.InstanceNorm2d, nn.InstanceNorm3d)) and not norm.track_running_stats:
        w = b = None
        if norm.affine:
            w, b = norm.weight.float().contiguous(), norm.bias.float().contiguous()
        return ("in", norm.eps, w, b)
    msg = f"no kernels epilogue for {type(norm).__name__}"
    raise NotImplementedError(msg)


class _Convs:
    """The conv / norm leaves of the network as fp16 kernels; one instance per network."""

    def __init__(self, device: torch.device) -> None:
        cnn_kernels.load(device)
        self.ep = _Epilogue(cnn_kernels.kernels(device))
        self.cutlass = fs_cutlass.load(device)

    # -------------------------------------------------------------------------------------------- convs
    def input(self, m: Patched, x: torch.Tensor) -> torch.Tensor:
        """`x` with the zero input channels `m`'s padded weights expect (reuses a padded conv output)."""
        if x.dtype != torch.float16:
            x = x.half()
        cin = getattr(m, "_cin_pad", 0)
        if not cin or x.shape[1] == cin:
            return x
        base = getattr(x, "_ff_padded", None)  # (`_base` is None for views made under inference mode)
        if (
            base is not None
            and base.shape[1] == cin
            and base.shape[0] == x.shape[0]
            and base.shape[2:] == x.shape[2:]
            and base.data_ptr() == x.data_ptr()
        ):
            return base
        z = torch.empty((x.shape[0], cin, *x.shape[2:]), device=x.device, dtype=x.dtype, memory_format=_mf(x.dim()))
        z[:, x.shape[1] :] = 0
        z[:, : x.shape[1]] = x
        return z

    def conv(
        self,
        m: Patched,
        x: torch.Tensor,
        s: torch.Tensor | None,
        t: torch.Tensor | None,
        act: int = 0,
        r: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`act(conv(x) * s + t + r)` (no conv bias: `t` carries it), channels-last, real output channels."""
        x = self.input(m, x)
        if (
            r is None
            and act in {0, 1}
            and getattr(m, "_fuse", False)
            and m.weight.shape[0] == m.out_channels  # the CUTLASS epilogue reads s / t for every padded channel
            and x.dim() == 4
            and x.shape[0] == 1
            and x.is_contiguous(memory_format=cnn_kernels.CL)
        ):
            y = cnn_kernels.conv_fused(m, x, s, t, act)
            if y is not None:
                return y
        y = self._conv3d(m, x, s, t, act, r)
        if y is not None:
            return y
        y = cnn_kernels.conv_raw(m, x)
        c = m.out_channels
        if y.shape[1] != c:
            if r is None and act in {0, 1, 2} and (s is not None or t is not None):  # all padded channels: vector path
                pad = y.shape[1] - c
                s = None if s is None else F.pad(s, (0, pad))
                t = None if t is None else F.pad(t, (0, pad))
                self.ep(y, s, t, act)
                s = t = None
                act = 0
            v = y[:, :c]
            v._ff_padded = y  # the zero-padded buffer, for the next conv's `input`
            y = v
        return self.ep(y, s, t, act, r)

    def _conv3d(
        self,
        m: Patched,
        x: torch.Tensor,
        s: torch.Tensor | None,
        t: torch.Tensor | None,
        act: int,
        r: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Stride-1 3-D conv with a BatchNorm / bias epilogue as one CUTLASS kernel on the zero-padded channels-last
        buffers (the BatchNorm scale folded into the fp16 weights, like the foundationstereo kernels path); None if the
        conv does not qualify."""
        if (
            r is not None
            or x.dim() != _DIM3D
            or not isinstance(m, nn.Conv3d)
            or m.groups != 1
            or tuple(m.stride) != (1, 1, 1)
            or (s is None and t is None)
            or act not in {0, 1}
            or x.dtype != torch.float16
            or m.weight.dtype != torch.float16
        ):
            return None
        kp, cp = m.weight.shape[:2]
        if x.shape[1] != cp or kp % 8 or cp % 8 or kp > C3_MAX_K:
            return None
        long_t = m.weight.shape[2] > C3_MAX_T
        if long_t and (tuple(m.weight.shape[3:]) != (1, 1) or kp != C3_MIN_K):
            return None  # only the (T, 1, 1) disparity convs gain from the slice-major order
        plan = getattr(m, "_c3", None)
        if plan is None:
            w = m.weight.detach().float()
            c = m.out_channels
            if s is not None:
                w = w.clone()
                w[:c] *= s[:c].view(-1, 1, 1, 1, 1)
            b = torch.zeros(kp, device=w.device)
            if t is not None:
                b[:c] = t[:c]
            plan = m._c3 = (w.half().contiguous(memory_format=CL3), b)
        w, b = plan
        if not x.is_contiguous(memory_format=CL3):
            x = x.contiguous(memory_format=CL3)
        pad = cast("tuple[int, int, int]", tuple(m.padding))
        out_sp = [x.shape[2 + i] + 2 * pad[i] - w.shape[2 + i] + 1 for i in range(3)]
        y = torch.empty((x.shape[0], kp, *out_sp), device=x.device, dtype=x.dtype, memory_format=CL3)
        e = FsEpiArgs(mode=2, act=act, b=b.data_ptr(), o=y.data_ptr(), os=kp)
        # taps > 3 along D: consecutive blocks take the same pixel tile of consecutive output slices (cfg 6),
        # so the 17 input slices of a tile are read from L2 (2.7x faster than cuDNN at Midd-H, same results)
        cfg = 6 if long_t else (0 if kp == C3_MIN_K else 3)
        if not self.cutlass.conv3d(cfg, x, w, e, pad, unsupported_ok=True):
            return None
        c = m.out_channels
        if kp == c:
            return y
        v = y[:, :c]
        v._ff_padded = y  # the zero-padded buffer, for the next conv's `input`
        return v

    def apply(self, m: Patched, x: torch.Tensor, r: torch.Tensor | None = None) -> torch.Tensor:
        """Conv `m` with its plan (norm + activation), plus a residual `r` before the activation."""
        plan, act = m._plan, m._act
        if plan is None:
            return self.conv(m, x, None, m._bias32, act, r)
        if plan[0] == "bn":
            return self.conv(m, x, plan[1], plan[2], act, r)
        y = self.conv(m, x, None, None)  # the conv bias cancels in the instance norm
        return self.inorm(y, plan, act, r)

    def inorm(self, y: torch.Tensor, plan: tuple, act: int = 0, r: torch.Tensor | None = None) -> torch.Tensor:
        """In place `act(InstanceNorm(y) + r)` on channels-last rows."""
        _, eps, w, b = plan
        mean, var = _instance_stats(y)
        s = torch.rsqrt(var + eps)
        t = -mean * s
        if w is not None:
            s, t = s * w, t * w + b
        if y.shape[0] == 1:  # (keeps a padded view's buffer for the vector kernel)
            self.ep(y, s[0].contiguous(), t[0].contiguous(), act, r)
            return y
        for i in range(y.shape[0]):
            self.ep(y[i : i + 1], s[i].contiguous(), t[i].contiguous(), act, None if r is None else r[i : i + 1])
        return y

    def norm(self, m: Patched, x: torch.Tensor) -> torch.Tensor:
        """A standalone norm module (out of place)."""
        y = torch.empty_like(x, dtype=torch.float16, memory_format=_mf(x.dim()))
        y.copy_(x)
        if m._plan[0] == "bn":
            return self.ep(y, m._plan[1], m._plan[2], m._act)
        return self.inorm(y, m._plan, m._act)

    # -------------------------------------------------------------------------------------------- patching
    def prepare(self, net: nn.Module, *, pad_deconv: bool = False) -> None:
        """Plans while fp32 (norm affines), then the fp16 cast and the forwards.

        `pad_deconv`: transposed convs get zero output channels up to a multiple of 8 (C-FS-S's 28-channel volumes).
        """
        consumed: set[int] = set()
        mods: list[Patched] = list(net.modules())
        for m in mods:
            if isinstance(m, cnn_kernels.CONVS):
                m._plan, m._act = _norm_plan(None, m), 0
        for m in mods:
            if isinstance(m, Conv):
                m.conv._plan, m.conv._act = _norm_plan(m.norm, m.conv), 2 if m.relu else 0
                consumed.add(id(m.norm))
                override(m, _call_child)
            elif isinstance(m, (ResnetBasicBlock, ResnetBasicBlock3D)):
                n1, n2 = (m.bn1, m.bn2) if m.norm_layer is not None else (None, None)
                m.conv1._plan, m.conv1._act = _norm_plan(n1, m.conv1), 1
                m.conv2._plan, m.conv2._act = _norm_plan(n2, m.conv2), 1
                consumed.update((id(n1), id(n2)))
                override(m, _resnet_forward)
                m._ff = self
            elif isinstance(m, ResidualBlock):
                m.conv1._plan, m.conv1._act = _norm_plan(m.norm1, m.conv1), 1
                m.conv2._plan, m.conv2._act = _norm_plan(m.norm2, m.conv2), 1
                consumed.update((id(m.norm1), id(m.norm2)))
                override(m, _residual_forward)
                m._ff = self
            elif isinstance(m, nn.Sequential):
                _fuse_sequential(m, consumed)
        for m in mods:
            if isinstance(m, NORMS) and id(m) not in consumed:
                m._plan, m._act = _norm_plan(m, None), 0
                override(m, _norm_forward)
                m._ff = self
        cnn_kernels.prepare_convs(net, fuse=True)
        for m in mods:
            if isinstance(m, cnn_kernels.CONVS):
                w = m.weight.data
                cin = w.shape[1]
                if getattr(m, "_hacc", False) and m.groups == 1 and cin >= PAD_IN_MIN and cin % 8:
                    pad = torch.zeros((w.shape[0], 8 - cin % 8, *w.shape[2:]), dtype=w.dtype, device=w.device)
                    m.weight.data = torch.cat([w, pad], 1).contiguous(memory_format=_mf(w.dim()))
                    m._cin_pad = cin + 8 - cin % 8
                k = w.shape[1]
                if pad_deconv and getattr(m, "_hacc_t", False) and m.groups == 1 and k >= PAD_IN_MIN and k % 8:
                    # transposed conv: zero output channels up to a multiple of 8 (vector epilogue, padded next input)
                    pad = torch.zeros((w.shape[0], 8 - k % 8, *w.shape[2:]), dtype=w.dtype, device=w.device)
                    m.weight.data = torch.cat([w, pad], 1).contiguous(memory_format=_mf(w.dim()))
                    if m._bias32 is not None:
                        m._bias32 = torch.cat([m._bias32, m._bias32.new_zeros(pad.shape[1])]).contiguous()
                override(m, _conv_forward)
                m._ff = self


def _fuse_sequential(seq: nn.Sequential, consumed: set[int]) -> None:
    """`conv [-> norm] [-> act]` runs inside `seq`: the conv takes the plan, the other slots become Identity."""
    items: list[tuple[str, Patched]] = list(seq._modules.items())
    i = 0
    while i < len(items):
        _, m = items[i]
        if not isinstance(m, cnn_kernels.CONVS):
            i += 1
            continue
        j = i + 1
        norm = None
        if j < len(items) and isinstance(items[j][1], NORMS):
            norm, j = items[j][1], j + 1
        act = 0
        if j < len(items) and (code := _act_code(items[j][1])):
            act, j = code, j + 1
        if norm is None and not act:
            i += 1
            continue
        m._plan, m._act = _norm_plan(norm, m), act
        consumed.add(id(norm))
        for k in range(i + 1, j):
            seq._modules[items[k][0]] = nn.Identity()
        i = j


def _call_child(self: Patched, x: torch.Tensor) -> torch.Tensor:
    return self.conv(x)  # TAO `Conv`: the conv carries the norm and the LeakyReLU


def _conv_forward(self: Patched, x: torch.Tensor) -> torch.Tensor:
    return self._ff.apply(self, x)


def _norm_forward(self: Patched, x: torch.Tensor) -> torch.Tensor:
    return self._ff.norm(self, x)


def _resnet_forward(self: Patched, x: torch.Tensor) -> torch.Tensor:
    ff: _Convs = self._ff
    out = ff.apply(self.conv1, x)
    identity = x if self.downsample is None else self.downsample(x)
    return ff.apply(self.conv2, out, r=identity)  # relu(bn2(conv2(out)) + identity)


def _residual_forward(self: Patched, x: torch.Tensor) -> torch.Tensor:
    ff: _Convs = self._ff
    y = ff.apply(self.conv2, ff.apply(self.conv1, x))
    if self.downsample is not None:
        x = self.downsample(x)
    return ff.ep(y, act=1, r=x)  # relu(x + relu(norm2(conv2(...))))


# ------------------------------------------------------------------------------------------------ transformer
class _DisparityTransformer:
    """TAO's `CostVolumeDisparityAttention` of C-FS-S ('standard' layout, 28 channels, 4 heads, FFN 28)."""

    def __init__(self, att: Patched, device: torch.device) -> None:
        """Pack the layer parameters (call while fp32)."""
        self.k = cfs_kernels.load(device)
        self.fs = fs_kernels.load(device)
        self.layers = []
        for layer in att.sa:
            a = layer.self_attn
            ok = (
                a.qkv_layout == "standard"
                and a.embed_dim == _TC
                and a.num_heads == 4
                and layer.linear1.out_features == _TC
                and isinstance(layer.act, nn.GELU)
                and layer.act.approximate == "none"
                and layer.norm1.eps == layer.norm2.eps == 1e-5
            )
            if not ok:
                msg = "the cfs kernels implement the C-FS-S disparity transformer only"
                raise NotImplementedError(msg)
            wqkv = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]).float().reshape(-1)
            bqkv = torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias]).float()
            post = [
                a.out_proj.weight,
                a.out_proj.bias,
                layer.norm1.weight,
                layer.norm1.bias,
                layer.linear1.weight,
                layer.linear1.bias,
                layer.linear2.weight,
                layer.linear2.bias,
                layer.norm2.weight,
                layer.norm2.bias,
            ]
            post = torch.cat([p.float().reshape(-1) for p in post])
            self.layers.append((wqkv.contiguous(), bqkv.contiguous(), post.contiguous()))
        self.pe = att.pos_embed0.pe.to(device)  # [1, max_len, 28] fp32 (TAO adds it in the activation dtype)

    def _layers(self, x: torch.Tensor) -> None:
        """The 4 layers in place on `x` `[n, d, 28]` fp16 (contiguous sequences of `d` tokens)."""
        n, d, _ = x.shape
        m = min(n, ATT_CHUNK)
        qkv = torch.zeros((3, m, 4, d, 8), device=x.device, dtype=torch.float16)  # padded head dims stay zero
        o = torch.empty((m, 4, d, 8), device=x.device, dtype=torch.float16)
        for i in range(0, n, ATT_CHUNK):
            xs = x[i : i + ATT_CHUNK]
            ns = xs.shape[0]
            tok = ns * d
            grid = ((min((tok + 3) // 4, 65535 * 8), 1, 1), (128, 1, 1))
            for wqkv, bqkv, post in self.layers:
                self.k.launch("tao_qkv", *grid, xs, tok, d, wqkv, bqkv, qkv[0], qkv[1], qkv[2])
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    for j in range(0, ns, SDPA_CHUNK):
                        q, k, v = (qkv[e, j : min(j + SDPA_CHUNK, ns)] for e in range(3))
                        o[j : j + q.shape[0]] = F.scaled_dot_product_attention(q, k, v, scale=7**-0.5)
                self.k.launch("tao_post", *grid, o, xs, tok, d, post)

    def add_into(self, conv: torch.Tensor, xp: torch.Tensor) -> None:
        """`conv += attention(interpolate(xp, x4, trilinear))` for one sample, in strips of rows.

        TAO's HourGlass tail (`atts_before_upsample=False`): the patch volume `xp` `[28, Ds, Hs, Ws]` upsampled to
        `conv`'s `[28, D, H, W]`, the positional embedding added (fp16), the 4 layers, `conv + x`. Here a strip of rows
        at a time is upsampled (`fs.trilerp4_add`, same fp16 rounding points), transformed and added, so no full-size
        copy of the volume exists besides `conv`.
        """
        c, d, h, w = conv.shape
        ds, hs, ws = xp.shape[1:]
        if c != _TC or (4 * ds, 4 * hs, 4 * ws) != (d, h, w):
            msg = f"patch volume {tuple(xp.shape)} does not upsample to {tuple(conv.shape)}"
            raise RuntimeError(msg)
        if d > self.pe.shape[1]:
            msg = f"x:{(h * w, d, c)}, pe:{tuple(self.pe.shape)}"
            raise RuntimeError(msg)
        src = xp.permute(2, 3, 1, 0).half().contiguous()  # [Hs, Ws, Ds, C]
        pe = self.pe[0, :d].half()
        rows = max(1, min(h, STRIP_SEQ // w))
        y = torch.empty((rows, w, d, c), device=conv.device, dtype=torch.float16)
        dst = conv.permute(2, 3, 1, 0)  # [H, W, D, C] view
        for h0 in range(0, h, rows):
            nh = min(rows, h - h0)
            ys = y[:nh]
            ys.copy_(pe.expand(nh, w, d, c))
            self.fs.launch("trilerp4_add", *linear(nh * w * d * c // 4), src, hs, ws, ds, c, ys, c, h0, nh, w, d)
            self._layers(ys.view(nh * w, d, c))
            dst[h0 : h0 + nh].add_(ys)


# ------------------------------------------------------------------------------------------------ volumes
def _leaves(m: nn.Module) -> list[nn.Module]:
    """The modules a `Sequential` / `Conv3dNormActReduced` chain runs, in order (Identity slots dropped)."""
    if isinstance(m, nn.Identity):
        return []
    if isinstance(m, Conv3dNormActReduced):
        return _leaves(m.conv1) + _leaves(m.conv2)
    if isinstance(m, nn.Sequential):
        return [leaf for child in m for leaf in _leaves(child)]
    return [m]


def _chain(m: nn.Module, box: list[torch.Tensor]) -> torch.Tensor:
    """`m(x)` for the single tensor in `box`, holding no more than one step's input and output at a time."""
    x = box.pop()
    for leaf in _leaves(m):
        x = leaf(x)
    return x


def _gate(fa: Patched, cv: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
    """TAO's `FeatureAtt` (`sigmoid(feat_att(feat)) * cv`) in place on a volume nothing else holds."""
    return cv.mul_(torch.sigmoid(fa.feat_att(feat)).unsqueeze(2))


def _cost_volume(
    fl: torch.Tensor, fr: torch.Tensor, proj: Patched, d: int, groups: int, *, normalize: bool, gemm: bool = False
) -> torch.Tensor:
    """TAO's `cat([build_gwc_volume, build_concat_volume(proj(fl), proj(fr))])`, channels-last 3-D fp16.

    `gemm`: the group-wise correlation of all disparities as one batched matmul per row block (fp32, the diagonal of
    the left-by-right product), instead of one multiply-and-sum per disparity.
    """
    b, c, h, w = fl.shape
    lp, rp = proj(fl), proj(fr)
    cc = lp.shape[1]
    vol = torch.zeros((b, d, h, w, groups + 2 * cc), device=fl.device, dtype=torch.float16)
    a = fl.float().reshape(b, groups, c // groups, h, w)
    z = fr.float().reshape(b, groups, c // groups, h, w)
    if normalize:
        a, z = F.normalize(a, dim=2), F.normalize(z, dim=2)
    lpn, rpn = lp.permute(0, 2, 3, 1), rp.permute(0, 2, 3, 1)
    vol[..., groups : groups + cc] = lpn[:, None]
    n = min(d, w)  # shifts >= W stay zero, as TAO's slices
    if gemm:
        _gwc_gemm(vol, a, z, n)
        for i in range(n):
            vol[:, i, :, i:, groups + cc :] = rpn[:, :, : w - i]
        return vol.permute(0, 4, 1, 2, 3)
    a, z = a.permute(0, 3, 4, 1, 2).contiguous(), z.permute(0, 3, 4, 1, 2).contiguous()  # [B, H, W, G, c/G]
    for i in range(n):
        vol[:, i, :, i:, :groups] = (a[:, :, i:] * z[:, :, : w - i]).sum(-1)
        vol[:, i, :, i:, groups + cc :] = rpn[:, :, : w - i]
    return vol.permute(0, 4, 1, 2, 3)


GWC_ROWS = 16  # image rows per matmul block: bounds the [rows, W, W] product (memory), the result does not depend on it


def _gwc_gemm(vol: torch.Tensor, a: torch.Tensor, z: torch.Tensor, n: int) -> None:
    """Group-wise correlation `sum_c a[x] z[x - d]` for d < n into `vol[..., :G]`; a, z: [B, G, c/G, H, W] fp32."""
    b, g, k, h, w = a.shape
    valid = torch.arange(w, device=a.device).view(1, w) >= torch.arange(n, device=a.device).view(n, 1)
    for r0 in range(0, h, GWC_ROWS):
        r1 = min(h, r0 + GWC_ROWS)
        rows = r1 - r0
        f1 = a[..., r0:r1, :].permute(0, 1, 3, 4, 2).reshape(b * g * rows, w, k)
        f2 = z[..., r0:r1, :].flip(-1).permute(0, 1, 3, 2, 4).reshape(b * g * rows, k, w)
        m = torch.bmm(f1, f2)  # m[x, W-1-x'] = a[x] . z[x']; the disparity-d term is on the strided diagonal
        v = m.as_strided((b * g * rows, n, w), (w * w, 1, w - 1), m.storage_offset() + w - 1)
        vol[:, :n, r0:r1, :, :g] = torch.where(valid, v, 0.0).view(b, g, rows, n, w).permute(0, 3, 2, 4, 1).half()


class _Geometry:
    """TAO's `CombinedGeoEncodingVolume` (2 levels) as one `fs` lookup launch per GRU iteration.

    The volume is copied once into `(B, H, W, D, C)` memory (`fs.geo_lookup_hwd`: the taps along D of a pixel are
    neighbours, and the K + 1 samples of the 2r + 1 taps are loaded once); the lookup writes into a row-padded buffer
    (the motion encoder's input) when one is given.
    """

    def __init__(self, k: CubinModule, fl: torch.Tensor, fr: torch.Tensor, vol: torch.Tensor, r: int) -> None:
        self.k, self.r = k, r
        b, c, d, h, w = vol.shape
        self.c, self.d, self.h, self.w, self.b = c, d, h, w, b
        self.rows = 2 * (c + 1) * (2 * r + 1)
        self.g = vol.half().permute(0, 1, 3, 4, 2).contiguous(memory_format=CL3)  # (B, H, W, D, C) memory
        f1, f2 = F.normalize(fl.float(), dim=1), F.normalize(fr.float(), dim=1)
        self.c0 = torch.einsum("aijk,aijh->ajkh", f1, f2).half().reshape(b * h * w, w).contiguous()
        self.out: torch.Tensor | None = None

    def __call__(self, disp: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """The lookup `[B, ch, H, W]` (channels-last view); into `out` `[P, >= ch]` (row-padded) if given."""
        p = self.b * self.h * self.w
        if out is None:
            if self.out is None:
                self.out = torch.empty((p, self.rows), device=self.g.device, dtype=torch.float16)
            out = self.out
        args = (self.g, self.c, self.d, self.c0, self.w, self.w, disp.contiguous(), out, p, self.r, out.shape[1])
        self.k.launch("geo_lookup_hwd", *linear(p * 2 * (self.c + 1)), *args)
        return out[:, : self.rows].view(self.b, self.h, self.w, self.rows).permute(0, 3, 1, 2)


# ------------------------------------------------------------------------------------------------ update block
def _krsc(*convs: Patched) -> torch.Tensor:
    """The prepared (fp16, channels-last) weights of `convs` stacked as a dense `[K, R, S, C]` filter."""
    return torch.cat([m.weight for m in convs]).permute(0, 2, 3, 1).contiguous()


def _bias(*convs: Patched) -> torch.Tensor:
    return torch.cat([m._bias32 for m in convs]).contiguous()


class _Scale:
    """One `SelectiveConvGRU` scale on dense NHWC buffers (P pixels, hidden 128).

    xin = conv0 input without the context (TAO's `cat(x)[:, 128:]`): the context part of conv0 is the same in every
    step, so it is computed once per forward (`inp0`) and added in conv0's epilogue; qin = [r*h | x0] (both
    RaftConvGRUs' convq input); hx = [x0 | h] (conv1 input); hx2 = conv1 output (convz / convr input); h = the state,
    rewritten in place by the large GRU's q conv. All convs use the large CUTLASS tile (faster here, also at 1/16).
    """

    def __init__(self, gru: Patched, c_in: int) -> None:
        c0, c1 = gru.conv0[0], gru.conv1[0]
        self.c_in, self.c0, self.c1 = c_in - HID, c0.out_channels, c1.out_channels
        w0 = _krsc(c0)
        self.w0_inp, self.w0 = w0[..., :HID].contiguous(), w0[..., HID:].contiguous()
        self.b0, self.zero = _bias(c0), torch.zeros_like(_bias(c0))
        self.w1, self.b1 = _krsc(c1), _bias(c1)
        s, g = gru.small_gru, gru.large_gru
        self.wzr_s, self.bzr_s = _krsc(s.convz, s.convr), _bias(s.convz, s.convr)
        self.wq_s, self.bq_s = _krsc(s.convq), _bias(s.convq)
        self.wzr_l, self.bzr_l = _krsc(g.convz, g.convr), _bias(g.convz, g.convr)
        self.wq_l, self.bq_l = _krsc(g.convq), _bias(g.convq)
        self.k = (g.convq.kernel_size[0], s.convq.kernel_size[0])

    def start(self, lib: fs_cutlass.FsCutlass, h0: torch.Tensor, inp: torch.Tensor, att: torch.Tensor) -> None:
        """Buffers for one forward; the initial state `h0` and the context `inp` (`[1, C, H, W]`)."""
        _, _, self.hw_h, self.hw_w = h0.shape
        p = self.hw_h * self.hw_w
        new = h0.new_empty
        self.xin, self.qin = new((p, self.c_in)), new((p, HID + self.c0))
        self.hx, self.hx2 = new((p, self.c0 + HID)), new((p, self.c1))
        self.z, self.g, self.h = new((p, HID)), new((p, HID)), new((p, HID))
        self.inp0 = new((p, self.c0))
        self.h.copy_(h0.permute(0, 2, 3, 1).reshape(p, HID))
        self.hx[:, self.c0 :] = self.h
        self.att = att.half().contiguous()
        self.cfg = 0  # 128 x 128 tile at every scale (1/16 too: faster than the small-problem tile)
        x = inp.half().permute(0, 2, 3, 1).reshape(p, HID).contiguous()
        e = FsEpiArgs(mode=2, b=self.b0.data_ptr(), o=self.inp0.data_ptr(), os=self.c0)
        _fs_conv(lib, self.cfg, (self.hw_h, self.hw_w), x, self.w0_inp, e)

    def nchw(self) -> torch.Tensor:
        """The state as a channels-last `[1, 128, H, W]` view."""
        return self.h.view(1, self.hw_h, self.hw_w, HID).permute(0, 3, 1, 2)

    def step(self, lib: fs_cutlass.FsCutlass) -> None:
        """`h = small(h, x, hx) * att + large(h, x, hx) * (1 - att)` (TAO `SelectiveConvGRU.forward`)."""
        hw, w2 = (self.hw_h, self.hw_w), 2  # bytes per fp16
        qw, xw = HID + self.c0, self.c0 + HID
        conv = functools.partial(_fs_conv, lib, self.cfg, hw)
        # x0 = relu(conv0(x)) -> qin[:, 128:] and hx[:, :c0]; hx2 = relu(conv1([x0 | h]))
        conv(
            self.xin,
            self.w0,
            FsEpiArgs(
                mode=2,
                act=1,
                b=self.zero.data_ptr(),
                o=self.qin.data_ptr() + w2 * HID,
                os=qw,
                o2=self.hx.data_ptr(),
                o2s=xw,
                c=self.inp0.data_ptr(),
                cs=self.c0,
            ),
        )
        conv(self.hx, self.w1, FsEpiArgs(mode=2, act=1, b=self.b1.data_ptr(), o=self.hx2.data_ptr(), os=self.c1))
        h = self.h.data_ptr()
        for wzr, bzr, wq, bq, mode in (
            (self.wzr_s, self.bzr_s, self.wq_s, self.bq_s, 1),
            (self.wzr_l, self.bzr_l, self.wq_l, self.bq_l, 3),
        ):
            # z -> z; r * h -> qin[:, :128]
            conv(
                self.hx2,
                wzr,
                FsEpiArgs(
                    mode=0, b=bzr.data_ptr(), o=self.z.data_ptr(), os=HID, o2=self.qin.data_ptr(), o2s=qw, h=h, hs=HID
                ),
            )
            e = FsEpiArgs(mode=mode, b=bq.data_ptr(), z=self.z.data_ptr(), zs=HID, h=h, hs=HID)
            if mode == 1:  # small GRU: its new state -> g
                e.o, e.os = self.g.data_ptr(), HID
            else:  # large GRU + the selective blend -> h (in place) and hx[:, c0:] for the next step
                e.o, e.os, e.g, e.gs, e.att = h, HID, self.g.data_ptr(), HID, self.att.data_ptr()
                e.o2, e.o2s = self.hx.data_ptr() + w2 * self.c0, xw
            conv(self.qin, wq, e)


def _fs_conv(
    lib: fs_cutlass.FsCutlass, cfg: int, hw: tuple[int, int], x: torch.Tensor, w: torch.Tensor, e: FsEpiArgs
) -> None:
    """One fs_cutlass conv (stride 1, 'same' padding) of the dense NHWC buffer `x`; results via `e`."""
    lib.conv(cfg, x, w, hw, x.shape[-1], w.shape[0], e)


class _UpdateBlock:
    """TAO's `BasicSelectiveMultiUpdateBlock` of C-FS-S (3 scales, hidden 128) on `fs_cutlass` / `fs` kernels.

    The GRU gates, the selective blend and the motion encoder's concatenations run inside the conv epilogues; the
    pooled / interpolated neighbour states go straight into the next conv's input buffer. The mask head runs once,
    after the last step (only the final mask is used).
    """

    def __init__(self, ub: Patched, device: torch.device) -> None:
        """Pack the EdgeNeXt MLPs while fp32 (call before the fp16 cast); `prepare` takes the conv weights."""
        self.ub, self.k = ub, fs_kernels.load(device)
        self.lib = fs_cutlass.load(device)
        self.edge = []
        for blk in ub.disp_head.conv[2:4]:
            ok = (
                blk.dwconv.kernel_size == (7, 7)
                and isinstance(blk.norm, nn.Identity)
                and blk.pwconv1.weight.shape == (4 * HID, HID)
                and blk.act.approximate == "none"
                and blk.gamma is not None
            )
            if not ok:
                msg = "the fs kernels implement the 128 / 512 EdgeNeXt blocks of C-FS-S only"
                raise NotImplementedError(msg)
            gamma = blk.gamma.detach().float().clone()  # copies: the fp16 cast rewrites the parameters in place
            self.edge.append(
                (
                    blk.dwconv.weight.detach().float()[:, 0].permute(1, 2, 0).contiguous(),  # [7, 7, C]
                    blk.dwconv.bias.detach().float().clone(),
                    blk.pwconv1.weight.detach().half().contiguous(),
                    blk.pwconv1.bias.detach().float().clone(),
                    blk.pwconv2.weight.detach().half().contiguous(),
                    gamma,
                    gamma * blk.pwconv2.bias.detach().float(),
                )
            )

    def prepare(self, convs: _Convs) -> None:
        """Conv weights after the fp16 cast."""
        ub, self._ff = self.ub, convs
        self.scales = [_Scale(ub.gru04, 3 * HID), _Scale(ub.gru08, 3 * HID), _Scale(ub.gru16, 2 * HID)]
        enc, head = ub.encoder, ub.disp_head.conv
        self.enc = enc
        self.wc2, self.bc2 = _krsc(enc.convc2), _bias(enc.convc2)
        self.wd2, self.bd2 = _krsc(enc.convd2), _bias(enc.convd2)
        self.wenc, self.benc = _krsc(enc.conv), _bias(enc.conv)
        self.wh0, self.bh0 = _krsc(head[0]), _bias(head[0])
        self.head = head
        self.c1_in, self.cc2, self.cd2 = enc.convc1.weight.shape[1], enc.convc1.out_channels, self.wd2.shape[0]
        ok = self.wenc.shape[0] == HID and self.cc2 + self.cd2 == self.wenc.shape[-1]
        if not ok:
            msg = "unexpected C-FS-S motion encoder widths"
            raise NotImplementedError(msg)

    def start(self, hidden: list, inp: list, att: list) -> None:
        """Buffers and initial states of one forward (batch 1)."""
        for s, h0, x, a in zip(self.scales, hidden, inp, att, strict=True):
            s.start(self.lib, h0, x, a)
        s0 = self.scales[0]
        p = s0.hw_h * s0.hw_w
        pp = -(-p // 128) * 128  # the MLP kernel reads whole 128-row tiles
        new = s0.h.new_empty
        self.corr = new((p, self.c1_in))
        self.corr[:, :] = 0  # zero pad channels (convc1 has zero weights there)
        self.corr_nchw = self.corr.view(1, s0.hw_h, s0.hw_w, self.c1_in).permute(0, 3, 1, 2)
        self.cd = new((p, self.cc2 + self.cd2))
        self.t0, self.t1, self.u = new((pp, HID)), new((pp, HID)), new((pp, HID))

    def _pool_into(self, src: _Scale, dst: _Scale, c0: int) -> None:
        hd, wd = dst.hw_h, dst.hw_w
        self.k.launch(
            "pool2x_into",
            *linear(hd * wd * HID // 8),
            src.h,
            HID,
            src.hw_h,
            src.hw_w,
            dst.xin.data_ptr() + 2 * c0,
            dst.c_in,
            hd,
            wd,
            HID,
        )

    def _interp_into(self, src: _Scale, dst: _Scale, c0: int) -> None:
        hd, wd = dst.hw_h, dst.hw_w
        self.k.launch(
            "interp_into",
            *linear(hd * wd * HID // 8),
            src.h,
            HID,
            src.hw_h,
            src.hw_w,
            dst.xin.data_ptr() + 2 * c0,
            dst.c_in,
            hd,
            wd,
            HID,
        )

    def __call__(self, geo: _Geometry, disp: torch.Tensor) -> torch.Tensor:
        """One update step on the states; returns the disparity delta `[1, 1, H, W]` fp16 (`disp`: fp32)."""
        s0, s1, s2 = self.scales
        self._pool_into(s1, s2, 0)
        s2.step(self.lib)
        self._pool_into(s0, s1, 0)
        self._interp_into(s2, s1, HID)
        s1.step(self.lib)
        # motion encoder -> s0.xin[:, :128] = [relu(conv([cor | d])) | disp]
        hw, cfg, lib = (s0.hw_h, s0.hw_w), s0.cfg, self.lib
        conv = functools.partial(_fs_conv, lib, cfg, hw)
        p = hw[0] * hw[1]
        geo(disp, self.corr)  # the lookup straight into the zero-padded convc1 input
        cor1 = self._ff.conv(self.enc.convc1, self.corr_nchw, None, self.enc.convc1._bias32, 1)  # 1x1: cuDNN
        cor1 = cor1.permute(0, 2, 3, 1).reshape(p, -1)
        cdw = self.cc2 + self.cd2
        conv(cor1, self.wc2, FsEpiArgs(mode=2, act=1, b=self.bc2.data_ptr(), o=self.cd.data_ptr(), os=cdw))
        d1 = self._ff.conv(self.enc.convd1, disp.half(), None, self.enc.convd1._bias32, 1)
        d1 = d1.permute(0, 2, 3, 1).reshape(p, -1)
        conv(d1, self.wd2, FsEpiArgs(mode=2, act=1, b=self.bd2.data_ptr(), o=self.cd.data_ptr() + 2 * self.cc2, os=cdw))
        conv(
            self.cd,
            self.wenc,
            FsEpiArgs(mode=2, act=1, b=self.benc.data_ptr(), o=s0.xin.data_ptr(), os=s0.c_in, disp=disp.data_ptr()),
        )
        self._interp_into(s1, s0, HID)
        s0.step(lib)
        # disp head: relu(conv) -> 2 x EdgeNeXt -> conv
        conv(s0.h, self.wh0, FsEpiArgs(mode=2, act=1, b=self.bh0.data_ptr(), o=self.t0.data_ptr(), os=HID))
        x, y = self.t0, self.t1
        grid = ((hw[1] + 31) // 32, (hw[0] + 15) // 16, HID // 8), (8, 16, 1)
        for wd, bd, w1, b1, w2, s2_, t2 in self.edge:
            self.k.launch("dwconv7_nhwc", *grid, x, wd, bd, self.u, hw[0], hw[1], HID)
            self.k.launch(
                "mlp_gelu_res", (self.u.shape[0] // 128, 1, 1), (128, 1, 1), self.u, w1, b1, w2, s2_, t2, x, y, p
            )
            x, y = y, x
        x = x[:p].view(1, *hw, HID).permute(0, 3, 1, 2)
        return self._ff.conv(self.head[4], x, None, self.head[4]._bias32, 0)  # 1 output channel: cuDNN

    def mask(self) -> torch.Tensor:
        """TAO's `0.25 * mask(net[0])` of the current state."""
        return 0.25 * self.ub.mask(self.scales[0].nchw())


# ------------------------------------------------------------------------------------------------ forward
class KernelsForward:
    """TAO's test-mode forward (`iters` GRU steps, upsampled disparity) on the kernels leaves."""

    def __init__(self, net: Patched, device: torch.device) -> None:
        """Prepare `net` (fp32, eval, CPU) for `device`."""
        self.fs = isinstance(net, FoundationStereo)
        net = net.to(device).eval()
        self.args = net.args
        self.geo_k = fs_kernels.load(device)
        convs = _Convs(device)
        self.att = self.update = self.ffs = None
        net.spx_gru[0] = cnn_kernels.pad_deconv_out(net.spx_gru[0])  # 9 superpixel weights: 16 channels, no transform
        if not self.fs:
            self.ffs = FfsForward(net, device)  # C-FFS: the ffs_cutlass forward, planned from the fp32 weights
        if self.fs:
            self.att = _DisparityTransformer(net.cost_agg.atts["cost_vol_disp"], device)
            self.update = _UpdateBlock(net.update_block, device)
        else:
            prepare_portable(net)  # C-FFS: 12-step sequences, torch (chunked)
        convs.prepare(net, pad_deconv=self.fs)
        if self.update is not None:
            self.update.prepare(convs)
        if self.fs:
            use_fast_vit(net.feature.dino.depth_anything.pretrained)
        self.net = net

    def _volume(self, box: list[torch.Tensor], fl: list[torch.Tensor]) -> torch.Tensor:
        """C-FS-S: `corr_stem`, `corr_feature_att` and TAO's `HourGlass.forward` on the cost volume in `box`.

        The same ops in the same order; each full-size volume is dropped as soon as it is consumed, the feature gates
        multiply in place and the attention tail adds into the `conv1_up` output strip by strip (`add_into`).
        """
        net, att = self.net, self.att
        hg = net.cost_agg
        x = _chain(net.corr_stem, box)
        _gate(net.corr_feature_att, x, fl[0])
        xp = hg.conv_patch(x)
        box.append(x)
        del x
        c1 = _gate(hg.feature_att_8, _chain(hg.conv1, box), fl[1])
        c2 = _gate(hg.feature_att_16, hg.conv2(c1), fl[2])
        c3 = _gate(hg.feature_att_32, hg.conv3(c2), fl[3])
        c2 = _gate(hg.feature_att_up_16, hg.agg_0(torch.cat((hg.conv3_up(c3), c2), dim=1)), fl[2])
        del c3
        c1 = _gate(hg.feature_att_up_8, hg.agg_1(torch.cat((hg.conv2_up(c2), c1), dim=1)), fl[1])
        del c2
        conv = hg.conv1_up(c1)
        del c1
        if att is None:
            raise RuntimeError
        for i in range(conv.shape[0]):
            att.add_into(conv[i], xp[i])
        del xp
        return _chain(hg.conv_out, [conv])

    @torch.inference_mode()
    def __call__(self, left: torch.Tensor, right: torch.Tensor, iters: int) -> torch.Tensor:
        """Disparity `[B, H, W]` fp32 of a scaled, padded pair."""
        if self.ffs is not None:
            return self.ffs(left, right, iters)
        net, a = self.net, self.args
        b = left.shape[0]
        left, right = left.half(), right.half()
        out, vit_feat = net.feature(torch.cat([left, right]))
        fl, fr = [t[:b] for t in out], [t[b:] for t in out]
        stem_2x = net.stem_2(left)
        d4 = a.max_disparity // 4
        vol = _cost_volume(
            fl[0],
            fr[0],
            net.proj_cmb,
            d4,
            net.cv_group,
            normalize=a.get("gwc_feature_normalize", True),
            gemm=not self.fs,
        )
        if self.fs:
            box = [vol]
            del vol
            vol = self._volume(box, fl)
        else:
            vol = net.cost_agg(net._run_corr_feature_att(vol, fl[0]), fl, atts_before_upsample=True)
        prob = F.softmax(net.classifier(vol).squeeze(1).float(), dim=1)
        disp = utils.disparity_regression(prob, d4)
        if self.fs:
            cnet = net.cnet(left, vit_feat=vit_feat[:b], num_layers=a.n_gru_layers)
        else:
            cnet = net.cnet(fl[0], fl[1], fl[2])
        hidden = [torch.tanh(x[0]) for x in cnet]
        inp = [torch.relu(x[1]) for x in cnet]
        inp = [cam(net.cam, x) * x for x in inp]
        att = [net.sam(x) for x in inp]
        geo = _Geometry(self.geo_k, fl[0], fr[0], vol, a.corr_radius)
        del vol, out, fl, fr, cnet
        mask = None
        if self.update is not None and b == 1:
            upd = self.update
            upd.start(hidden, inp, att)
            for _ in range(iters):
                disp = disp + upd(geo, disp.contiguous()).float()
            mask = upd.mask()
        else:
            for _ in range(iters):
                hidden, mask, delta = net.update_block(hidden, inp, geo(disp), disp.half(), att)
                disp = disp + delta.float()
        spx = F.softmax(net.spx_gru(net.spx_2_gru(mask, stem_2x)).float(), 1)
        return utils.context_upsample(disp * 4.0, spx)
