"""C-FFS kernels forward: TAO's Fast-FoundationStereo test-mode forward on the `ffs_cutlass` kernels.

The hot path is explicit (plans built once from the fp32 modules, before the rest of the network is cast):

- features: EdgeNeXt stem, downsampling and `ConvBlock`s as fused CUTLASS convs (LayerNorm / GELU / layer scale +
  residual in the kernels); `SplitTransposeBlock`s and the decoder (`Conv2xDownScale`) stay modules.
- cost volume: group-wise correlation and concatenation in one kernel, straight into the channels-last 3-D buffer.
- cost aggregation: every 3-D conv with its BatchNorm folded, bias, activation and residual in one CUTLASS conv; the
  28-channel volumes zero-padded to 32 (exact); stride-2 transposed convs as 8 parity-class 2x2x2 convs writing
  into their output voxels; the trilinear x4 upsample fused with its sum.
- classifier: the 3-tap disparity conv as one 2-D conv (the taps as output channels) + soft-argmax in one kernel.
- GRU: persistent zero-padded buffers, convs writing into channel slices of their concatenations, one geometry
  lookup kernel (both levels; level 1 pooled on the fly), two gate kernels, the mask head only after the last
  iteration, and the context upsample in one kernel.

Context net, `stem_2`, the attention maps, the disparity transformer and the upsample convs stay the network's
modules (prepared by `inference.KernelsForward`). Unsupported module trees raise; there is no fallback.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.patch import is_base
from fastforks.kernels import ffs_cutlass as ffs_kernels
from fastforks.kernels.ffs_cutlass import (
    ACT_GELU,
    ACT_LEAKY,
    ACT_NONE,
    ACT_RELU,
    ACT_SIGMOID,
    ACT_TANH,
    stem_c8_weight,
)
from fastforks.shared.edgenext.net.edgenext import ConvBlock, LayerNorm2d, SplitTransposeBlock

from ..net.fast_foundation_stereo.fast_foundation_stereo import _FFSForwardHelper, _FFSPostForwardHelper
from ..net.foundation_stereo.convolution_helper import Conv, Conv3dNormActReduced, ResnetBasicBlock3D
from ..net.foundation_stereo.submodule import CostVolumeDisparityAttention, FeatureAtt

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks.kernels.ffs_cutlass import FfsCutlass

CL, CL3 = torch.channels_last, torch.channels_last_3d
CORR_LEVELS = 2  # the lookup kernel's pyramid
CFG_WIDE = 7  # 128 x 128 tiles: wide GRU convs (measured best at KITTI and Midd-H alike, gru_cfg)
CFG_HX = 8
CFG_MLP = 9  # 128 x 64: EdgeNeXt fc1
CFG_HEAD = 13  # the 8-channel disparity head

_Box: TypeAlias = "torch.Tensor | list[torch.Tensor]"  # a tensor, or one in a box (see `_take`)


class _Step(Protocol):
    def __call__(self, x: _Box, feat: torch.Tensor | None = None, /) -> torch.Tensor: ...


class _Post(Protocol):
    def __call__(self, high: _Box, low: _Box, feat: torch.Tensor | None = None, /) -> torch.Tensor: ...


def p8(c: int) -> int:
    return (c + 7) // 8 * 8


def _unsupported(what: str) -> RuntimeError:
    return RuntimeError(f"c-fast-foundationstereo kernels: {what} is not supported by the ffs_cutlass forward")


def _wb(m: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(weight `[K, C, ...]` fp32, bias fp32 or None) of a conv or linear."""
    w = m.weight.detach().float()
    if w.dim() == 2:
        w = w[:, :, None, None]
    return w, None if m.bias is None else m.bias.detach().float()


def _bn(bn: nn.Module | None, k: int, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Eval BatchNorm as `(scale, shift)`; identity for None."""
    if bn is None or isinstance(bn, nn.Identity):
        return torch.ones(k, device=dev), torch.zeros(k, device=dev)
    if not isinstance(bn, (nn.BatchNorm2d, nn.BatchNorm3d)) or not bn.track_running_stats:
        raise _unsupported(f"norm {type(bn).__name__}")
    running_var, running_mean = cast("torch.Tensor", bn.running_var), cast("torch.Tensor", bn.running_mean)
    s = bn.weight.detach().float() / torch.sqrt(running_var.float() + bn.eps)
    return s, bn.bias.detach().float() - running_mean.float() * s


def _take(x: _Box) -> torch.Tensor:
    """A tensor, or the one in a box (emptied, so the callee holds the last reference)."""
    return x.pop() if isinstance(x, list) else x


def _act(m: nn.Module) -> int | None:
    if isinstance(m, nn.ReLU):
        return ACT_RELU
    if isinstance(m, nn.LeakyReLU) and m.negative_slope == 0.01:
        return ACT_LEAKY
    return None


def cam(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """`ChannelAttentionEnhancement(x)` with the global mean / max as plain reductions, both through one `fc` call.

    The adaptive pools run a generic kernel that takes ~1.5 ms on the 1/4-size context features.
    """
    b = x.shape[0]
    y = m.fc(torch.cat((x.mean((2, 3), keepdim=True), x.amax((2, 3), keepdim=True))))
    return torch.sigmoid(y[:b] + y[b:])


class _PConv:
    """2-D conv on zero-padded channels. `srcs`: `((w, b), o0, n_out, out_dst)`; `ins`: `(i0, n_in, in_dst)`."""

    def __init__(
        self,
        k: FfsCutlass,
        srcs: list[tuple[tuple[torch.Tensor, torch.Tensor | None], int, int, int]],
        ins: list[tuple[int, int, int]],
        cin: int,
        kout: int,
        pad: tuple[int, int],
        *,
        stride: tuple[int, int] = (1, 1),
        act: int = ACT_NONE,
        scale: torch.Tensor | None = None,
        cfg: int = -1,
    ) -> None:
        w0 = srcs[0][0][0]
        wt = w0.new_zeros((kout, cin, *w0.shape[2:]))
        bt = w0.new_zeros(kout)
        for (w, b), o0, no, od in srcs:
            if scale is not None:
                w = w * scale.view(-1, 1, 1, 1)
                b = None if b is None else b * scale
            for i0, ni, idst in ins:
                wt[od : od + no, idst : idst + ni] = w[o0 : o0 + no, i0 : i0 + ni]
            if b is not None:
                bt[od : od + no] = b[o0 : o0 + no]
        self.k, self.pad, self.stride, self.act, self.cfg = k, pad, stride, act, cfg
        self.w = wt.half().contiguous(memory_format=CL)
        self.b = bt.half()
        self.c8 = None
        if cin == 8 and kout <= 16 and stride == (1, 1) and cfg < 0:
            self.c8 = stem_c8_weight(wt.half())

    @classmethod
    def of(
        cls, k: FfsCutlass, wb: tuple[torch.Tensor, torch.Tensor | None], pad: tuple[int, int], *,
        cin: int | None = None, kout: int | None = None, **kw: torch.Tensor | int,
    ) -> _PConv:  # fmt: skip
        """A single conv `[K, C, R, S]` on `cin` / `kout` (default: padded to 8) channels."""
        n_out, n_in = wb[0].shape[:2]
        return cls(k, [(wb, 0, n_out, 0)], [(0, n_in, 0)], cin or p8(n_in), kout or p8(n_out), pad, **kw)

    def __call__(
        self, x: torch.Tensor, res: torch.Tensor | None = None, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.c8 is not None and res is None and out is None:
            return self.k.stem_c8(
                x,
                self.c8,
                self.b,
                self.w.shape[0],
                cast("tuple[int, int]", tuple(self.w.shape[2:])),
                self.pad,
                self.act,
            )
        k = self.k
        return k.conv2d(
            x, self.w, self.b, pad=self.pad, stride=self.stride, act=self.act, res=res, out=out, cfg=self.cfg
        )


class _C3:
    """3-D conv + folded BatchNorm + bias + activation (+ residual), channels zero-padded to 8."""

    def __init__(
        self, k: FfsCutlass, conv: nn.Module, bn: nn.Module | None, act: int,
    ) -> None:  # fmt: skip
        if (
            not is_base(conv, nn.Conv3d)
            or conv.groups != 1
            or conv.padding_mode != "zeros"
            or conv.dilation != (1, 1, 1)
        ):
            raise _unsupported(f"3-D conv {conv}")
        w, b = _wb(conv)
        s, t = _bn(bn, w.shape[0], w.device)
        w = w * s.view(-1, 1, 1, 1, 1)
        b = t if b is None else b * s + t
        kp, cp = p8(w.shape[0]), p8(w.shape[1])
        wt = w.new_zeros((kp, cp, *w.shape[2:]))
        wt[: w.shape[0], : w.shape[1]] = w
        bt = w.new_zeros(kp)
        bt[: w.shape[0]] = b
        self.k, self.act = k, act
        self.pad = cast("tuple[int, int, int]", tuple(conv.padding))
        self.stride = cast("tuple[int, int, int]", tuple(conv.stride))
        self.w, self.b = wt.half().contiguous(memory_format=CL3), bt.half()

    def __call__(
        self, x: torch.Tensor, res: torch.Tensor | None = None, gate: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.k.conv3d(x, self.w, self.b, pad=self.pad, stride=self.stride, act=self.act, res=res, gate=gate)


class _Sub3:
    """Stride-2 ConvTranspose3d (k 4 / p 1, or k 3 / p 1 / output_padding 1) + folded BatchNorm + activation as 8
    parity-class 2x2x2 convs, each writing straight into its output voxels (sub-pixel decomposition, same sums)."""

    def __init__(self, k: FfsCutlass, conv: nn.ConvTranspose3d, bn: nn.Module | None, act: int) -> None:
        ks, op = tuple(conv.kernel_size), tuple(conv.output_padding)
        if (
            not is_base(conv, nn.ConvTranspose3d)
            or conv.groups != 1
            or conv.stride != (2, 2, 2)
            or conv.padding != (1, 1, 1)
            or conv.dilation != (1, 1, 1)
            or any(not ((kk == 4 and o == 0) or (kk == 3 and o == 1)) for kk, o in zip(ks, op, strict=True))
        ):
            raise _unsupported(f"transposed 3-D conv {conv}")
        w = conv.weight.detach().float()  # [Cin, Cout, kd, kh, kw]
        cin, cout = w.shape[:2]
        s, t = _bn(bn, cout, w.device)
        w = w * s.view(1, -1, 1, 1, 1)
        b = t if conv.bias is None else conv.bias.detach().float() * s + t
        kp, cp = p8(cout), p8(cin)
        self.subs = []
        for rd in (0, 1):
            for rh in (0, 1):
                for rw in (0, 1):
                    ws = w.new_zeros((kp, cp, 2, 2, 2))
                    for jd in (0, 1):
                        for jh in (0, 1):
                            for jw in (0, 1):
                                kd, kh, kw = 3 - rd - 2 * jd, 3 - rh - 2 * jh, 3 - rw - 2 * jw  # 2 - r - 2 j + pad
                                if kd < ks[0] and kh < ks[1] and kw < ks[2] and min(kd, kh, kw) >= 0:
                                    ws[:cout, :cin, jd, jh, jw] = w[:, :, kd, kh, kw].t()
                    self.subs.append((ws.half().contiguous(memory_format=CL3), (-rd, -rh, -rw)))
        bt = w.new_zeros(kp)
        bt[:cout] = b
        self.k, self.act, self.ks, self.op, self.kp = k, act, ks, op, kp
        self.b = bt.half()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        n, _, *sp = x.shape
        size = [(i - 1) * 2 - 2 + kk + o for i, kk, o in zip(sp, self.ks, self.op, strict=True)]
        y = torch.empty((n, self.kp, *size), device=x.device, dtype=x.dtype, memory_format=CL3)
        for w, off in self.subs:
            self.k.conv3d_subpixel(x, w, self.b, self.act, y, off)
        return y


class _Gate:
    """TAO's `FeatureAtt`: `cv * sigmoid(feat_att(feat))`, in place (every caller hands over a fresh volume)."""

    def __init__(self, k: FfsCutlass, fa: FeatureAtt) -> None:
        seq = fa.feat_att
        if not (
            isinstance(seq, nn.Sequential)
            and len(seq) == 2
            and isinstance(seq[0], Conv)
            and is_base(seq[0].conv, nn.Conv2d)
            and is_base(seq[1], nn.Conv2d)
            and seq[0].conv.kernel_size == seq[1].kernel_size == (1, 1)
        ):
            raise _unsupported("FeatureAtt layout")
        w, b = _wb(seq[0].conv)
        s, t = _bn(seq[0].norm, w.shape[0], w.device)
        wb0 = (w * s.view(-1, 1, 1, 1), t if b is None else b * s + t)
        self.c0 = _PConv.of(k, wb0, (0, 0), act=ACT_LEAKY if seq[0].relu else ACT_NONE)
        self.c1 = _PConv.of(k, _wb(seq[1]), (0, 0), act=ACT_SIGMOID)

    def gate(self, feat: torch.Tensor | None) -> torch.Tensor:
        """`sigmoid(feat_att(feat))` `[B, C (padded), H, W]` channels-last."""
        if feat is None:
            raise _unsupported("FeatureAtt without a feature")
        return self.c1(self.c0(feat))

    def __call__(self, cv: torch.Tensor, feat: torch.Tensor | None) -> torch.Tensor:
        return cv.mul_(self.gate(feat).unsqueeze(2))


class _Attention:
    """The disparity transformer module on the real 28 channels, re-padded to the volume's channels."""

    def __init__(self, m: CostVolumeDisparityAttention, c: int) -> None:
        self.m, self.c = m, c

    def __call__(self, x: torch.Tensor, _feat: torch.Tensor | None = None) -> torch.Tensor:
        y = self.m(x[:, : self.c].contiguous(memory_format=CL3))
        return F.pad(y.permute(0, 2, 3, 4, 1), (0, x.shape[1] - self.c)).permute(0, 4, 1, 2, 3)


class _Cost:
    """Compiles the cost-volume modules (`corr_feature_att`, the FFS hourglass) into kernel steps."""

    def __init__(self, k: FfsCutlass, c: int) -> None:
        self.k, self.c = k, c

    def _seq_c3(self, mods: list[nn.Module]) -> _C3 | None:
        """`mods` (Identity skipped) as one fused conv when they are exactly `Conv3d [-> BatchNorm3d] [-> act]`."""
        mods = [m for m in mods if not isinstance(m, nn.Identity)]
        if not mods or not is_base(mods[0], nn.Conv3d):
            return None
        j, bn, act = 1, None, ACT_NONE
        if j < len(mods) and isinstance(mods[j], nn.BatchNorm3d):
            bn, j = mods[j], j + 1
        if j < len(mods) and (a := _act(mods[j])) is not None:
            act, j = a, j + 1
        return _C3(self.k, mods[0], bn, act) if j == len(mods) else None

    def _tail(self, m: nn.Module) -> tuple[Callable | None, _C3] | None:
        """`m` as `(steps before, final 3-D conv)` when it ends in one 3-D conv (a following gate fuses into it)."""
        if isinstance(m, Conv) and is_base(m.conv, nn.Conv3d):
            return None, _C3(self.k, m.conv, m.norm, ACT_LEAKY if m.relu else ACT_NONE)
        if isinstance(m, Conv3dNormActReduced) and isinstance(m.conv2, nn.Sequential):
            c3 = self._seq_c3(list(m.conv2))
            if c3 is not None:
                return self.chain([m.conv1]), c3
        return None

    @staticmethod
    def _run(t: tuple[Callable | None, _C3], gate: _Gate | None = None) -> Callable:
        pre, c3 = t

        def run(x: torch.Tensor | list[torch.Tensor], f: torch.Tensor | None = None) -> torch.Tensor:
            y = _take(x) if pre is None else pre(x, f)
            return c3(y, gate=None if gate is None else gate.gate(f))

        return run

    def step(self, m: nn.Module) -> Callable[..., torch.Tensor]:
        k = self.k
        if (t := self._tail(m)) is not None:
            return self._run(t)
        if isinstance(m, Conv):
            act = ACT_LEAKY if m.relu else ACT_NONE
            if is_base(m.conv, nn.ConvTranspose3d):
                f = _Sub3(k, m.conv, m.norm, act)
                return lambda x, _f=None: f(x)
            raise _unsupported(f"Conv {m.conv}")
        if isinstance(m, Conv3dNormActReduced):
            return self.chain([m.conv1, m.conv2])
        if isinstance(m, nn.Sequential):
            return self.chain(list(m))
        if isinstance(m, _FFSForwardHelper):
            return self.chain(list(m.layers))
        if isinstance(m, ResnetBasicBlock3D):
            if m.downsample is not None or m.stride != 1:
                raise _unsupported("strided ResnetBasicBlock3D")
            n1, n2 = (m.bn1, m.bn2) if m.norm_layer is not None else (None, None)
            c1, c2 = _C3(k, m.conv1, n1, ACT_RELU), _C3(k, m.conv2, n2, ACT_RELU)
            return lambda x, _f=None: c2(c1(x), res=x.contiguous(memory_format=CL3))
        if isinstance(m, FeatureAtt):
            return _Gate(k, m)
        if isinstance(m, CostVolumeDisparityAttention):
            return _Attention(m, self.c)
        if isinstance(m, _FFSPostForwardHelper):
            return self.post(m)
        raise _unsupported(f"module {type(m).__name__}")

    def chain(self, mods: list[nn.Module]) -> _Step:
        """Steps of a module list; `Conv3d [-> BatchNorm3d] [-> ReLU / LeakyReLU]` runs fuse into one conv, and a
        `FeatureAtt` right after a step that ends in a 3-D conv multiplies in that conv's epilogue."""
        steps: list[Callable] = []
        tail: tuple[Callable | None, _C3] | None = None
        i = 0
        while i < len(mods):
            m = mods[i]
            if isinstance(m, nn.Identity):
                i += 1
                continue
            if is_base(m, nn.Conv3d):
                j, bn, act = i + 1, None, ACT_NONE
                if j < len(mods) and isinstance(mods[j], nn.BatchNorm3d):
                    bn, j = mods[j], j + 1
                if j < len(mods) and (a := _act(mods[j])) is not None:
                    act, j = a, j + 1
                tail = (None, _C3(self.k, m, bn, act))
                steps.append(self._run(tail))
                i = j
                continue
            if isinstance(m, FeatureAtt) and tail is not None:
                ga = _Gate(self.k, m)
                if ga.c1.w.shape[0] == tail[1].w.shape[0]:
                    steps[-1] = self._run(tail, ga)
                    tail = None
                    i += 1
                    continue
            tail = self._tail(m)
            steps.append(self.step(m) if tail is None else self._run(tail))
            i += 1

        def run(x: _Box, feat: torch.Tensor | None = None) -> torch.Tensor:
            x = _take(x)
            for s in steps:
                x = s(x, feat)
            return x

        return run

    def post(self, m: _FFSPostForwardHelper) -> _Post:
        """`out(upsample(conv_low) + conv_high)`; a trailing trilinear `nn.Upsample` is fused with the sum."""
        if m.op != "sum":
            raise _unsupported(f"post-forward op {m.op!r}")
        ups = list(m.upsample)
        tri = None
        if ups and isinstance(ups[-1], nn.Upsample):
            u = ups.pop()
            if u.mode != "trilinear" or u.align_corners or u.size is not None:
                raise _unsupported(f"upsample {u}")
            sf = u.scale_factor if isinstance(u.scale_factor, tuple) else (u.scale_factor,)
            tri = float(sf[0])
            if any(float(s) != tri for s in sf):
                raise _unsupported(f"upsample {u}")
        up, out = self.chain(ups), self.chain(list(m.out))
        k = self.k

        def run(high: _Box, low: _Box, feat: torch.Tensor | None = None) -> torch.Tensor:
            y = up(low)
            if tri is None:
                box = [y.add_(_take(high))]
            else:
                high = _take(high)
                if any(o != i * tri for o, i in zip(high.shape[2:], y.shape[2:], strict=True)):
                    msg = f"c-fast-foundationstereo kernels: volume {tuple(high.shape)} is no x{tri:g} multiple"
                    raise RuntimeError(msg)
                box = [k.up3_add(y, high.contiguous(memory_format=CL3))]
                del high
            del y
            return out(box, feat)

        return run


class FfsForward:
    """The FFS hot path (plans from the fp32 network; build before the network is cast and patched)."""

    def __init__(self, net: nn.Module, device: torch.device) -> None:
        k = self.k = ffs_kernels.load(device)
        self.net = net
        a = net.args
        if a.n_gru_layers != 1 or a.corr_levels != CORR_LEVELS:
            raise _unsupported(f"{a.n_gru_layers} GRU layers / {a.corr_levels} correlation levels")
        self.k_taps = 2 * a.corr_radius + 1
        self.c = c = net.volume_dim
        self.groups = net.cv_group
        self.normalize = bool(a.get("gwc_feature_normalize", True))
        self.d4 = a.max_disparity // 4
        self._features(net.feature)
        cc = net.proj_cmb.out_channels
        self.cc = cc
        self.proj = _PConv.of(k, _wb(net.proj_cmb), (0, 0))
        cost = _Cost(k, c)
        self.corr_att = cost.chain(list(net.corr_feature_att))
        hg = net.cost_agg
        self.fa8, self.fa16 = cost.step(hg.feature_att_8), cost.step(hg.feature_att_16)
        self.conv3, self.fa32 = cost.step(hg.conv3), cost.step(hg.feature_att_32)
        self.p32, self.p16, self.p4 = cost.post(hg.post32_to_16), cost.post(hg.post16_to_8), cost.post(hg.post8_to_4)
        self.conv1_up = cost.step(hg.conv1_up)
        cls = net.classifier[0]
        if (
            not is_base(cls, nn.Conv3d)
            or cls.out_channels != 1
            or tuple(cls.kernel_size) != (3, 3, 3)
            or len(net.classifier) != 1
        ):
            raise _unsupported("classifier layout")
        wc = cls.weight.detach().float()
        self.cls = _PConv(k, [((wc[:, :, i], None), 0, 1, i) for i in range(3)], [(0, c, 0)], p8(c), 8, (1, 1))
        self.cls_b = float(cls.bias.detach().float()[0]) if cls.bias is not None else 0.0
        self._update(net.update_block)

    # -------------------------------------------------------------------------------------------- plans
    def _features(self, f: nn.Module) -> None:
        k = self.k
        if f.use_vit_backbone or not is_base(f.conv4, nn.Conv2d):
            raise _unsupported("feature extractor with a ViT branch")
        stem = f.stem
        if not (len(stem) == 2 and is_base(stem[0], nn.Conv2d) and isinstance(stem[1], LayerNorm2d)):
            raise _unsupported("feature stem")
        if stem[0].in_channels > 8 or stem[0].padding != (0, 0):
            raise _unsupported("feature stem conv")
        self.stem = _PConv.of(k, _wb(stem[0]), (0, 0), cin=8, stride=tuple(stem[0].stride))
        self.stem_ln = self._ln(stem[1])
        self.stages = []
        for st in f.stages:
            ds = None
            if not isinstance(st.downsample, nn.Identity):
                ln, cv = st.downsample
                if not isinstance(ln, LayerNorm2d) or not is_base(cv, nn.Conv2d) or cv.padding != (0, 0):
                    raise _unsupported("EdgeNeXt downsample")
                ds = (self._ln(ln), _PConv.of(k, _wb(cv), (0, 0), stride=tuple(cv.stride)))
            blocks = []
            for blk in st.blocks:
                if isinstance(blk, SplitTransposeBlock):
                    blocks.append(blk)
                elif isinstance(blk, ConvBlock):
                    blocks.append(self._block(blk))
                else:
                    raise _unsupported(f"EdgeNeXt block {type(blk).__name__}")
            self.stages.append((ds, blocks))
        self.deconvs = (f.deconv32_16, f.deconv16_8, f.deconv8_4)
        self.conv4 = _PConv.of(k, _wb(f.conv4), (0, 0))

    @staticmethod
    def _ln(m: nn.LayerNorm) -> tuple[torch.Tensor, torch.Tensor, float]:
        if m.weight is None or m.bias is None or m.normalized_shape[0] % 8:
            raise _unsupported("LayerNorm without affine / 8 | C")
        return m.weight.detach().half(), m.bias.detach().half(), m.eps

    def _block(self, blk: ConvBlock) -> tuple:
        dw = blk.conv_dw
        c = dw.in_channels
        e = blk.mlp.fc1.out_features
        if (
            dw.groups != c
            or dw.stride != (1, 1)
            or dw.padding != (dw.kernel_size[0] // 2, dw.kernel_size[1] // 2)
            or c % 8
            or e % 8
            or not isinstance(blk.mlp.act, nn.GELU)
            or blk.mlp.act.approximate != "none"
        ):
            raise _unsupported("EdgeNeXt ConvBlock")
        w = dw.weight.detach()[:, 0].permute(1, 2, 0).contiguous().half()
        b = (dw.bias.detach() if dw.bias is not None else torch.zeros(c, device=w.device)).half()
        fc1 = _PConv.of(self.k, _wb(blk.mlp.fc1), (0, 0), act=ACT_GELU, cfg=CFG_MLP)
        fc2 = _PConv.of(self.k, _wb(blk.mlp.fc2), (0, 0), scale=blk.gamma.detach().float())
        return (w, b, dw.padding, self._ln(blk.norm), fc1, fc2)

    def _update(self, ub: nn.Module) -> None:
        k = self.k
        enc, gru, seq = ub.encoder, ub.gru04, ub.disp_head.conv
        c, kt = self.c, self.k_taps
        hd = gru.small_gru.convz.out_channels
        cx = gru.conv0[0].out_channels
        c_inp = self.net.cam.fc[0].in_channels
        c_mo = enc.conv.out_channels
        self.hd, self.cx, self.c_inp, self.c_mo = hd, cx, c_inp, c_mo
        self.hdp, self.cxp = p8(hd), p8(cx)
        if c_inp % 8 or c_mo % 8 or gru.small_gru.convz.kernel_size != (1, 1):
            raise _unsupported("GRU widths")
        # convc1 input: TAO's per-level order [geo c * K + k | corr k] -> the kernel's [geo l0 (k * C + c) | geo l1 |
        # corr l0 | corr l1]
        g = c * kt
        wc1, bc1 = _wb(enc.convc1)
        if wc1.shape[1] != CORR_LEVELS * (g + kt):
            raise _unsupported("motion encoder input")
        perm, base = [], 0
        for _ in range(CORR_LEVELS):
            perm += [base + ci * kt + j for j in range(kt) for ci in range(c)]
            base += g
            perm += [base + j for j in range(kt)]
            base += kt
        self.ccorr = p8(wc1.shape[1])
        ins = [(0, g, 0), (g + kt, g, g), (g, kt, 2 * g), (2 * g + kt, kt, 2 * g + kt)]
        n1 = wc1.shape[0]
        self.convc1 = _PConv(k, [((wc1[:, perm], bc1), 0, n1, 0)], ins, self.ccorr, p8(n1), (0, 0), act=ACT_RELU)
        c2, d2 = enc.convc2.out_channels, enc.convd2.out_channels
        self.c2p, self.cenc = p8(c2), p8(c2) + p8(d2)
        self.convc2 = _PConv.of(k, _wb(enc.convc2), (1, 1), act=ACT_RELU, cfg=CFG_WIDE)
        self.convd1 = _PConv.of(k, _wb(enc.convd1), tuple(enc.convd1.padding), cin=8, act=ACT_RELU)
        self.convd2 = _PConv.of(k, _wb(enc.convd2), tuple(enc.convd2.padding), act=ACT_RELU)
        self.conv = _PConv(
            k, [(_wb(enc.conv), 0, c_mo, 0)], [(0, c2, 0), (c2, d2, p8(c2))], self.cenc, c_mo, (1, 1), act=ACT_RELU
        )
        self.cmf = p8(c_inp + c_mo + 1)  # motion features [inp | encoder | disp | 0]
        self.conv0 = _PConv.of(k, _wb(gru.conv0[0]), (1, 1), cin=self.cmf, act=ACT_RELU, cfg=CFG_HX)
        c1o = gru.conv1[0].out_channels
        self.chx, self.chx2 = self.cxp + self.hdp, p8(c1o)  # hx = [x | h]
        self.conv1 = _PConv(
            k, [(_wb(gru.conv1[0]), 0, c1o, 0)], [(0, cx, 0), (cx, hd, self.cxp)], self.chx, self.chx2, (1, 1),
            act=ACT_RELU,
        )  # fmt: skip
        zr_in = [(0, gru.small_gru.convz.in_channels, 0)]
        hp = self.hdp

        def zr(m: nn.Module, cfg: int = -1) -> _PConv:
            pad = tuple(m.convz.padding)
            srcs = [(_wb(m.convz), 0, hd, 0), (_wb(m.convr), 0, hd, hp)]
            return _PConv(k, srcs, zr_in, self.chx2, 2 * hp, pad, act=ACT_SIGMOID, cfg=cfg)

        self.zr_s, self.zr_l = zr(gru.small_gru, CFG_WIDE), zr(gru.large_gru)
        q_in = [(0, hd, 0), (hd, cx, hp)]

        def q(m: nn.Module) -> _PConv:
            return _PConv(k, [(_wb(m.convq), 0, hd, 0)], q_in, hp + self.cxp, hp, tuple(m.convq.padding), act=ACT_TANH)

        self.q_s, self.q_l = q(gru.small_gru), q(gru.large_gru)
        c36 = seq[0].out_channels
        cdh = p8(c36)
        self.dh0 = _PConv.of(k, _wb(seq[0]), (1, 1), cin=hp, act=ACT_RELU)
        self.enx = []
        for blk in (seq[2], seq[3]):
            if not isinstance(blk.norm, nn.Identity) or blk.dwconv.groups != c36:
                raise _unsupported("disparity head block")
            r, s = blk.dwconv.kernel_size
            dw = torch.zeros((r, s, cdh), device=blk.dwconv.weight.device)
            dw[:, :, :c36] = blk.dwconv.weight.detach().float()[:, 0].permute(1, 2, 0)
            dwb = torch.zeros(cdh, device=dw.device)
            if blk.dwconv.bias is not None:
                dwb[:c36] = blk.dwconv.bias.detach().float()
            gam = blk.gamma.detach().float() if blk.gamma is not None else None
            pw1 = _PConv.of(k, _wb(blk.pwconv1), (0, 0), cin=cdh, act=ACT_GELU, cfg=CFG_WIDE)
            pw2 = _PConv.of(k, _wb(blk.pwconv2), (0, 0), kout=cdh, scale=gam)
            self.enx.append((dw.half().contiguous(), dwb.half(), tuple(blk.dwconv.padding), pw1, pw2))
        self.dh_out = _PConv.of(k, _wb(seq[4]), (1, 1), cin=cdh, kout=8, cfg=CFG_HEAD)

    # -------------------------------------------------------------------------------------------- forward
    def _ln_apply(self, x: torch.Tensor, ln: tuple) -> torch.Tensor:
        """LayerNorm over the channels of a channels-last `[N, C, H, W]` tensor."""
        n, c, h, w = x.shape
        y = self.k.ln_rows(x.permute(0, 2, 3, 1).reshape(-1, c), *ln)
        return y.view(n, h, w, c).permute(0, 3, 1, 2)

    def features(self, img: torch.Tensor) -> list[torch.Tensor]:
        k = self.k
        x = self._ln_apply(self.stem(k.input_nhwc8(img)), self.stem_ln)
        outs = []
        for ds, blocks in self.stages:
            if ds is not None:
                x = ds[1](self._ln_apply(x, ds[0]))
            for blk in blocks:
                if isinstance(blk, nn.Module):
                    x = blk(x).contiguous(memory_format=CL)
                    continue
                w, b, pad, ln, fc1, fc2 = blk
                t = self._ln_apply(k.dwconv(x, w, b, pad), ln)
                x = fc2(fc1(t), res=x)
            outs.append(x)
        x4, x8, x16, x32 = outs
        d32, d16, d8 = self.deconvs
        x16 = d32(x32, x16).contiguous(memory_format=CL)
        x8 = d16(x16, x8).contiguous(memory_format=CL)
        x4 = d8(x8, x4).contiguous(memory_format=CL)
        return [self.conv4(x4), x8, x16, x32]

    def _hourglass(self, box: list[torch.Tensor], f: list[torch.Tensor]) -> torch.Tensor:
        """TAO's `_FFSHourGlass.forward`; full-size volumes are handed over in boxes and dropped once consumed."""
        x = box.pop()
        c1 = self.fa8(x, f[1])
        c2 = self.fa16(c1, f[2])
        c3 = self.fa32(self.conv3(c2), f[3])
        hb, lb = [c2], [c3]
        del c2, c3
        c2 = self.p32(hb, lb, f[2])
        hb, lb = [c1], [c2]
        del c1, c2
        c1 = self.p16(hb, lb, f[1])
        lb = [self.conv1_up(c1)]
        del c1
        hb = [x]
        del x
        return self.p4(hb, lb)

    @torch.inference_mode()
    def __call__(self, left: torch.Tensor, right: torch.Tensor, iters: int) -> torch.Tensor:
        """Disparity `[B, H, W]` fp32 of a scaled, padded (normalized) pair."""
        net, k = self.net, self.k
        b = left.shape[0]
        fl, fr = self.features(left), self.features(right)
        stem_2x = net.stem_2(left.half())
        d4 = self.d4
        _, _, h, w = fl[0].shape
        if d4 % 2 or w % 2:
            msg = f"c-fast-foundationstereo kernels: the geometry lookup needs an even disparity range and width ({d4}, {w})"
            raise RuntimeError(msg)
        lp, rp = (self.proj(f[0]).permute(0, 2, 3, 1)[..., : self.cc].contiguous() for f in (fl, fr))
        nl, nr = fl[0].permute(0, 2, 3, 1), fr[0].permute(0, 2, 3, 1)
        box = [k.build_volume(nl, nr, lp, rp, d4, self.groups, normalize=self.normalize)]
        del lp, rp
        box = [self.corr_att(box, fl[0])]
        vol = self._hourglass(box, fl).contiguous(memory_format=CL3)
        cp = vol.shape[1]
        g0 = vol.permute(0, 2, 3, 4, 1)
        o = self.cls(g0.reshape(b * d4, h, w, cp).permute(0, 3, 1, 2))
        disp = k.cls_regress(o, b, d4, self.cls_b)
        del o
        hc, inp = net.cnet(fl[0])[0]
        hc, inp = torch.tanh(hc), torch.relu(inp)
        inp = cam(net.cam, inp) * inp
        att = net.sam(inp)
        cf = fl[0].shape[1]
        f1 = k.ln_rows(nl.reshape(-1, cf), None, None, 1e-12).view(b * h, w, cf)
        f2 = k.ln_rows(nr.reshape(-1, cf), None, None, 1e-12).view(b * h, w, cf)
        c0 = torch.bmm(f1, f2.transpose(1, 2)).view(b, h, w, w)
        del f1, f2, fl, fr, nl, nr
        p = b * h * w
        st = self._state(p, hc, inp, att)
        for _ in range(iters):
            k.geo_lookup(
                g0, c0, disp, st["corr"], st["d8"], st["mf"], k=self.k_taps, cw=self.c, mfoff=self.c_inp + self.c_mo
            )
            disp = (disp + self._step(st, b, h, w)).contiguous()
        del g0, vol, c0
        hv = st["h"].view(b, h, w, -1)[..., : self.hd].permute(0, 3, 1, 2).contiguous(memory_format=CL)
        mask = 0.25 * net.update_block.mask(hv)
        logits = net.spx_gru(net.spx_2_gru(mask, stem_2x))
        if logits.dtype != torch.float16:
            logits = logits.half()
        return k.ctx_up(logits, disp, 4.0)

    def _state(self, p: int, hc: torch.Tensor, inp: torch.Tensor, att: torch.Tensor) -> dict[str, torch.Tensor]:
        dev = hc.device

        def z(c: int) -> torch.Tensor:
            return torch.zeros((p, c), device=dev, dtype=torch.float16)

        hp, cxp = self.hdp, self.cxp
        st = {
            "corr": z(self.ccorr), "d8": z(8), "enc": z(self.cenc), "mf": z(self.cmf), "hx": z(self.chx), "h": z(hp),
            "qs": z(hp + cxp), "ql": z(hp + cxp), "att": att.reshape(p).half().contiguous(),
        }  # fmt: skip
        st["h"][:, : self.hd] = hc.permute(0, 2, 3, 1).reshape(p, self.hd)
        st["hx"][:, cxp : cxp + self.hd] = st["h"][:, : self.hd]
        st["mf"][:, : self.c_inp] = inp.permute(0, 2, 3, 1).reshape(p, self.c_inp)
        return st

    def _step(self, st: dict[str, torch.Tensor], b: int, h: int, w: int) -> torch.Tensor:
        """One GRU iteration on the persistent buffers; the disparity delta `[B, H, W]` fp32."""
        k = self.k

        def nc(buf: torch.Tensor, c0: int | None = None, c1: int | None = None) -> torch.Tensor:
            v = buf.view(b, h, w, -1)
            return (v if c0 is None else v[..., c0:c1]).permute(0, 3, 1, 2)

        c1 = self.convc1(nc(st["corr"]))
        self.convc2(c1, out=nc(st["enc"], 0, self.c2p))
        self.convd2(self.convd1(nc(st["d8"])), out=nc(st["enc"], self.c2p, self.cenc))
        self.conv(nc(st["enc"]), out=nc(st["mf"], self.c_inp, self.c_inp + self.c_mo))
        self.conv0(nc(st["mf"]), out=nc(st["hx"], 0, self.cxp))
        hx2 = self.conv1(nc(st["hx"]))
        hp = self.hdp
        zrs = self.zr_s(hx2).permute(0, 2, 3, 1).reshape(-1, 2 * hp)
        zrl = self.zr_l(hx2).permute(0, 2, 3, 1).reshape(-1, 2 * hp)
        k.gru_rh(zrs, zrl, st["h"], st["hx"], st["qs"], st["ql"], self.cxp)
        qs = self.q_s(nc(st["qs"])).permute(0, 2, 3, 1).reshape(-1, hp)
        ql = self.q_l(nc(st["ql"])).permute(0, 2, 3, 1).reshape(-1, hp)
        k.gru_out(zrs, zrl, qs, ql, st["att"], st["h"], st["hx"], self.cxp)
        t = self.dh0(nc(st["h"]))
        for dw, dwb, pad, pw1, pw2 in self.enx:
            t = pw2(pw1(k.dwconv(t, dw, dwb, pad)), res=t)
        return self.dh_out(t)[:, 0].float()
