"""HITNet kernels path: fp16-accumulate fused convolutions and fused cost kernels (any size, nothing compiled per size).

The same network and weights as `net/hitnet.py`; what changes:
  - convolutions with 8 | C: `hitnet_conv` (CUTLASS implicit GEMM, fp16 accumulation, dilation supported) with
    bias, residual and LeakyReLU(0.2) in the epilogue; output channels are zero-padded to a multiple of 8 (sliced off
    afterwards). The 13-channel hypothesis convolution stays on cuDNN under autocast.
  - the full-resolution head: the first feature convolution reads the raw images and does the padding, gray
    conversion and scaling itself (`img_conv3`); the 2x2 transposed convolutions + LeakyReLU write their cropped
    output straight into the skip concatenation (`deconv2x2`).
  - the level-init cost volume never exists: cost and argmin in one pass (`init_cost`); tile SAD features and the
    refinement warp SAD are computed straight from the NHWC fp16 features (`tile_sad`, `ref_sad`).
Hypotheses, disparities, argmin decisions and warp positions stay float32, as in the portable path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import nn

from fastforks.kernels import hitnet, hitnet_conv

from .net.hitnet import HYP, HITNet, LevelInit, LevelProp, Refinement, ResBlock, ResNet, upsample_hyp

if TYPE_CHECKING:
    from fastforks._core.kernels import CubinModule

BLOCK = 256  # fixed: threads per block of the element-wise kernels
IMG_MAXC = 32  # img_conv3: at most this many output channels (hitnet_kernels.cu IMG_MAXC)
CAT_PARTS = 4  # cat_cl: at most this many parts (hitnet_kernels.cu CAT_PARTS)
DECONV_SMEM = 96 << 10  # deconv2x2: its fp16 weights (4 C K halves) live in shared memory


class FusedConv:
    """A convolution on `hitnet_conv`: fp16 weights (output channels zero-padded to a multiple of 8) and bias."""

    def __init__(self, conv: nn.Conv2d, lib: hitnet_conv.HitnetConv, *, out: int | None = None) -> None:
        """Take the weights of `conv` (of its first `out` output channels).

        Raises:
            RuntimeError: `conv` has no bias or a channel count the kernel cannot take (C % 8 != 0).
        """
        if conv.bias is None or conv.in_channels % 8:
            msg = f"hitnet_conv: needs a biased convolution with 8 | C, got {conv}"
            raise RuntimeError(msg)
        w, b = conv.weight.detach()[:out], conv.bias.detach()[:out]
        self.k = w.shape[0]
        kp = -(-self.k // 8) * 8
        if kp != self.k:
            w = torch.cat((w, w.new_zeros(kp - self.k, *w.shape[1:])), 0)
            b = torch.cat((b, b.new_zeros(kp - self.k)), 0)
        self.w = w.half().contiguous(memory_format=torch.channels_last)
        self.b = b.half().contiguous()
        self.lib = lib
        self.padding: tuple[int, int] = conv.padding  # ty: ignore[invalid-assignment] - an int pair for these layers
        self.stride: tuple[int, int] = conv.stride  # ty: ignore[invalid-assignment]
        self.dilation: tuple[int, int] = conv.dilation  # ty: ignore[invalid-assignment]

    def __call__(
        self,
        x: torch.Tensor,
        *,
        leaky: bool = True,
        res: torch.Tensor | None = None,
        padding: tuple[int, int] | None = None,
        stride: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """`lrelu(conv(x) + res)` (`leaky`) or `conv(x) + res`, float16, channels-last."""
        pad = self.padding if padding is None else padding
        st = self.stride if stride is None else stride
        x = x.half().contiguous(memory_format=torch.channels_last)
        n, _, h, wd = x.shape
        r, s = self.w.shape[2:]
        p = (h + 2 * pad[0] - self.dilation[0] * (r - 1) - 1) // st[0] + 1
        q = (wd + 2 * pad[1] - self.dilation[1] * (s - 1) - 1) // st[1] + 1
        y = torch.empty(
            n, self.w.shape[0], p, q, device=x.device, dtype=torch.float16, memory_format=torch.channels_last
        )
        rr = None if res is None else res.half().contiguous(memory_format=torch.channels_last)
        self.lib.conv(
            x, self.w, self.b, rr, y, pad, st, self.dilation, hitnet_conv.ACT_LEAKY if leaky else hitnet_conv.ACT_NONE
        )
        return y if self.k == y.shape[1] else y[:, : self.k]


def _nhwc(t: torch.Tensor) -> torch.Tensor:
    """`[n, c, h, w]` (any memory format) -> packed `[n, h, w, c]`."""
    return t.permute(0, 2, 3, 1).contiguous()


def _need(t: torch.Tensor, dtype: torch.dtype, what: str, channels: int | None = None) -> None:
    """Raise unless `t` is contiguous `dtype` (with `channels` in the last dimension): what the kernels read."""
    if t.dtype != dtype or not t.is_contiguous() or (channels is not None and t.shape[-1] != channels):
        msg = f"hitnet {what}: needs contiguous {dtype} [..., {channels}], got {t.dtype} {tuple(t.shape)}"
        raise RuntimeError(msg)


def _cat_cl(k: CubinModule, parts: list[torch.Tensor]) -> torch.Tensor:
    """`torch.cat(parts, 1)` (up to four float16 / float32 parts, any layout) as float16 channels-last, one pass."""
    n, _, h, w = parts[0].shape
    if len(parts) > CAT_PARTS or any(
        p.dtype not in (torch.float16, torch.float32) or p.shape[0] != n or p.shape[2:] != (h, w) for p in parts
    ):
        msg = f"hitnet cat_cl: parts {[(tuple(p.shape), p.dtype) for p in parts]}"
        raise RuntimeError(msg)
    out = _empty_cl(parts[0].device, n, sum(p.shape[1] for p in parts), h, w)
    args: list = []
    for p in [*parts, *[None] * (CAT_PARTS - len(parts))]:
        if p is None:
            args += [None, 0, 0, 0, 0, 0, 0]
        else:
            args += [p, int(p.dtype == torch.float32), p.shape[1], *p.stride()]
    k.launch("cat_cl", *_grid(n * h * w), out, n, h, w, out.shape[1], *args)
    return out


def _vec8(t: torch.Tensor) -> bool:
    """True if `t` (fp16 [n, c, h, w]) can be read / written 8 channels at a time: channel stride 1, aligned."""
    return t.stride(1) == 1 and t.data_ptr() % 16 == 0 and all(t.stride(d) % 8 == 0 for d in (0, 2, 3))


def _empty_cl(device: torch.device, n: int, c: int, h: int, w: int) -> torch.Tensor:
    return torch.empty(n, c, h, w, device=device, dtype=torch.float16, memory_format=torch.channels_last)


def _grid(total: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return ((total + BLOCK - 1) // BLOCK, 1, 1), (BLOCK, 1, 1)


class KernelsHITNet:
    """The network of a `HITNet` on the fused kernels."""

    def __init__(self, net: HITNet, device: torch.device) -> None:
        """Load the kernel libraries for `device` (raises `KernelUnavailableError` when there is no binary)."""
        self.net = net
        self.k: CubinModule = hitnet.load(device)
        self.lib = hitnet_conv.load(device)
        self._conv: dict[int, FusedConv] = {}
        self._first: dict[int, FusedConv] = {}
        fe = net.fe
        c0 = cast("nn.Conv2d", fe.conv_down_1[0])
        if (
            c0.kernel_size != (3, 3)
            or c0.padding != (1, 1)
            or c0.stride != (1, 1)
            or c0.dilation != (1, 1)
            or c0.groups != 1
            or c0.bias is None
            or c0.out_channels > IMG_MAXC
        ):
            msg = f"hitnet img_conv3: needs a biased 3x3 / pad 1 conv with <= {IMG_MAXC} outputs, got {c0}"
            raise RuntimeError(msg)
        self.w0, self.b0 = c0.weight.detach().float().contiguous(), c0.bias.detach().float().contiguous()
        self.up: list[tuple[torch.Tensor, torch.Tensor]] = []  # deconv2x2 fp16 weights [2, 2, C, K], biases
        for m in fe.conv_upsample:
            d = cast("nn.ConvTranspose2d", m)
            if d.kernel_size != (2, 2) or d.stride != (2, 2) or d.padding != (0, 0) or d.groups != 1 or d.bias is None:
                msg = f"hitnet deconv2x2: needs a biased 2x2 / stride 2 transposed conv, got {d}"
                raise RuntimeError(msg)
            if d.in_channels % 8 or d.out_channels % 8 or 8 * d.in_channels * d.out_channels > DECONV_SMEM:
                msg = f"hitnet deconv2x2: needs multiples of 8 channels and 8 C K <= {DECONV_SMEM} bytes, got {d}"
                raise RuntimeError(msg)
            w = d.weight.detach().half().permute(2, 3, 0, 1).contiguous()
            self.up.append((w, d.bias.detach().float().contiguous()))

    def fc(self, conv: nn.Module) -> FusedConv:
        """The fused wrapper of the convolution `conv` (built once)."""
        conv = cast("nn.Conv2d", conv)
        f = self._conv.get(id(conv))
        if f is None:
            f = self._conv[id(conv)] = FusedConv(conv, self.lib)
        return f

    def fc_first(self, conv: nn.Module) -> FusedConv:
        """The fused wrapper of `conv` cut to its first output channel (built once)."""
        f = self._first.get(id(conv))
        if f is None:
            f = self._first[id(conv)] = FusedConv(cast("nn.Conv2d", conv), self.lib, out=1)
        return f

    def resnet(self, net: nn.Module, x: torch.Tensor, *, first: bool = False) -> torch.Tensor:
        """Fused `ResNet`; `first`: only the first output channel."""
        net = cast("ResNet", net)
        x = self.fc(net.conv1)(x)
        for r in net.res:
            block = cast("ResBlock", r)
            x = self.fc(block.conv2)(self.fc(block.conv1)(x), res=x)
        return (self.fc_first if first else self.fc)(net.convn)(x, leaky=False)

    def deconv_into(self, i: int, x: torch.Tensor, out: torch.Tensor, skip: torch.Tensor | None = None) -> None:
        """`out[:, :K] = lrelu(conv_upsample[i](x))` cropped to `out`, `out[:, K:] = skip` (channels-last buffers)."""
        w, b = self.up[i]
        x = x.contiguous(memory_format=torch.channels_last)
        n, c, h, wd = x.shape
        k, ho, wo = w.shape[3], out.shape[2], out.shape[3]
        cs = 0 if skip is None else skip.shape[1]
        if (
            x.dtype != torch.float16
            or c != w.shape[2]
            or out.shape[1] < k + cs
            or ho > 2 * h
            or wo > 2 * wd
            or not all(_vec8(t) for t in (out, *(() if skip is None else (skip,))))
            or (skip is not None and (skip.shape[2:] != out.shape[2:] or cs % 8))
        ):
            msg = f"hitnet deconv2x2: input {tuple(x.shape)} {x.dtype}, output {tuple(out.shape)} {out.stride()}"
            raise RuntimeError(msg)
        ss = (0, 0, 0) if skip is None else (skip.stride(0), skip.stride(2), skip.stride(3))
        self.k.launch(
            "deconv2x2", *_grid(n * -(-ho // 2) * -(-wo // 2) * (k + cs) // 8), x, n, h, wd, c, w, b, k, out,
            out.stride(0), out.stride(2), out.stride(3), ho, wo, skip, cs, *ss, smem=w.numel() * 2,
        )  # fmt: skip

    def features(
        self, left: torch.Tensor, right: torch.Tensor, size: tuple[int, int], pads: tuple[float, float]
    ) -> list[torch.Tensor]:
        """Feature U-Net of both images (padded to `size` with `pads`, 0..255), coarse to fine."""
        fe = self.net.fe
        n, _, h, w = left.shape
        hp, wp = size
        k0, ku = self.w0.shape[0], self.up[0][0].shape[3]
        x = _empty_cl(left.device, 2 * n, k0, hp, wp)
        cat0 = _empty_cl(left.device, 2 * n, ku + k0, hp, wp)  # [upsampled, skip]: the skip is written here directly
        sk = cat0[:, ku:]
        self.k.launch(
            "img_conv3", *_grid(2 * n * hp * wp), left.float().contiguous(), right.float().contiguous(),
            n, h, w, hp, wp, pads[0], pads[1], self.net.in_ch, self.w0, self.b0, k0, x,
            sk, sk.stride(0), sk.stride(2), sk.stride(3),
        )  # fmt: skip
        skips = [cat0]
        for i in range(fe.n):
            x = self.fc(fe.conv_down[i])(x)
            x = self.fc(fe.conv_down_1[i + 1])(x)
            skips.append(x)
        x = self.fc(fe.conv_up_2n)(self.fc(fe.conv_up_1n)(x))
        outs = [x]
        for i in range(fe.n - 1, -1, -1):
            s = skips[i]
            if i == 0:
                buf = s
            else:
                ku = self.up[i][0].shape[3]
                buf = _empty_cl(s.device, s.shape[0], ku + s.shape[1], s.shape[2], s.shape[3])
            self.deconv_into(i, x, buf, None if i == 0 else s)
            x = self.fc(fe.conv_up_0[i])(buf)
            x = self.fc(fe.conv_up_2[i])(self.fc(fe.conv_up_1[i])(x))
            outs.append(x)
        return outs

    def init_cost(self, lt: torch.Tensor, rt: torch.Tensor, max_disp: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Best disparity and its cost (first minimum) from the tile embeddings, no cost volume."""
        n, _, h, w = lt.shape
        lt_, rt_ = _nhwc(lt), _nhwc(rt)
        _need(lt_, torch.float16, "init_cost", HYP)
        _need(rt_, torch.float16, "init_cost", HYP)
        d = torch.empty(n, 1, h, w, device=lt.device, dtype=torch.float32)
        c = torch.empty_like(d)
        self.k.launch("init_cost_f16", *_grid(n * h * w), lt_, rt_, n, h, w, rt.shape[3], max_disp, d, c)
        return d, c

    def tile_sad(self, hyp: torch.Tensor, fl_: torch.Tensor, fr_: torch.Tensor) -> torch.Tensor:
        """`[n, 64, h, w]` tile costs of a hypothesis `[n, 16, h, w]` on NHWC features at 4x its resolution."""
        n, _, h, w = hyp.shape
        hyp = hyp.contiguous()
        _need(hyp, torch.float32, "tile_sad")
        _need(fl_, torch.float16, "tile_sad", fr_.shape[3])
        if fl_.shape != fr_.shape or fl_.shape[1:3] != (4 * h, 4 * w):
            msg = f"hitnet tile_sad: features {tuple(fl_.shape)} / {tuple(fr_.shape)} for hypotheses {(h, w)}"
            raise RuntimeError(msg)
        out = torch.empty(n, 64, h, w, device=hyp.device, dtype=torch.float32)
        self.k.launch(
            "tile_sad_f16", *_grid(n * 16 * h * w), hyp, hyp.stride(0), hyp.stride(1), fl_, fr_,
            n, h, w, fl_.shape[3], out, out.stride(0),
        )  # fmt: skip
        return out

    def ref_sad(self, d: torch.Tensor, fl_: torch.Tensor, fr_: torch.Tensor, scale: float) -> torch.Tensor:
        """`[n, 3, H, W]` SAD of NHWC features against the right features warped by `d * scale`, `+ 1`, `- 1`."""
        n, h, w, c = fl_.shape
        d = d.contiguous()
        _need(d, torch.float32, "ref_sad")
        _need(fl_, torch.float16, "ref_sad", fr_.shape[3])
        if fl_.shape != fr_.shape or d.shape[2:] != (h, w):
            msg = f"hitnet ref_sad: features {tuple(fl_.shape)} / {tuple(fr_.shape)}, disparity {tuple(d.shape)}"
            raise RuntimeError(msg)
        out = torch.empty(n, 3, h, w, device=d.device, dtype=torch.float32)
        self.k.launch(
            "ref_sad_f16", *_grid(n * h * w), d, d.stride(0), fl_, fr_, n, h, w, c, scale, out
        )  # fmt: skip
        return out

    def init(self, k: int, fl: torch.Tensor, fr: torch.Tensor, ref: torch.Tensor | None) -> torch.Tensor:
        """First hypothesis of level `k`."""
        mod = cast("LevelInit", self.net.init[k])
        red, emb = self.fc(mod.conv_reduce), self.fc(mod.conv_embedding)
        lt = emb(red(fl, padding=(0, 0), stride=(4, 4)))
        rt = emb(red(F.pad(fr, (1, 2, 0, 0)), padding=(0, 0), stride=(4, 1)))
        d, cmin = self.init_cost(lt, rt, mod.max_disp)
        with torch.autocast("cuda", dtype=torch.float16):
            return mod.hypothesis(d, cmin, lt, ref)

    def prop(self, k: int, hyps: list[torch.Tensor], fl_: torch.Tensor, fr_: torch.Tensor) -> torch.Tensor:
        """Propagated hypothesis of level `k`."""
        mod = cast("LevelProp", self.net.prop[k])
        parts = []
        for hp in hyps:
            parts += [hp, self.fc(mod.conv_neighbors)(self.tile_sad(hp, fl_, fr_))]
        x = self.resnet(mod.prop, _cat_cl(self.k, parts))
        return mod.select(hyps, x)

    def refine(self, r: int, hyp: torch.Tensor, feats: tuple[torch.Tensor, ...], *, last: bool = False) -> torch.Tensor:
        """Refinement stage `r` on `feats` = (left NCHW, right NHWC, left NHWC); `last`: only the disparity channel."""
        fl, fr_, fl_ = feats
        mod = cast("Refinement", self.net.refine[r])
        parts = [fl, hyp]
        if mod.sad_scale is not None:  # WarpSadAndReplace
            parts = [fl[:, :-3], self.ref_sad(hyp[:, 0:1], fl_, fr_, mod.sad_scale), hyp]
        x = self.fc(mod.conv1x1)(_cat_cl(self.k, parts))
        if last:
            return hyp[:, 0:1] + self.resnet(mod.refiner, x, first=True).float()
        return hyp + self.resnet(mod.refiner, x).float()

    def __call__(
        self, left: torch.Tensor, right: torch.Tensor, size: tuple[int, int], pads: tuple[float, float]
    ) -> torch.Tensor:
        """Disparity `[N, 1, H, W]` of images `[N, 3, h, w]` (0..255).

        The images are bottom / right padded to `size` = (H, W) with the constants `pads` (left, right; 0..255) on
        the fly, by the first convolution.
        """
        net = self.net
        with torch.autocast("cuda", dtype=torch.float16):
            feats = self.features(left, right, size, pads)
            n = left.shape[0]
            fl, fr = [f[:n] for f in feats], [f[n:] for f in feats]
            fl_, fr_ = [_nhwc(f) for f in fl], [_nhwc(f) for f in fr]
            hyp = torch.empty(0)
            for k in range(len(net.init)):
                hi = self.init(k, fl[k], fr[k], None if k < 2 else fl[k - 2])  # noqa: PLR2004 - fixed: first two levels
                hyp = self.prop(k, [hi] if k == 0 else [hi, upsample_hyp(hyp, 2.0, 2.0)], fl_[k], fr_[k])
            base = len(net.init) - 3  # fixed: refinement r works at scale 2 - r
            for r in range(len(net.refine)):
                last = r == len(net.refine) - 1
                hyp = self.refine(r, hyp, (fl[base + r], fr_[base + r], fl_[base + r]), last=last)
                if not last:
                    hyp = upsample_hyp(hyp, 1.0, 0.5)
        return hyp[:, 0:1]
