"""NAFNet runners: the network of megvii-research/NAFNet and its test call (`net(lq)`, clamped to `[0, 1]`).

The network pads the input with zeros to a multiple of 16 itself and crops its output (`check_image_size`). The
deblurring models are `NAFNetLocal` (test-time local converter, TLC): every channel attention averages over a window of
1.5 x the training size at its level (upstream: `AvgPool2d` with the window found by one forward pass at the training
size on construction, sums from a cumulative sum, replicate-padded); here the windows are computed from the level, the
box sums are float32 as upstream's. Each block's channel LayerNorm runs as `F.layer_norm` over the channel dimension
of a channels-last view (upstream: per-pixel mean / variance as an autograd function); its affine and the block's
residual scales (`beta`, `gamma`) are folded into the 1 x 1 convs next to them at load. The residual stream runs
scaled by 1 / 16 and the channel attention by 1 / 64 (exact: weights and norm epsilons scaled at load), as float16
overflows them on small inputs (upstream's `.half()` too). On CUDA the network runs channels-last.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks._core.errors import WeightsError
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import is_base, override
from fastforks.kernels import cudnn_hacc
from fastforks.restoration._runner import Forward, Runner

from .net import NAFNet
from .net.NAFNet_arch import NAFBlock

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks.restoration._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params` (the test options' `network_g`)."""

    scale: int  # 1
    width: int  # channels at full resolution
    enc_blocks: tuple[int, ...]  # NAFBlocks per encoder level
    middle_blocks: int
    dec_blocks: tuple[int, ...]  # NAFBlocks per decoder level (deepest first)
    key: str  # checkpoint key ("params")
    tlc: int  # training patch size of `NAFNetLocal` (local channel attention, window 1.5 x); 0: `NAFNet` (global)


class _LocalMean(torch.nn.Module):
    """`local_arch.AvgPool2d` (TLC, `fast_imp=False`) with its window given: box means, replicate-padded to the size."""

    def __init__(self, window: int) -> None:
        super().__init__()
        self.window = window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if self.window >= h and self.window >= w:
            return x.mean((2, 3), keepdim=True)
        k1, k2 = min(h, self.window), min(w, self.window)
        s = x.new_zeros((*x.shape[:2], h + 1, w + 1), dtype=torch.float32)  # one float32 buffer, sums in place
        s[:, :, 1:, 1:] = x
        s = s.cumsum_(-1).cumsum_(-2)
        out = s[:, :, k1:, k2:] + s[:, :, :-k1, :-k2]
        out = out.sub_(s[:, :, :-k1, k2:]).sub_(s[:, :, k1:, :-k2]).div_(k1 * k2)
        del s
        _h, _w = out.shape[2:]
        pad = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
        return F.pad(out.to(x.dtype), pad, mode="replicate")


def _norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Channel LayerNorm without affine (folded into the next conv), over a channels-last view.

    On the CPU the result is made contiguous: the channels-last view sends the next (depthwise) convs down
    `_slow_conv2d` (5.5x slower).
    """
    y = F.layer_norm(x.permute(0, 2, 3, 1), x.shape[1:2], eps=eps).permute(0, 3, 1, 2)
    return y.contiguous() if x.device.type == "cpu" else y


def _block(m: NAFBlock, inp: torch.Tensor) -> torch.Tensor:
    """`NAFBlock.forward` (eval: no dropout) with the norms' affine and `beta` / `gamma` folded into the convs."""
    x = m.conv2(m.conv1(_norm(inp, m.norm1.eps)))
    a, b = x.chunk(2, 1)
    x = a * b
    y = inp + m.conv3(x * m.sca(x))
    a, b = m.conv4(_norm(y, m.norm2.eps)).chunk(2, 1)
    return y + m.conv5(a * b)


def _fold(m: NAFBlock) -> None:
    """Fold norm1 / norm2's affine into conv1 / conv4 and `beta` / `gamma` into conv3 / conv5."""
    with torch.no_grad():
        for norm, conv in ((m.norm1, m.conv1), (m.norm2, m.conv4)):
            bias: torch.Tensor = conv.bias  # ty: ignore[invalid-assignment]
            weight, shift = norm.weight, norm.bias  # LayerNorm2d parameters
            bias.add_(conv.weight[:, :, 0, 0] @ shift)  # ty: ignore[unsupported-operator]
            conv.weight.mul_(weight[None, :, None, None])  # ty: ignore[not-subscriptable]
        for scale, conv in ((m.beta, m.conv3), (m.gamma, m.conv5)):
            bias: torch.Tensor = conv.bias  # ty: ignore[invalid-assignment]
            conv.weight.mul_(scale.view(-1, 1, 1, 1))
            bias.mul_(scale.view(-1))
    eps1, eps2 = m.norm1.eps, m.norm2.eps
    m.norm1, m.norm2 = _Eps(eps1), _Eps(eps2)  # ty: ignore[invalid-assignment]
    del m.beta, m.gamma


_STREAM, _GATE = 1 / 16, 1 / 64  # powers of two: exact in the weights


def _scale_stream(net: NAFNet) -> None:
    """Carry the residual stream times `_STREAM` and the channel attention times `_GATE`: float16 overflows both.

    Upstream's GoPro model reaches ~2e5 in the stream of its deepest levels and ~6e5 in `x * sca(x)` on a 101 x 151
    image (~3e3 on GoPro frames). The stream is linear in `intro`, `downs` (their bias), each block's conv3 / conv5
    and the skips; the blocks' norms see it scale-free once their eps is scaled by `_STREAM^2`; `ending` takes the
    factor back. `sca`'s 1 x 1 conv carries `_GATE`, conv3 takes it back.
    """
    with torch.no_grad():
        net.intro.weight.mul_(_STREAM)
        net.intro.bias.mul_(_STREAM)  # ty: ignore[unresolved-attribute]
        net.ending.weight.div_(_STREAM)
        for down in net.downs:
            down.bias.mul_(_STREAM)  # ty: ignore[call-non-callable]
        for m in net.modules():
            if isinstance(m, _Eps):
                m.eps *= _STREAM**2
            elif isinstance(m, NAFBlock):
                for conv in (m.conv3, m.conv5):
                    conv.weight.mul_(_STREAM)
                    conv.bias.mul_(_STREAM)  # ty: ignore[unresolved-attribute]
                gate: torch.nn.Conv2d = m.sca[1]  # ty: ignore[invalid-assignment]
                gate.weight.mul_(_GATE)
                gate.bias.mul_(_GATE)  # ty: ignore[unresolved-attribute]
                m.conv3.weight.div_(_GATE)


class _Eps(torch.nn.Module):
    """A folded LayerNorm2d: only its epsilon is left."""

    def __init__(self, eps: float) -> None:
        super().__init__()
        self.eps = eps


class Portable(Runner):
    """Plain PyTorch in the card's dtype, channels-last on CUDA."""

    dtypes = ("float32", "float16", "bfloat16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network, fold the norms, set the TLC windows."""
        super().__init__(ctx)
        p = ctx.params
        self.net = NAFNet(
            img_channel=3, width=p.width, middle_blk_num=p.middle_blocks,
            enc_blk_nums=list(p.enc_blocks), dec_blk_nums=list(p.dec_blocks),
        )  # fmt: skip
        load_strict(self.net, ctx.state_dict(key=p.key), f"{ctx.variant.ref} checkpoint")
        n = len(p.enc_blocks)
        levels = [*((i, s) for i, s in enumerate(self.net.encoders)), (n, self.net.middle_blks)]
        levels += [(n - 1 - i, s) for i, s in enumerate(self.net.decoders)]
        for level, stage in levels:
            for m in stage:  # ty: ignore[not-iterable]
                _fold(m)
                override(m, _block)
                if p.tlc:  # upstream's window: (tlc >> level) * (1.5 tlc) // tlc
                    window = (p.tlc >> level) * (3 * p.tlc // 2) // p.tlc
                    m.sca[0] = _LocalMean(window)
        _scale_stream(self.net)
        self.fmt = torch.channels_last if torch.device(ctx.device).type == "cuda" else torch.contiguous_format
        self.net.eval().to(ctx.device, getattr(torch, ctx.dtype)).to(memory_format=self.fmt)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The restored batch."""
        return {"image": self.net(image.to(getattr(torch, self.ctx.dtype)).contiguous(memory_format=self.fmt))}

    def module(self) -> torch.nn.Module:
        """The network in the card's dtype."""
        dtype, fmt = getattr(torch, self.ctx.dtype), self.fmt
        return Forward(self.net, lambda net, image: net(image.to(dtype).contiguous(memory_format=fmt)))


def _hacc_conv(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    return cudnn_hacc.conv_half_acc(x, conv.weight, conv.bias, conv.padding, conv.stride, conv.dilation)  # ty: ignore[invalid-argument-type]


class Kernels(Portable):
    """float16, channels-last, fp16-accumulate convolutions (`cudnn_hacc`; the depthwise convs stay PyTorch's)."""

    module = Runner.module  # fp16-accumulate convolutions are cuDNN plans, not exportable operators

    def __init__(self, ctx: Context) -> None:
        """Build the network and route its dense convolutions through `cudnn_hacc`."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        for m in self.net.modules():
            if is_base(m, torch.nn.Conv2d) and m.groups == 1:
                override(m, _hacc_conv)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The restored batch."""
        x = image.to(torch.float16).contiguous(memory_format=self.fmt)
        with fp16_accumulate():
            return {"image": self.net(x)}


RUNNERS = {"portable": Portable, "kernels": Kernels}

_BLOCK = re.compile(r"(encoders|decoders)\.(\d+)\.(\d+)\.beta|middle_blks\.(\d+)\.beta")


def from_checkpoint(sd: Mapping[str, torch.Tensor]) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """`load(path)`: a NAFNet state dict as a variant and its params; keys unchanged.

    The deblurring structure (encoder `[1, 1, 1, 28]`, one middle block: GoPro, REDS) runs as `NAFNetLocal` with
    training size 256 (upstream's test options), any other as `NAFNet`.
    """
    if "intro.weight" not in sd or "ending.weight" not in sd or "encoders.0.0.conv1.weight" not in sd:
        return None
    if sd["intro.weight"].shape[1] != 3:  # noqa: PLR2004 - RGB
        msg = f"NAFNet checkpoint with {sd['intro.weight'].shape[1]} input channels: only RGB models are supported"
        raise WeightsError(msg)
    enc: dict[int, int] = {}
    dec: dict[int, int] = {}
    middle = 0
    for k in sd:
        if m := _BLOCK.fullmatch(k):
            if m.group(4) is not None:
                middle = max(middle, int(m.group(4)) + 1)
            else:
                d = enc if m.group(1) == "encoders" else dec
                d[int(m.group(2))] = max(d.get(int(m.group(2)), 0), int(m.group(3)) + 1)
    enc_blocks = tuple(enc[i] for i in range(len(enc)))
    deblur = middle == 1 and enc_blocks[-1:] == (28,)
    params = {
        "scale": 1,
        "width": sd["intro.weight"].shape[0],
        "enc_blocks": enc_blocks,
        "middle_blocks": middle,
        "dec_blocks": tuple(dec[i] for i in range(len(dec))),
        "tlc": 256 if deblur else 0,
    }
    return "nafnet-gopro" if deblur else "nafnet-sidd", params, dict(sd)
