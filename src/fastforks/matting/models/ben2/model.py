"""BEN2 runners: weights -> network and the portable / kernels forward passes.

Inference as the authors' `BEN_Base.inference`: the image resized to `size x size` (upstream: PIL Lanczos on the
8-bit image; here bicubic, antialiased, on the float image), ImageNet normalisation, the network's mask, resized back
to the image size (bilinear) and min-max normalised per image (upstream then rounds it to 8 bits).

`portable`: upstream's modules in plain PyTorch, fp32 or fp16 weights and activations (the authors run fp16 autocast
on CUDA), SDPA attention (Swin windows and the decoder's cross-attention). `kernels`: fp16 channels-last; every conv
runs as a cuDNN fp16-accumulate graph (`cudnn_hacc`) with an fp32 bias epilogue, the GEMMs accumulate in fp16 and the
InstanceNorms run as one pass each (`shared.cnn_kernels`); the two nearest 2x upsamplings and the convs after them
run as stride-2 transposed convs (`up_deconv`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.half import fp16_accumulate
from fastforks.matting._runner import Forward, Runner, scale
from fastforks.shared import mlx_wrap, swin_v1
from fastforks.shared.cnn_kernels import CL, prepare_generic

from .net.ben2 import BEN_Base

if TYPE_CHECKING:
    from fastforks.matting._runner import Context


@dataclass(frozen=True)
class Params:
    """Variant `params`.

    Attributes:
        size: Network input side (square).
    """

    size: int


def _cpu_net(ctx: Context) -> BEN_Base:
    net = BEN_Base()
    net.load_state_dict(ctx.state_dict())
    return net.eval()


# the Swin windows (144 tokens, a bias mask) on the memory-efficient kernel, the decoder's attention on the runner's
# cuDNN-first order (1080p, RTX 5060 Ti, Windows: windows 9.3 -> 7.6 ms; the decoder on memory-efficient 8.9 -> 22 ms)
_SWIN_SDPA = "efficient,cudnn,math"
_TINY = 1e-8  # fixed: our floor of the normalisation divisors (constant image / map: not 0 / 0)


# nearest 2x upsampling then a 3x3 conv (padding 1) along one axis: output row 2m reads input rows m - 1 (w0) and m
# (w1 + w2), row 2m + 1 reads m (w0 + w1) and m + 1 (w2): a stride-2 4-tap transposed conv with these summed taps
_UP_TAPS = ((0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0))  # fixed: deconv tap k = sum of the conv taps marked 1


def up_deconv(conv: nn.Conv2d) -> nn.ConvTranspose2d:
    """`conv(nearest_2x(x))` (3x3, padding 1, stride 1) as one transposed conv of `x` (exact up to rounding).

    A quarter of the multiply-adds per output pixel minus the halo (4 instead of 9) and no upsampled input map.

    Raises:
        ValueError: `conv` is not a plain 3x3 stride-1 padding-1 convolution.
    """
    if conv.kernel_size != (3, 3) or conv.stride != (1, 1) or conv.padding != (1, 1) or conv.groups != 1:
        msg = f"up_deconv: needs a 3x3 stride-1 padding-1 conv, got {conv}"
        raise ValueError(msg)
    w = conv.weight.detach()
    a = torch.tensor(_UP_TAPS, dtype=torch.float64, device=w.device)
    d = nn.ConvTranspose2d(
        conv.in_channels,
        conv.out_channels,
        4,
        stride=2,
        padding=1,
        bias=conv.bias is not None,
        device=w.device,
        dtype=w.dtype,
    )
    with torch.no_grad():
        d.weight.copy_(torch.einsum("ya,xb,oiab->ioyx", a, a, w.double()))
        if conv.bias is not None and d.bias is not None:
            d.bias.copy_(conv.bias.detach())
    return d


class _Matting(Runner):
    """Resize, normalise, network, sigmoid, resize back, min-max."""

    Params = Params
    net: BEN_Base
    dtype: torch.dtype

    def _net(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def __call__(self, image: torch.Tensor, trimap: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Alpha `[B, H, W]` of `[B, 3, H, W]` RGB in `[0, 1]`."""
        s = self.ctx.params.size
        x = F.interpolate(image, (s, s), mode="bicubic", align_corners=False, antialias=True)
        y = self._net(scale(x, self.ctx.io).to(self.dtype)).float().sigmoid()
        y = F.interpolate(y, image.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        lo, hi = y.amin((1, 2), keepdim=True), y.amax((1, 2), keepdim=True)
        return {"alpha": (y - lo) / (hi - lo).clamp_min(_TINY)}

    def module(self) -> torch.nn.Module:
        """The network with its pre- and post-processing: `[B, 3, H, W]` in `[0, 1]` -> alpha `[B, H, W]`."""
        return Forward(self.net, lambda _net, image: self(image, None)["alpha"])


class Portable(_Matting):
    """Upstream's modules in plain PyTorch, float32 or float16 (SDPA attention)."""

    dtypes = ("float32", "float16")

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.dtype = getattr(torch, ctx.dtype)
        self.net = _cpu_net(ctx).to(ctx.device, self.dtype)


class Kernels(_Matting):
    """fp16 channels-last network, cuDNN fp16-accumulate convs and GEMMs, InstanceNorm kernels."""

    def __init__(self, ctx: Context) -> None:
        """Build the network and convert its convs and norms."""
        super().__init__(ctx)
        net = _cpu_net(ctx).to(ctx.device)
        # the two nearest 2x + conv stages as transposed convs (1024 input, RTX 5060 Ti: 88.2 -> 81.7 ms)
        net.up_deconv1, net.up_deconv2 = up_deconv(net.upsample1[0]), up_deconv(net.upsample2[0])
        prepare_generic(net, ctx.device)  # no BatchNorm to fuse: convs + InstanceNorms
        swin_v1.fast(net.backbone, sdpa_auto=_SWIN_SDPA)
        self.dtype = torch.float16
        self.net = net

    def _net(self, x: torch.Tensor) -> torch.Tensor:
        with fp16_accumulate():  # the Swin and decoder GEMMs (1080p 102 -> 96 ms, golden error unchanged)
            return self.net(x.contiguous(memory_format=CL))


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(Portable),
}
