"""IS-Net runners: weights -> network and the portable / kernels forward passes.

Inference as upstream's `IS-Net/Inference.py`: the image resized to `size x size` (bilinear, no anti-aliasing; upstream
`F.upsample` on the uint8 tensor, so it truncates to 8 bits; here it stays float), normalised with mean 0.5 and std 1,
the first side output `d1` (sigmoid), resized back to the image size (bilinear) and min-max normalised.

`portable`: upstream's modules in plain PyTorch, fp32 or fp16; in fp16 each BatchNorm is folded into its conv (as in
U²-Net, unnormalised conv outputs can exceed the float16 range). `kernels`: fp16 channels-last; each conv ->
BatchNorm -> ReLU block (`REBNCONV`) is one cuDNN fp16-accumulate conv (`cudnn_hacc`), the BatchNorm folded into it,
with the bias and the ReLU in an fp32 epilogue (`shared.cnn_kernels`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.patch import override
from fastforks.matting._runner import Forward, Runner, scale
from fastforks.shared.cnn_kernels import ACT, CL, conv_ep, fold_bn, prepare_generic

from .net import ISNetDIS
from .net.isnet import REBNCONV

if TYPE_CHECKING:
    from fastforks.matting._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params`.

    Attributes:
        size: Network input side (square).
    """

    size: int


def _cpu_net(ctx: Context, *, fold: bool) -> torch.nn.Module:
    net = ISNetDIS(3, 1)
    net.load_state_dict(ctx.state_dict())
    if fold:
        for m in net.modules():
            if isinstance(m, REBNCONV):
                fold_bn(m.conv_s1, m.bn_s1)
    return net.eval()


class _Matting(Runner):
    """Resize, normalise, network (`d1`), resize back, min-max."""

    Params = Params
    compiled_methods = ("_net",)
    net: torch.nn.Module
    dtype: torch.dtype

    def _net(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)[0][0]

    def __call__(self, image: torch.Tensor, trimap: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Alpha `[B, H, W]` of `[B, 3, H, W]` RGB in `[0, 1]`."""
        s = self.ctx.params.size
        x = F.interpolate(image, (s, s), mode="bilinear", align_corners=False)
        y = self._net(scale(x, self.ctx.io).to(self.dtype)).float()
        y = F.interpolate(y, image.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        lo, hi = y.amin((1, 2), keepdim=True), y.amax((1, 2), keepdim=True)
        return {"alpha": (y - lo) / (hi - lo)}

    def module(self) -> torch.nn.Module:
        """The network with its pre- and post-processing: `[B, 3, H, W]` in `[0, 1]` -> alpha `[B, H, W]`."""
        return Forward(self.net, lambda _net, image: self(image, None)["alpha"])


class Portable(_Matting):
    """Upstream's modules in plain PyTorch, float32 or float16."""

    dtypes = ("float32", "float16")

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.dtype = getattr(torch, ctx.dtype)
        self.net = _cpu_net(ctx, fold=self.dtype == torch.float16).to(ctx.device, self.dtype)


def _rebnconv(m: REBNCONV, x: torch.Tensor) -> torch.Tensor:
    """`relu(bn(conv(x)))` as one conv (the BatchNorm folded in) with an fp32 bias + ReLU epilogue."""
    return conv_ep(m.conv_s1, x, act=ACT["relu"])


class Kernels(_Matting):
    """fp16 channels-last network, cuDNN fp16-accumulate convs with BatchNorm + ReLU epilogues."""

    module = Runner.module  # fp16-accumulate convolutions are cuDNN plans, not exportable operators

    def __init__(self, ctx: Context) -> None:
        """Build the network, fold each BatchNorm into its conv and the ReLU into the conv's epilogue."""
        super().__init__(ctx)
        net = _cpu_net(ctx, fold=True).to(ctx.device)
        for m in net.modules():
            if isinstance(m, REBNCONV):
                override(m, _rebnconv)
        prepare_generic(net, ctx.device)
        self.dtype = torch.float16
        self.net = net

    def _net(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.contiguous(memory_format=CL))[0][0]


RUNNERS = {"portable": Portable, "kernels": Kernels}
