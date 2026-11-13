"""SPAN runners: the network of hongyuanyu/SPAN and BasicSR's test call (`net(lq)`, clamped to `[0, 1]`).

Upstream's `Conv3XC` re-fuses its training branches (1 x 1 -> 3 x 3 -> 1 x 1 + 1 x 1 skip) into `eval_conv` on every
eval forward; here they are fused once at load (`update_params`, upstream's own code) and the forward is the fused 3 x 3
conv (+ leaky ReLU where upstream has one). The input mean is a plain attribute upstream (`type_as` per call), here a
buffer of the network's dtype and device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import override
from fastforks.kernels import cudnn_hacc
from fastforks.restoration._runner import Forward, Runner

from .net import SPAN
from .net.span_arch import Conv3XC

if TYPE_CHECKING:
    from fastforks.restoration._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params`."""

    scale: int  # 4 or 2
    channels: int  # feature channels (48 in the release models)


def _fused(m: Conv3XC, x: torch.Tensor) -> torch.Tensor:
    """`Conv3XC.forward` in eval mode, without re-fusing the weights."""
    out = m.eval_conv(x)
    return F.leaky_relu(out, negative_slope=0.05) if m.has_relu else out


class Portable(Runner):
    """Plain PyTorch in the card's dtype."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network and fuse its `Conv3XC` branches once."""
        super().__init__(ctx)
        p = ctx.params
        self.net = SPAN(3, 3, feature_channels=p.channels, upscale=p.scale)
        load_strict(self.net, ctx.state_dict(key="params_ema"), f"{ctx.variant.ref} checkpoint")
        for m in self.net.modules():
            if isinstance(m, Conv3XC):
                m.update_params()
                del m.sk, m.conv  # training branches, folded into eval_conv
                override(m, _fused)
        mean = self.net.mean
        del self.net.mean
        self.net.register_buffer("mean", mean, persistent=False)
        self.net.eval().to(ctx.device, getattr(torch, ctx.dtype))

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The upscaled batch."""
        return {"image": self.net(image.to(getattr(torch, self.ctx.dtype)))}

    def module(self) -> torch.nn.Module:
        """The network in the card's dtype."""
        dtype = getattr(torch, self.ctx.dtype)
        return Forward(self.net, lambda net, image: net(image.to(dtype)))


def _hacc_conv(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    return cudnn_hacc.conv_half_acc(x, conv.weight, conv.bias, conv.padding, conv.stride, conv.dilation)  # ty: ignore[invalid-argument-type]


class Kernels(Portable):
    """float16, channels-last, fp16-accumulate convolutions (`cudnn_hacc`)."""

    module = Runner.module  # fp16-accumulate convolutions are cuDNN plans, not exportable operators

    def __init__(self, ctx: Context) -> None:
        """Build the network and route its convolutions through `cudnn_hacc`."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        for m in self.net.modules():
            if type(m) is torch.nn.Conv2d:
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
                override(m, _hacc_conv)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The upscaled batch."""
        x = image.to(torch.float16).contiguous(memory_format=torch.channels_last)
        with fp16_accumulate():
            return {"image": self.net(x)}


RUNNERS = {"portable": Portable, "kernels": Kernels}
