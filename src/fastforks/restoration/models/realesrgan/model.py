"""Real-ESRGAN runners: BasicSR's `RRDBNet` and the call of upstream `RealESRGANer.enhance`.

As upstream (`inference_realesrgan.py`, no tiling, `pre_pad` 0): the RGB image in `[0, 1]` goes through the network
(x2: reflect-padded at the bottom and right to even sizes, then pixel-unshuffled), the output is cropped and clamped to
`[0, 1]`. Upstream runs float16 by default (`--fp32` for float32).
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
from fastforks.restoration._runner import Crop, Forward, Runner

from .net import RRDBNet

if TYPE_CHECKING:
    from fastforks.restoration._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params`."""

    scale: int  # 4 or 2 (x2: the input is pixel-unshuffled)
    num_block: int  # RRDB blocks: 23, 6 in the anime model


class Portable(Runner):
    """Plain PyTorch in the card's dtype."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        p = ctx.params
        self.net = RRDBNet(num_in_ch=3, num_out_ch=3, scale=p.scale, num_feat=64, num_block=p.num_block, num_grow_ch=32)
        load_strict(self.net, ctx.state_dict(key="params_ema"), f"{ctx.variant.ref} checkpoint")
        self.net.eval().to(ctx.device, getattr(torch, ctx.dtype))

    def pad(self, x: torch.Tensor) -> tuple[torch.Tensor, Crop]:
        """x2: reflect padding at the bottom and right to even sizes (`RealESRGANer.pre_process`); x4: none."""
        h, w = x.shape[-2:]
        if self.ctx.params.scale == 2 and (h % 2 or w % 2):  # noqa: PLR2004 - x2 pixel-unshuffles
            x = F.pad(x, (0, w % 2, 0, h % 2), mode="reflect")
        return x, Crop(0, 0, h, w)

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
