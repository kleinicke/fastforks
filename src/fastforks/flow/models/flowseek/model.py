"""FlowSeek runners: weights -> network and the portable forward pass."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import check_fits
from fastforks.flow._runner import Forward, Runner
from fastforks.kernels import flow_upsample, flowseek_corr
from fastforks.shared import mlx_wrap

from .net import FlowSeek

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks._core.kernels import CubinModule
    from fastforks.flow._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[options]`."""

    iters: int  # flow updates after the initial regression (upstream `iters`; its "S" is T with 12)


class Portable(Runner):
    """Plain PyTorch; on CUDA in fp16 the networks run under autocast, motion bases and correlation in fp32.

    The card says `pad = "none"`: the network pads itself (upstream pads after Depth Anything has seen the unpadded
    images and the motion bases are made at the image size).
    """

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = FlowSeek()
        # upstream stores each shortcut batch norm twice (`bn3` and `downsample.1`, the same module); net.py has one
        sd = {k: v for k, v in ctx.state_dict().items() if ".downsample.1." not in k}
        check_fits(self.net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
        self.net.load_state_dict(sd)
        self.net.fp16 = ctx.dtype == "float16"
        self.net.eval().to(ctx.device)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a batch."""
        return {"flow": self.net(image1, image2, self.ctx.params.iters)}

    def module(self) -> torch.nn.Module:
        """The network; returns the flow."""
        iters = self.ctx.params.iters
        return Forward(self.net, lambda net, image1, image2: net(image1, image2, iters))


def _lookup(k: CubinModule, f1: torch.Tensor, f2: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Prepare native-width channels-last feature maps for the four pyramid levels."""
    b, c, h, w = f1.shape
    if c != 384 or f2.shape != f1.shape:  # noqa: PLR2004 - FlowSeek-T architecture
        msg = "FlowSeek-T needs two matching 384-channel feature maps"
        raise ValueError(msg)
    left = f1.to(dtype=torch.float16, memory_format=torch.channels_last)
    levels = [f2.to(dtype=torch.float16, memory_format=torch.channels_last)]
    for _ in range(3):
        levels.append(F.avg_pool2d(levels[-1], 2))

    def lookup(coords: torch.Tensor) -> torch.Tensor:
        if coords.shape != (b, 2, h, w) or coords.device != left.device or coords.dtype != torch.float32:
            msg = "FlowSeek coordinates must be float32 BCHW pixel coordinates on the feature device"
            raise ValueError(msg)
        out = left.new_empty((b, h, w, 324))
        k.launch("lookup", (h * w, b, 4), (128, 1, 1), left, *levels, coords.contiguous(), out, h, w, 1 / math.sqrt(c))
        return out.permute(0, 3, 1, 2)

    return lookup


class Kernels(Portable):
    """Portable FP16 network with native-width on-demand correlation and fused convex flow upsampling."""

    def __init__(self, ctx: Context) -> None:
        """Build the network and load the CUDA correlation kernel."""
        super().__init__(ctx)
        self.k = flowseek_corr.load(ctx.device)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a batch, with on-demand correlation."""
        return {
            "flow": self.net(
                image1, image2, self.ctx.params.iters, lambda a, b: _lookup(self.k, a, b), flow_upsample.convex
            )
        }

    module = Runner.module


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(Portable),
}
