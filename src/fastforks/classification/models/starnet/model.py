"""StarNet: explicitly optimise its convolutional blocks as a CNN."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fastforks._core.runner import Runner
from fastforks._optimize import HACC_PIX, optimize
from fastforks.shared import libnet
from fastforks.shared.cuda_graphs import SmallShapeGraphs

if TYPE_CHECKING:
    from fastforks._core.runner import Context


class Portable(libnet.Portable):
    """Fold BatchNorm in StarNet's CNN (its timm `Block` name also occurs in ViTs)."""

    def __init__(self, ctx: Context) -> None:
        """Build the upstream network and select the convolutional passes explicitly."""
        Runner.__init__(self, ctx)
        self.net = optimize(
            self.network(ctx), mode=ctx.mode, family="cnn", device=ctx.device, fp16=ctx.dtype == "float16"
        )
        if ctx.device.type == "cuda":
            torch.cuda.empty_cache()


class Kernels(libnet.Kernels):
    """Fused CNN kernels with the shared small-batch CUDA graph policy."""

    def __init__(self, ctx: Context) -> None:
        """Build the explicit CNN path and prepare its bounded graph cache."""
        Runner.__init__(self, ctx)
        self.net = optimize(self.network(ctx), mode="kernels", family="cnn", device=ctx.device)
        torch.cuda.empty_cache()
        self._graphs = SmallShapeGraphs(self.net, pixels=HACC_PIX - 1, graphs=2)


RUNNERS = {"portable": Portable, "kernels": Kernels}
