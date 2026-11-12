"""NeuFlow v2 runners: weights -> network, the portable forward pass and the kernels one (no correlation volume)."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from fastforks._core.checkpoint import check_fits
from fastforks.flow._runner import Forward, Runner

from .net import RADIUS, NeuFlow, _grid

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks._core.kernels import CubinModule
    from fastforks.flow._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[options]`."""

    iters: int  # refinements at 1/8 resolution (upstream `iters_s8`)


class Portable(Runner):
    """Plain PyTorch; in fp16 the convolutions and the cross-attention run under autocast, the volume is fp16."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = NeuFlow()
        sd = ctx.state_dict(key="model")
        check_fits(self.net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
        self.net.load_state_dict(sd)
        self.net.fp16 = ctx.dtype == "float16"
        self.net.eval().to(ctx.device)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a padded batch."""
        return {"flow": self.net(image1, image2, self.ctx.params.iters)}

    def module(self) -> torch.nn.Module:
        """The network; returns the flow."""
        iters = self.ctx.params.iters
        return Forward(self.net, lambda net, image1, image2: net(image1, image2, iters))


def _kernel_lookup(k: CubinModule, features: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """The 9 x 9 window of the correlation computed on the fly by `raft_flow`'s lookup: O(HW) memory, no volume.

    Same operands as the portable path (the fp16 features), fp32 sums and bilinear weights; the window is written in
    fp16, which the refinement convolutions cast it to anyway.
    """
    f0, f1 = (f.to(torch.float16).permute(0, 2, 3, 1).contiguous() for f in features.chunk(2, dim=0))
    b, h, w, c = f0.shape
    side = (2 * RADIUS + 1) ** 2

    def lookup(flow: torch.Tensor) -> torch.Tensor:
        coords = (_grid(flow) + flow).permute(0, 2, 3, 1).contiguous()
        out = torch.empty(b, h, w, side, dtype=torch.float16, device=flow.device)
        for i in range(b):  # one warp per pixel, 8 warps per block
            k.launch(
                "corr_lookup_c128_r4_l1",
                (-(-h * w // 8), 1, 1),
                (256, 1, 1),
                f0[i],
                f1[i],
                h,
                w,
                coords[i],
                out[i],
                side,
                1 / math.sqrt(c),
            )
        return out.permute(0, 3, 1, 2)

    return lookup


class Kernels(Portable):
    """The portable fp16 pass with the correlation window computed on the fly (`raft_flow`): no all-pairs volume."""

    def __init__(self, ctx: Context) -> None:
        """Build the network and bind the lookup kernel."""
        super().__init__(ctx)
        from fastforks.kernels import raft_flow  # noqa: PLC0415 - needs the kernel library

        self.net.correlate = functools.partial(_kernel_lookup, raft_flow.load(ctx.device))

    def module(self) -> torch.nn.Module:
        """No exportable module: the lookup is a kernel launch."""
        return Runner.module(self)


RUNNERS = {"portable": Portable, "kernels": Kernels}
