"""FlowSeek runners: weights -> network and the portable forward pass."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import check_fits
from fastforks.flow._runner import Forward, Runner

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
    """Compute the four correlation windows on demand without an all-pairs volume.

    FlowSeek has 384 features; pad them for the existing 1024-channel lookup and retain the original scale.
    """
    b, c, h, w = f1.shape
    width = 1024  # fixed by the existing correlation kernel
    if c > width:
        msg = f"FlowSeek correlation width {c} exceeds the {width}-channel lookup"
        raise ValueError(msg)
    a = F.pad(f1.half().permute(0, 2, 3, 1), (0, width - c)).contiguous()
    pyr, x = [], f2.half()
    for _ in range(4):
        pyr.append(F.pad(x.permute(0, 2, 3, 1), (0, width - c)).contiguous())
        x = F.avg_pool2d(x, 2)

    def lookup(coords: torch.Tensor) -> torch.Tensor:
        xy = coords.permute(0, 2, 3, 1).contiguous()
        out = torch.empty(b, h, w, 324, dtype=torch.float16, device=coords.device)
        for i in range(b):
            k.launch(
                "corr_lookup_c1024_r4",
                (-(-h * w // 8), 1, 1),
                (256, 1, 1),
                a[i],
                *(p[i] for p in pyr),
                h,
                w,
                xy[i],
                out[i],
                324,
                1 / math.sqrt(c),
            )
        return out.permute(0, 3, 1, 2)

    return lookup


class Kernels(Portable):
    """Portable FP16 network with on-demand correlation windows (`raft_flow`) instead of all-pairs volumes."""

    dtypes = ("float16",)

    def __init__(self, ctx: Context) -> None:
        """Build the network and load the CUDA correlation kernel."""
        super().__init__(ctx)
        from fastforks.kernels import raft_flow  # noqa: PLC0415 - needs the kernel library

        self.k = raft_flow.load(ctx.device)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a batch, with on-demand correlation."""
        return {"flow": self.net(image1, image2, self.ctx.params.iters, lambda a, b: _lookup(self.k, a, b))}

    module = Runner.module


RUNNERS = {"portable": Portable, "kernels": Kernels}
