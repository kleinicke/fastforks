"""FlowSeek runners: weights -> network and the portable forward pass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastforks._core.checkpoint import check_fits
from fastforks.flow._runner import Forward, Runner

from .net import FlowSeek

if TYPE_CHECKING:
    import torch

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


RUNNERS = {"portable": Portable}
