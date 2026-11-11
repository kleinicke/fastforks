"""NeuFlow v2 runners: weights -> network and the portable forward pass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastforks._core.checkpoint import check_fits
from fastforks.flow._runner import Forward, Runner

from .net import NeuFlow

if TYPE_CHECKING:
    import torch

    from fastforks.flow._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[options]`."""

    iters: int  # refinements at 1/8 resolution (upstream `iters_s8`)


class Portable(Runner):
    """Plain PyTorch; on CUDA in fp16 the convolutions and the cross-attention run under autocast, the rest in fp32."""

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


RUNNERS = {"portable": Portable}
