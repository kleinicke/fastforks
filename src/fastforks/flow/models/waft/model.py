"""WAFT runners: weights -> network and the portable forward pass."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from fastforks._core.checkpoint import check_fits
from fastforks.flow._runner import Forward, Runner
from fastforks.shared.depth_anything_v2 import fold_layer_scale, sdpa_attention

from .net import ViTWarpV8

if TYPE_CHECKING:
    from fastforks.flow._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[options]`."""

    iters: int  # warping iterations (upstream config/eval: 5)


def _net() -> ViTWarpV8:
    # upstream config/a1/*.json: DAv2-S features, ViT-S refinement; var_max / var_min only shape the training loss
    args = SimpleNamespace(dav2_backbone="vits", network_backbone="vits", iters=5, var_max=10, var_min=0)
    return ViTWarpV8(args)


class Portable(Runner):
    """Plain PyTorch; on CUDA in fp16 the network runs under autocast (warping coordinates and the flow in fp32).

    The card says `pad = "none"`: the network normalises and pads itself (zeros after the normalisation, to multiples
    of 112, as upstream's Padder 'sintel' mode). Depth Anything's attention runs through SDPA with LayerScale folded.
    """

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net()
        sd = ctx.state_dict()
        check_fits(self.net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
        self.net.load_state_dict(sd)
        sdpa_attention(self.net.da_feature)
        fold_layer_scale(self.net.da_feature)
        self.net.eval().to(ctx.device)
        self.fp16 = ctx.dtype == "float16"

    def _autocast(self, x: torch.Tensor) -> contextlib.AbstractContextManager:
        return torch.autocast(x.device.type, torch.float16) if self.fp16 else contextlib.nullcontext()

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a batch (the last iteration's)."""
        with self._autocast(image1):
            flow = self.net(image1, image2, self.ctx.params.iters)["flow"][-1]
        return {"flow": flow.float()}

    def module(self) -> torch.nn.Module:
        """The network; returns the last iteration's flow."""
        iters = self.ctx.params.iters
        return Forward(self.net, lambda net, image1, image2: net(image1, image2, iters)["flow"][-1])


RUNNERS = {"portable": Portable}
