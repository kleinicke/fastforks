"""DEFOM-Stereo runners: weights -> network and the portable / kernels forward passes."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from fastforks.shared.depth_anything_v2 import use_sdpa_attention
from fastforks.stereo._runner import Forward, Runner

from .net.defom_stereo import DEFOMStereo

if TYPE_CHECKING:
    from fastforks.stereo._runner import Context


@dataclass(frozen=True)
class Params:
    """Architecture (card `[params]`) and load options (card `[options]`)."""

    encoder: str  # DINOv2 encoder of the Depth Anything V2 branch
    iters: int
    scale_iters: int


def _net(ctx: Context, *, mixed_precision: bool = False) -> DEFOMStereo:
    args = types.SimpleNamespace(
        dinov2_encoder=ctx.params.encoder,
        idepth_scale=0.5,
        hidden_dims=[128] * 3,
        corr_levels=2,
        corr_radius=4,
        scale_list=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        scale_corr_radius=2,
        n_downsample=2,
        context_norm="batch",
        n_gru_layers=3,
        mixed_precision=mixed_precision,
        shared_backbone=False,
    )
    net = DEFOMStereo(args)
    net.load_state_dict(ctx.state_dict())
    return net.eval().to(ctx.device)


class Portable(Runner):
    """Plain PyTorch; bfloat16 on CUDA is upstream's autocast (`mixed_precision`), the correlation stays float32.

    float16 autocast overflows (NaN): the DAv2 activations exceed the float16 range.
    """

    dtypes = ("float32", "bfloat16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx, mixed_precision=ctx.dtype == "bfloat16")
        # fused attention: the eager q k^T matrix of the 7.5k-token Midd-H pair is 7 GB in float32 (softmax output)
        use_sdpa_attention(self.net.defomencoder.depth_anything.pretrained)

    def _run(self, net: DEFOMStereo, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        p = self.ctx.params
        if self.ctx.dtype != "bfloat16":
            return net(left, right, iters=p.iters, scale_iters=p.scale_iters, test_mode=True)[:, 0]
        # upstream's autocast regions take the default autocast dtype (float16): switch it for the call
        default = torch.get_autocast_dtype("cuda")
        torch.set_autocast_dtype("cuda", torch.bfloat16)
        try:
            return net(left, right, iters=p.iters, scale_iters=p.scale_iters, test_mode=True)[:, 0]
        finally:
            torch.set_autocast_dtype("cuda", default)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        return {"disparity": self._run(self.net, left, right)}

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        return Forward(self.net, self._run)


class Kernels(Runner):
    """Our kernels: fp16 NHWC update loop with fused GRU / lookup kernels; bf16 encoders (plain PyTorch)."""

    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network and the kernel state."""
        super().__init__(ctx)
        from .kernels.inference import forward_kernels, prepare_kernels  # noqa: PLC0415 - needs the kernel libraries

        self.net = _net(ctx)
        prepare_kernels(self.net, ctx.device)
        self._forward = forward_kernels

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        p = self.ctx.params
        return {"disparity": self._forward(self.net, left, right, iters=p.iters, scale_iters=p.scale_iters)[:, 0]}


class MetalKernels(Portable):
    """Apple MPS: float16 ViT / update loop, float32 DPT feature head, fused ConvGRUs and a one-kernel lookup."""

    module = Runner.module  # the Metal kernels are not exportable
    dtypes = ("float16",)

    def __init__(self, ctx: Context) -> None:
        """Build the network and switch it to the Metal kernels."""
        super().__init__(ctx)
        from fastforks._core import mps  # noqa: PLC0415 - MPS only
        from fastforks.kernels import stereo_metal  # noqa: PLC0415 - needs the kernel library

        from .kernels.metal import forward, prepare  # noqa: PLC0415

        mps.adapt(self.net)
        self.lib = stereo_metal.load(ctx.device)
        prepare(self.net, self.lib)
        self._forward = forward

    def _run(self, net: DEFOMStereo, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        p = self.ctx.params
        return self._forward(net, self.lib, left, right, p.iters, p.scale_iters)[:, 0]


RUNNERS = {"portable": Portable, "kernels": Kernels, "kernels_mps": MetalKernels}
