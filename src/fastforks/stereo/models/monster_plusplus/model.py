"""MonSter++ runners: weights -> network and the portable / kernels forward passes."""

from __future__ import annotations

import types
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from fastforks._core.checkpoint import load_strict
from fastforks._core.preprocess import IMAGENET_MEAN, IMAGENET_STD
from fastforks.shared import mlx_wrap
from fastforks.stereo._runner import Forward, Runner

from .net.inference import fold_for_inference, forward_fast
from .net.monster import Monster

if TYPE_CHECKING:
    from fastforks.stereo._runner import Context


@dataclass(frozen=True)
class Params:
    """Architecture (card `[params]`) and load options (card `[options]`)."""

    encoder: str  # DINOv2 encoder of the Depth Anything V2 branch
    max_disp: int
    iters: int


def _net(ctx: Context) -> Monster:
    args = types.SimpleNamespace(  # upstream evaluate_stereo.py defaults
        encoder=ctx.params.encoder,
        hidden_dims=[128] * 3,
        corr_levels=2,
        corr_radius=4,
        n_downsample=2,
        n_gru_layers=3,
        max_disp=ctx.params.max_disp,
        mixed_precision=False,  # the CUDA paths run in pure fp16 / fp32 weights instead of autocast
    )
    assign = (
        ctx.device.type == "cuda"
        and ctx.variant.ref == "monster-plusplus-mix-all@1"
        and ctx.user_weights is None
        and ctx.user_state is None
    )
    with torch.device("meta") if assign else nullcontext():
        net = Monster(args)
    state = ctx.state_dict(key="state_dict")
    if assign:
        # The vendored constructor keeps these tensors outside the checkpoint.
        net.mean = torch.tensor(IMAGENET_MEAN, device="cpu")
        net.std = torch.tensor(IMAGENET_STD, device="cpu")
        load_strict(net, state, f"{ctx.variant.ref} checkpoint", assign=True)
        if ctx.dtype == "float16":
            # FastViT prepares its interpolation cache from the unrounded checkpoint table.
            encoder = net.mono_encoder
            position = encoder.pos_embed
            encoder.register_parameter("pos_embed", None)
            encoder.to(device=ctx.device, dtype=torch.float16)
            encoder.register_parameter(
                "pos_embed", torch.nn.Parameter(position.to(ctx.device), requires_grad=position.requires_grad)
            )
    else:
        net.load_state_dict(state)
    del state
    return net.eval().to(ctx.device)


class Portable(Runner):
    """Plain PyTorch: BatchNorm as fp32 affine, pure fp16 (CUDA) or fp32 weights, NHWC on CUDA, SDPA ViT attention."""

    deterministic_cudnn = True

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        fold_for_inference(self.net, half=ctx.dtype == "float16", channels_last=ctx.device.type == "cuda")

    def _run(self, net: Monster, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return forward_fast(net, left, right, iters=self.ctx.params.iters)[:, 0]

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded batch."""
        return {"disparity": self._run(self.net, left, right)}

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        return Forward(self.net, self._run)


class Kernels(Runner):
    """Our kernels: fused GRUs / lookups, fp16-accumulate convolutions, fastbb ViT, fp16 NHWC."""

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
        return {"disparity": self._forward(self.net, left, right, iters=self.ctx.params.iters)[:, 0]}


class MetalKernels(Portable):
    """Apple MPS: the portable network with Metal kernels for the geometry / correlation lookup and the ConvGRUs."""

    module = Runner.module  # the Metal kernels are not exportable

    def __init__(self, ctx: Context) -> None:
        """Build the network and switch it to the Metal kernels."""
        super().__init__(ctx)
        from fastforks._core import mps  # noqa: PLC0415 - MPS only

        mps.adapt(self.net)  # the faster / fp16-capable transposed convolutions, as for the portable runner
        from fastforks.kernels import stereo_metal  # noqa: PLC0415 - needs the kernel library
        from fastforks.kernels.stereo_metal.modules import patch_igev  # noqa: PLC0415

        from .net.geometry import FastGeoEncodingVolume  # noqa: PLC0415
        from .net.update import ConvGRU  # noqa: PLC0415

        patch_igev(self.net, FastGeoEncodingVolume, ConvGRU, stereo_metal.load(ctx.device))


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "kernels_mps": MetalKernels,
    "portable_mlx": mlx_wrap.port(Portable, whole="disparity"),
}
