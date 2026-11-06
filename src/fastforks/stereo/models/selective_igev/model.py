"""Selective-IGEV runners: weights -> network and the portable / kernels forward passes."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastforks.stereo._runner import Forward, Runner

from .net.igev_stereo import IGEVStereo
from .net.inference import fold_for_inference
from .portable import forward_fast

if TYPE_CHECKING:
    import torch

    from fastforks.stereo._runner import Context


@dataclass(frozen=True)
class Params:
    """Architecture (card `[params]`) and load options (card `[options]`)."""

    max_disp: int
    iters: int


def _net(ctx: Context) -> IGEVStereo:
    args = types.SimpleNamespace(
        hidden_dims=[128] * 3,
        corr_levels=2,
        corr_radius=4,
        n_downsample=2,
        n_gru_layers=3,
        max_disp=ctx.params.max_disp,
        mixed_precision=False,  # the CUDA paths run in pure fp16 / fp32 weights instead of autocast
        precision_dtype="float32",
    )
    net = IGEVStereo(args)
    net.load_state_dict(ctx.state_dict())
    return net.eval().to(ctx.device)


class Portable(Runner):
    """Plain PyTorch: folded BatchNorm, pure fp16 (CUDA) or fp32 weights, NHWC on CUDA, sync-free update loop."""

    deterministic_cudnn = True

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        fold_for_inference(self.net, half=ctx.dtype == "float16", channels_last=ctx.device.type == "cuda")

    def _run(self, net: IGEVStereo, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return forward_fast(net, left, right, self.ctx.params.iters)[:, 0]

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded batch."""
        return {"disparity": self._run(self.net, left, right)}

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        return Forward(self.net, self._run)


class Kernels(Runner):
    """Our kernels: fused geometry lookup, fp16-accumulate convolutions with fused epilogues, fused selective GRU."""

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
    """Apple MPS: the portable network with its geometry / correlation lookup as Metal kernels (not its GRUs)."""

    module = Runner.module  # the Metal kernels are not exportable

    def __init__(self, ctx: Context) -> None:
        """Build the network and switch it to the Metal kernels."""
        super().__init__(ctx)
        from fastforks._core import mps  # noqa: PLC0415 - MPS only

        mps.adapt(self.net)  # the faster / fp16-capable transposed convolutions, as for the portable runner
        from fastforks.kernels import stereo_metal  # noqa: PLC0415 - needs the kernel library
        from fastforks.kernels.stereo_metal.modules import patch_igev  # noqa: PLC0415

        from .net.geometry import FastGeoEncodingVolume  # noqa: PLC0415

        patch_igev(self.net, FastGeoEncodingVolume, None, stereo_metal.load(ctx.device))


RUNNERS = {"portable": Portable, "kernels": Kernels, "kernels_mps": MetalKernels}
