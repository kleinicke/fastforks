"""MoCha-Stereo runners: weights -> network and the portable / kernels forward passes."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from fastforks.kernels import cudnn_hacc
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import SmallShapeGraphs
from fastforks.stereo._runner import Forward, Runner

from .net.inference import fold_for_inference, forward_fast
from .net.mocha_stereo import Mocha

if TYPE_CHECKING:
    from fastforks.stereo._runner import Context


@dataclass(frozen=True)
class Params:
    """Architecture (card `[params]`) and load options (card `[options]`)."""

    max_disp: int
    mccv: bool  # motif channel correlation volume (V1); V2 feeds the plain left features to the hourglass
    iters: int
    cuda_graphs: bool  # kernels replays CUDA graphs for small repeated shapes


class _LeftFeatures(torch.nn.Module):
    """MoCha-V2 stand-in for the MCCV: the hourglass reads the left features (see the card)."""

    def forward(self, features_left: list[torch.Tensor], _features_right: list[torch.Tensor]) -> list[torch.Tensor]:
        return list(features_left)


def _net(ctx: Context) -> Mocha:
    args = types.SimpleNamespace(  # upstream evaluate_stereo.py defaults
        hidden_dims=[128] * 3,
        n_downsample=2,
        n_gru_layers=3,
        corr_levels=2,
        corr_radius=4,
        max_disp=ctx.params.max_disp,
        mixed_precision=False,  # the CUDA path runs in pure fp16 weights instead of autocast
    )
    net = Mocha(args)
    sd = ctx.state_dict()
    if not ctx.params.mccv:
        net.mccv = _LeftFeatures()  # ty: ignore[invalid-assignment] - same call signature
        sd = {k: v for k, v in sd.items() if not k.startswith("CCV.")}
    net.load_state_dict(sd)
    return net.eval().to(ctx.device)


class Portable(Runner):
    """Plain PyTorch: fp32 BatchNorm affine, merged LSTM gate convs, pure fp16 (CUDA) or fp32 weights, NHWC on CUDA."""

    deterministic_cudnn = True

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        fold_for_inference(self.net, half=ctx.dtype == "float16", channels_last=ctx.device.type == "cuda")

    def _run(self, net: Mocha, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return forward_fast(net, left, right, iters=self.ctx.params.iters)[:, 0]

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded batch."""
        return {"disparity": self._run(self.net, left, right)}

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        return Forward(self.net, self._run)


_GRAPH_PIXELS = 500_000  # fixed: batch x padded H x W; launch-bound up to KITTI 375x1242, measured
_GRAPHS = 2  # fixed: graph cache policy


class Kernels(Runner):
    """Our kernels: fp16-accumulate convs with fused epilogues (incl. the LSTM gates), geometry lookup, fp16 NHWC.

    A repeated input shape up to `_GRAPH_PIXELS` replays a CUDA graph (`fastforks.shared.cuda_graphs`): the eager
    pass is launch-bound there.
    """

    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network and the kernel state."""
        super().__init__(ctx)
        from .kernels.inference import forward_kernels, prepare_kernels  # noqa: PLC0415 - needs the kernel libraries

        self.net = _net(ctx)
        prepare_kernels(self.net, ctx.device)
        self._forward = forward_kernels
        self._graphs = SmallShapeGraphs(
            self._run,
            pixels=_GRAPH_PIXELS,
            graphs=_GRAPHS,
            enabled=ctx.params.cuda_graphs,
            capture_after=cudnn_hacc.capture_after(),
        )

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        self._graphs.release()

    def _run(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self._forward(self.net, left, right, iters=self.ctx.params.iters)[:, 0]

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        return {"disparity": self._graphs(left, right)}


class MetalKernels(Portable):
    """Apple MPS: the portable network with its geometry / correlation lookup as Metal kernels (not its LSTMs)."""

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


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "kernels_mps": MetalKernels,
    "portable_mlx": mlx_wrap.port(Portable, whole="disparity"),
}
