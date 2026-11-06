"""IGEV++ runners: weights -> network and the portable / kernels forward passes."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastforks.kernels import cudnn_hacc
from fastforks.shared.cuda_graphs import SmallShapeGraphs
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
    cuda_graphs: bool  # option: kernels replays CUDA graphs for small repeated shapes


def _net(ctx: Context) -> IGEVStereo:
    md = ctx.params.max_disp
    args = types.SimpleNamespace(
        hidden_dims=[128] * 3,
        corr_levels=2,
        corr_radius=4,
        n_downsample=2,
        n_gru_layers=3,
        max_disp=md,
        # small / medium / large ranges at 1/4 resolution, each 48 bins of the full 1/4 volume (upstream: 48 / 96 / 192)
        s_disp_range=md // 16,
        m_disp_range=md // 8,
        l_disp_range=md // 4,
        s_disp_interval=1,
        m_disp_interval=2,
        l_disp_interval=4,
        mixed_precision=False,  # the CUDA paths run in pure fp16 / fp32 weights instead of autocast
        precision_dtype="float32",
    )
    net = IGEVStereo(args)
    net.load_state_dict(ctx.state_dict())
    return net.eval().to(ctx.device)


class Portable(Runner):
    """Plain PyTorch: folded BatchNorm, fp32-statistics InstanceNorm, pure fp16 (CUDA) or fp32, NHWC on CUDA."""

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


# graph vs eager, measured: -12.6 % at KITTI 384x1248, -7.7 % at 544x960, -2.1 % at 736x1280
_GRAPH_PIXELS = 500_000  # fixed: batch x padded H x W; launch-bound up to KITTI 384x1248
_GRAPHS = 2  # fixed: graph cache policy


class Kernels(Runner):
    """Our kernels: fused geometry lookup, fp16-accumulate convolutions with fused epilogues, fused ConvGRU.

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
            capture_after=cudnn_hacc.BUILD_AFTER,
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


RUNNERS = {"portable": Portable, "kernels": Kernels, "kernels_mps": MetalKernels}
