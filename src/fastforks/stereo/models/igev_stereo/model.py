"""IGEV-Stereo runners: weights -> network and the portable / kernels forward passes."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastforks.kernels import cudnn_hacc
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import SmallShapeGraphs
from fastforks.stereo._runner import Forward, Runner

from .net.igev_stereo import IGEVStereo
from .net.inference import fold_for_inference

if TYPE_CHECKING:
    import torch

    from fastforks.stereo._runner import Context


@dataclass(frozen=True)
class Params:
    """Architecture (card `[params]`) and load options (card `[options]`)."""

    max_disp: int
    iters: int
    cuda_graphs: bool


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
        return net.forward_fast(left, right, iters=self.ctx.params.iters)[:, 0]

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded batch."""
        return {"disparity": self._run(self.net, left, right)}

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        return Forward(self.net, self._run)


class Kernels(Runner):
    """Our kernels: fused correlation / geometry lookup, fp16-accumulate convolutions, fp16."""

    Params = Params
    _GRAPH_PIXELS = 384 * 640  # fixed: cap the pool at the measured launch-bound sizes

    def __init__(self, ctx: Context) -> None:
        """Build the network and the kernel state."""
        super().__init__(ctx)
        from .kernels.inference import forward_kernels, prepare_kernels  # noqa: PLC0415 - needs the kernel libraries

        self.net = _net(ctx)
        prepare_kernels(self.net, ctx.device)
        self._forward = forward_kernels
        self._graphs = SmallShapeGraphs(
            self._disparity,
            pixels=self._GRAPH_PIXELS,
            enabled=ctx.params.cuda_graphs,
            capture_after=cudnn_hacc.capture_after(),
            one_shape=True,
            capture_warmup=cudnn_hacc.BUILD_AFTER > 0,
        )

    def _disparity(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self._forward(self.net, left, right, iters=self.ctx.params.iters)[:, 0]

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        return {"disparity": self._graphs(left, right)}

    def release_graphs(self) -> None:
        """Free the CUDA graph and its memory pool."""
        self._graphs.release()


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


class _MlxPortable(Runner):
    """The network in MLX operations (`mlx_net`: no exported graph, so no trace on a first call), one pair per call."""

    batch = 1
    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the PyTorch network on the CPU and copy its weights to MLX."""
        super().__init__(ctx)
        from . import mlx_net  # noqa: PLC0415 - imports mlx

        self.net = mlx_net.MlxIGEV(_net(ctx), mlx_wrap.run_dtype(ctx), ctx.params.iters)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - imports mlx

        return {"disparity": to_torch(self.net(to_mx(left.float()), to_mx(right.float())))}


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "kernels_mps": MetalKernels,
    "portable_mlx": mlx_wrap.port(_MlxPortable, attrs=(), cache_mb=512),
}
