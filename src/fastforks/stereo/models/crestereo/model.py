"""CREStereo runners: MegEngine weights -> the translated network and the portable / kernels forward passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastforks._core.checkpoint import NUMPY_ARRAYS
from fastforks.kernels import cudnn_hacc
from fastforks.shared.cuda_graphs import SmallShapeGraphs
from fastforks.stereo._runner import Forward, Runner

from .net import Model
from .portable import Ops, forward_fast, prepare

if TYPE_CHECKING:
    from types import ModuleType

    import torch

    from fastforks.stereo._runner import Context


@dataclass(frozen=True)
class Params:
    """Load options (card `[options]`)."""

    iters: int
    cuda_graphs: bool  # kernels replays CUDA graphs for small repeated shapes


def _net(ctx: Context) -> Model:
    """The network with the authors' MegEngine checkpoint (a plain pickle of numpy arrays, read without MegEngine)."""
    state = ctx.state_dict(key="state_dict", strip="", extra=NUMPY_ARRAYS)
    # MegEngine stores conv biases as [1, C, 1, 1]
    state = {k: v.reshape(-1) if k.endswith(".bias") and v.dim() == 4 else v for k, v in state.items()}  # noqa: PLR2004
    net = Model(max_disp=256, mixed_precision=False, test_mode=True)  # upstream test.py load_model
    net.load_state_dict(state)
    return net.eval().to(ctx.device)


class Portable(Runner):
    """Plain PyTorch: upstream `test.py` inference without the unused work, fp16 convolutions on CUDA."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        prepare(self.net, half=ctx.dtype == "float16")

    def _run(self, net: Model, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return forward_fast(net, left, right, self.ctx.params.iters)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded batch."""
        return {"disparity": self._run(self.net, left, right)}

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        return Forward(self.net, self._run)


class MetalKernels(Portable):
    """Apple MPS: the portable network with the iterative group correlation and the GRU gates as Metal kernels."""

    module = Runner.module  # the Metal kernels are not exportable

    def __init__(self, ctx: Context) -> None:
        """Build the network and load the kernel."""
        super().__init__(ctx)
        from fastforks._core import mps  # noqa: PLC0415 - MPS only

        mps.adapt(self.net)  # the faster / fp16-capable transposed convolutions, as for the portable runner
        from fastforks.kernels import stereo_metal  # noqa: PLC0415 - needs the kernel library
        from fastforks.kernels.stereo_metal.modules import CrestereoOps  # noqa: PLC0415

        ops = CrestereoOps(stereo_metal.load(ctx.device))
        self.ops = Ops(ops.corr_iter, ops.gru)

    def _run(self, net: Model, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return forward_fast(net, left, right, self.ctx.params.iters, self.ops)


_GRAPH_PIXELS = 1_000_000  # fixed: batch x padded H x W; launch-bound up to 1280x720 (2 x 20 GRU passes), measured
_GRAPHS = 2  # fixed: graph cache policy


class Kernels(Runner):
    """Our kernels: fused-norm encoder, adaptive group correlation, fused GRU epilogues, fp16-accumulate convs.

    A repeated input shape up to `_GRAPH_PIXELS` replays a CUDA graph (`fastforks.shared.cuda_graphs`): the eager
    pass is launch-bound there.
    """

    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network and the kernel state."""
        super().__init__(ctx)
        from .kernels.inference import Engine  # noqa: PLC0415 - needs the kernel libraries

        self.engine = Engine(_net(ctx), ctx.device)
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
        return self.engine(left, right, self.ctx.params.iters)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        return {"disparity": self._graphs(left, right)}


class MlxPortable(Runner):
    """Apple MLX (`device="mlx"`): the network in MLX array operations (`mlx_net`, `mlx_ops`)."""

    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Convert or read the cached weights and build the MLX network."""
        super().__init__(ctx)
        from . import mlx_net  # noqa: PLC0415 - needs the mlx extra

        self.engine = mlx_net.Engine(ctx, self._ops())

    def _ops(self) -> ModuleType:
        from . import mlx_ops  # noqa: PLC0415 - needs the mlx extra

        return mlx_ops

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        return {"disparity": self.engine(left, right, self.ctx.params.iters)}


class MlxKernels(MlxPortable):
    """Apple MLX: correlation, GRU (Winograd 1x5 / 5x1), instance norm and upsampling as Metal kernels run by MLX."""

    def _ops(self) -> ModuleType:
        from fastforks.kernels.crestereo_mlx import ops  # noqa: PLC0415 - needs the mlx extra

        return ops


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "kernels_mps": MetalKernels,
    "portable_mlx": MlxPortable,
    "kernels_mlx": MlxKernels,
}
