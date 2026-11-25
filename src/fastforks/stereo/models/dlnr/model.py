"""DLNR runner: weights -> network and the portable forward pass."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastforks.shared import mlx_wrap
from fastforks.stereo._runner import Forward, Runner

from .net.dlnr import DLNR
from .net.inference import forward, prepare

if TYPE_CHECKING:
    import torch

    from fastforks.stereo._runner import Context


@dataclass(frozen=True)
class Params:
    """Load options (card `[options]`)."""

    iters: int


def _args() -> types.SimpleNamespace:
    """Upstream evaluate_stereo.py defaults (the architecture part)."""
    return types.SimpleNamespace(hidden_dims=[128] * 3, corr_levels=4, corr_radius=4, n_downsample=2, n_gru_layers=3)


class Portable(Runner):
    """Plain PyTorch: fp16 NHWC (CUDA) or fp32, merged LSTM gates, NHWC Restormer blocks, gather correlation lookup."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        net = DLNR(_args())
        net.load_state_dict(ctx.state_dict())
        self.net = prepare(net.to(ctx.device), half=ctx.dtype == "float16")

    def _run(self, net: DLNR, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return forward(net, left, right, self.ctx.params.iters)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded batch."""
        return {"disparity": self._run(self.net, left, right)}

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        return Forward(self.net, self._run)


class Kernels(Runner):
    """Our kernels: fp16-accumulate / CUTLASS convs with fused epilogues (incl. the LSTM gates), raft correlation."""

    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network and the kernel state."""
        super().__init__(ctx)
        from .kernels.inference import forward_kernels, prepare_kernels  # noqa: PLC0415 - needs the kernel libraries

        net = DLNR(_args())
        net.load_state_dict(ctx.state_dict())
        self.net = net.to(ctx.device)
        self.k = prepare_kernels(self.net, ctx.device)
        self._forward = forward_kernels

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        return {"disparity": self._forward(self.net, self.k, left, right, self.ctx.params.iters)}


class _MlxPortable(Runner):
    """Apple MLX: the network written in MLX operations (`mlx_net`), one pair per call."""

    batch = 1  # not measured on Apple GPUs
    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the PyTorch network and copy its weights to MLX."""
        super().__init__(ctx)
        from . import mlx_net  # noqa: PLC0415 - imports mlx

        net = DLNR(_args())
        net.load_state_dict(ctx.state_dict())
        self.net = mlx_net.MlxDLNR(prepare(net, half=False), mlx_wrap.run_dtype(ctx), ctx.params.iters)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - imports mlx

        return {"disparity": to_torch(self.net(to_mx(left.float()), to_mx(right.float())))}


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(_MlxPortable, attrs=(), cache_mb=1024),
}
