"""RT-MonSter++ runner: weights -> network and the portable forward pass."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastforks.kernels import cudnn_hacc
from fastforks.shared.cuda_graphs import GraphCache
from fastforks.stereo._runner import Forward, Runner

from .net.inference import forward_fast, prepare
from .net.monster import Monster

if TYPE_CHECKING:
    import torch

    from fastforks.stereo._runner import Context


@dataclass(frozen=True)
class Params:
    """Architecture (card `[params]`)."""

    encoder: str  # DINOv2 encoder of the Depth Anything V2 branch
    max_disp: int
    cuda_graphs: bool  # option: kernels replays CUDA graphs for small repeated shapes


def _net(ctx: Context) -> Monster:
    args = types.SimpleNamespace(  # upstream RT-MonSter++/evaluate_stereo.py defaults
        encoder=ctx.params.encoder,
        hidden_dims=[32, 64, 96],
        corr_radius=[2, 2, 4],
        n_downsample=2,
        n_gru_layers=3,
        max_disp=ctx.params.max_disp,
        mixed_precision=False,  # the CUDA path runs fp16 weights + autocast itself
    )
    net = Monster(args)
    net.load_state_dict(ctx.state_dict(key="state_dict"))  # {"state_dict": ...} checkpoint
    return net.eval().to(ctx.device)


class Portable(Runner):
    """Plain PyTorch: fp16 weights + fp16 autocast (CUDA) or fp32, NHWC on CUDA, SDPA ViT attention."""

    deterministic_cudnn = True

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        prepare(self.net, half=ctx.dtype == "float16", channels_last=ctx.device.type == "cuda")

    @staticmethod
    def _run(net: Monster, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return forward_fast(net, left, right)[:, 0]

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded batch."""
        return {"disparity": self._run(self.net, left, right)}

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        return Forward(self.net, self._run)


# Small inputs are launch-bound (the kernels path is ~1100 launches, ~32 ms of host time at 375x1242, more than its GPU
# time), so a repeated shape replays a CUDA graph (`fastforks.shared.cuda_graphs`). Above `_GRAPH_PIXELS` (batch x
# padded H x W) the GPU is the limit and the graph's activation pool would only cost memory; `_GRAPHS` bounds the
# cached shapes. Option `cuda_graphs = false` turns it off; `release_graphs()` frees the pool.
_GRAPH_PIXELS = 700_000  # fixed: launch-bound threshold, measured
_GRAPHS = 2  # fixed: graph cache policy


class Kernels(Runner):
    """Our kernels: the portable fp16 path with the fp16 DINOv2 and the convs on the fp16-accumulate kernels.

    The 4th call at a small shape (<= `_GRAPH_PIXELS`) captures a CUDA graph of the forward; later calls at it
    replay the graph (31 vs 41 ms at 375x1242, ~1.1 GB more reserved memory until `release_graphs()`).
    """

    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network and load the kernels."""
        super().__init__(ctx)
        from .kernels.inference import forward_kernels, prepare_kernels  # noqa: PLC0415 - needs the kernel libraries

        net = self.net = prepare_kernels(_net(ctx), ctx.device)
        self._forward = forward_kernels
        self._cache = (
            GraphCache(
                lambda left, right: forward_kernels(net, left, right),  # not `self`: no cycle, freed by `del`
                graphs=_GRAPHS,
                capture_after=cudnn_hacc.capture_after(),
            )
            if ctx.params.cuda_graphs and ctx.device.type == "cuda"
            else None
        )

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        if self._cache is not None:
            self._cache.release()

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded batch."""
        if self._cache is not None and left.shape[0] * left.shape[-2] * left.shape[-1] <= _GRAPH_PIXELS:
            key = (left.device, left.dtype, right.dtype, tuple(left.shape), tuple(right.shape))
            out = self._cache.run(key, left, right)
            if out is not None:
                return {"disparity": out}
        return {"disparity": self._forward(self.net, left, right)}


RUNNERS = {"portable": Portable, "kernels": Kernels}
