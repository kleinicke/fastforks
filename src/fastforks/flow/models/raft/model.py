"""RAFT runners: weights -> network and the portable forward pass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from fastforks._core.checkpoint import check_fits
from fastforks.flow._runner import Forward, Runner
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import SmallShapeGraphs

from .mlx import MlxRAFT
from .net import LARGE, RAFT, SMALL

if TYPE_CHECKING:
    import torch

    from fastforks.flow._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` and `[options]`."""

    size: Literal["large", "small"]  # torchvision raft_large / raft_small
    iters: int  # flow updates (torchvision's `num_flow_updates`)
    cuda_graphs: bool  # option: kernels replays CUDA graphs for small repeated shapes


class Portable(Runner):
    """Plain PyTorch; on CUDA in fp16 the encoders and the update block run under autocast, the correlation in fp32."""

    dtypes = ("float32", "float16")
    Params = Params
    batch = None
    # pixels (pairs x H x W) per network call: the eager call is launch bound at small sizes (RTX 5060 Ti, Windows,
    # fp16, 8 pairs 480 x 640: small 43.1 -> 13.7 ms per pair, large 50.3 -> 34.2), not at 720p (small 51 -> 59 ms
    # and 9.4 GB for 8 pairs a call)
    batch_pixels = 1_300_000

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = RAFT(LARGE if ctx.params.size == "large" else SMALL)
        sd = ctx.state_dict()
        check_fits(self.net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
        self.net.load_state_dict(sd)
        self.net.fp16 = ctx.dtype == "float16"
        self.net.eval().to(ctx.device)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a padded batch."""
        return {"flow": self.net(image1, image2, self.ctx.params.iters)}

    def module(self) -> torch.nn.Module:
        """The network; returns the flow."""
        iters = self.ctx.params.iters
        return Forward(self.net, lambda net, image1, image2: net(image1, image2, iters))


_GRAPH_PIXELS = 1_000_000  # fixed: batch x padded H x W; the engine's ~1000 launches per pass bound small inputs
_GRAPHS = 2  # fixed: graph cache policy


class Kernels(Runner):
    """Our CUDA engine (`kernels/engine.py`): no correlation volume, fused GRU and conv epilogues.

    The lookup computes the correlation on the fly from a pooled fmap2 pyramid (O(HW) memory); fp16 activations with
    fp16-accumulate convolutions, fp32 correlation sums and flow state. A repeated input shape up to `_GRAPH_PIXELS`
    replays a CUDA graph (`fastforks.shared.cuda_graphs`): the eager pass is launch-bound there.
    """

    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the engine."""
        super().__init__(ctx)
        from fastforks.kernels import cudnn_hacc, cutlass_conv, raft_flow  # noqa: PLC0415 - needs the kernel libraries

        from .kernels.engine import Engine  # noqa: PLC0415

        d = ctx.device
        sd = ctx.state_dict()
        check_fits(
            RAFT(LARGE if ctx.params.size == "large" else SMALL).state_dict(), sd, f"{ctx.variant.ref} checkpoint"
        )
        self.engine = Engine(
            sd, d, raft_flow.load(d), cutlass_conv.load(d), cudnn_hacc.load(d), ctx.params.iters,
            large=ctx.params.size == "large",
        )  # fmt: skip
        self._graphs = SmallShapeGraphs(
            self.engine,
            pixels=_GRAPH_PIXELS,
            graphs=_GRAPHS,
            enabled=ctx.params.cuda_graphs,
            capture_after=cudnn_hacc.capture_after(),
        )

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a padded batch, pair by pair."""
        return {"flow": self._graphs(image1, image2)}

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        self._graphs.release()


class MetalKernels(Portable):
    """Apple MPS: the portable network with the correlation volume replaced by a Metal window lookup (`flow_metal`).

    The pyramid keeps only fmap1 and the pooled fmap2 levels (O(HW) memory instead of O((HW)^2)); each lookup takes
    its dot products on the fly. Encoders and update block as portable.
    """

    batch = 1  # one pair per call, as before batching (not measured on Apple GPUs)

    module = Runner.module  # the Metal kernels are not exportable

    def __init__(self, ctx: Context) -> None:
        """Build the network; load the Metal library."""
        super().__init__(ctx)
        from fastforks.kernels import flow_metal  # noqa: PLC0415 - needs the kernel library

        self.lib = flow_metal.load(ctx.device)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a padded batch."""
        from .kernels.metal import forward  # noqa: PLC0415

        return {"flow": forward(self.net, self.lib, image1, image2, self.ctx.params.iters)}


class _Mlx(Portable):
    """The network the MLX port exports: `MlxRAFT` (`mlx.py`), without the correlation volume; one pair per call."""

    batch = 1

    def __init__(self, ctx: Context) -> None:
        """Build the network; regroup it for MLX."""
        super().__init__(ctx)
        self.net = MlxRAFT(cast("RAFT", self.net))  # `Portable`'s network


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "kernels_mps": MetalKernels,
    "portable_mlx": mlx_wrap.port(_Mlx, cache_mb=1024),
}
