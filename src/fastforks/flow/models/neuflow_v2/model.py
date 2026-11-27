"""NeuFlow v2 runners: weights -> network, the portable forward pass and the kernels one (no correlation volume)."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from fastforks._core.checkpoint import check_fits
from fastforks.flow._runner import Forward, Runner
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import SmallShapeGraphs

from .net import RADIUS, NeuFlow, _grid

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks._core.kernels import CubinModule
    from fastforks.flow._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[options]`."""

    iters: int  # refinements at 1/8 resolution (upstream `iters_s8`)
    cuda_graphs: bool  # option: kernels replays CUDA graphs for small repeated shapes


class Portable(Runner):
    """Plain PyTorch; in fp16 the convolutions and the cross-attention run under autocast, the volume is fp16."""

    dtypes = ("float32", "float16")
    Params = Params
    batch = None
    # pixels (pairs x H x W) per network call: the eager call is launch bound at small sizes (RTX 5060 Ti, Windows,
    # fp16, 8 pairs 480 x 640: 22.1 -> 7.9 ms per pair), not at 720p (30.2 vs 29.7 ms, and 4 GB for 8 pairs)
    batch_pixels = 1_300_000

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = NeuFlow()
        sd = ctx.state_dict(key="model")
        check_fits(self.net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
        self.net.load_state_dict(sd)
        self.net.fp16 = ctx.dtype == "float16"
        self.net.eval().to(ctx.device)
        if ctx.device.type == "cuda" and self.net.fp16:
            # channels-last weights: cuDNN's fp16 kernels skip the NCHW<->NHWC conversion around each conv
            # (1080p portable 84.8 -> 82.1 ms, kernels 71.7 -> 68.8 ms and 652 -> 517 MB; RTX 5060 Ti, Windows)
            self.net.to(memory_format=torch.channels_last)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a padded batch."""
        return {"flow": self.net(image1, image2, self.ctx.params.iters)}

    def module(self) -> torch.nn.Module:
        """The network; returns the flow."""
        iters = self.ctx.params.iters
        return Forward(self.net, lambda net, image1, image2: net(image1, image2, iters))


def _kernel_lookup(k: CubinModule, features: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """The 9 x 9 window of the correlation computed on the fly by `raft_flow`'s lookup: O(HW) memory, no volume.

    Same operands as the portable path (the fp16 features), fp32 sums and bilinear weights; the window is written in
    fp16, which the refinement convolutions cast it to anyway.
    """
    f0, f1 = (f.to(torch.float16).permute(0, 2, 3, 1).contiguous() for f in features.chunk(2, dim=0))
    b, h, w, c = f0.shape
    side = (2 * RADIUS + 1) ** 2

    def lookup(flow: torch.Tensor) -> torch.Tensor:
        coords = (_grid(flow) + flow).permute(0, 2, 3, 1).contiguous()
        out = torch.empty(b, h, w, side, dtype=torch.float16, device=flow.device)
        for i in range(b):  # one warp per pixel, 8 warps per block
            k.launch(
                "corr_lookup_c128_r4_l1",
                (-(-h * w // 8), 1, 1),
                (256, 1, 1),
                f0[i],
                f1[i],
                h,
                w,
                coords[i],
                out[i],
                side,
                1 / math.sqrt(c),
            )
        return out.permute(0, 3, 1, 2)

    return lookup


_GRAPH_PIXELS = 1_000_000  # fixed: batch x padded H x W; launch-bound up to 544x960, 720x1280 still gains 6 %
_GRAPHS = 2  # fixed: graph cache policy


class Kernels(Portable):
    """The portable fp16 pass with the correlation window computed on the fly (`raft_flow`): no all-pairs volume.

    A repeated input shape up to `_GRAPH_PIXELS` replays a CUDA graph (`fastforks.shared.cuda_graphs`): the eager
    pass is launch-bound there.
    """

    batch_pixels = _GRAPH_PIXELS  # each chunk of a batch replays a graph

    def __init__(self, ctx: Context) -> None:
        """Build the network and bind the lookup kernel."""
        super().__init__(ctx)
        from fastforks.kernels import raft_flow  # noqa: PLC0415 - needs the kernel library

        self.net.correlate = functools.partial(_kernel_lookup, raft_flow.load(ctx.device))
        self._graphs = SmallShapeGraphs(self._run, pixels=_GRAPH_PIXELS, graphs=_GRAPHS, enabled=ctx.params.cuda_graphs)

    def _run(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        return self.net(image1, image2, self.ctx.params.iters)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a padded batch."""
        return {"flow": self._graphs(image1, image2)}

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        self._graphs.release()

    def module(self) -> torch.nn.Module:
        """No exportable module: the lookup is a kernel launch."""
        return Runner.module(self)


class _OnePair(Portable):
    """Portable for the MLX graph port: one pair per call, as before batching (not measured on Apple GPUs)."""

    batch = 1


class _MlxPortable(Runner):
    """Apple MLX: the network written in MLX operations (`mlx_net`) from the checkpoint's tensors, one pair per call."""

    batch = 1  # one pair per call (`mlx_net`)
    dtypes = ("float32", "float16")
    Params = Params
    module = Runner.module  # the MLX network is not exportable
    CACHE_MB: ClassVar[dict[str, int]] = {"*": 512}  # the buffer cache this mode asks for, by variant ('*': all)

    def __init__(self, ctx: Context) -> None:
        """The MLX network from the conversion cache (the first load converts the checkpoint's weights)."""
        super().__init__(ctx)
        from fastforks.shared import mlx_port  # noqa: PLC0415 - imports mlx

        from . import mlx_net  # noqa: PLC0415 - imports mlx

        iters = ctx.params.iters
        self.mlx = mlx_port.recorded(
            ctx,
            mlx_net.LAYOUT,
            lambda sd: mlx_net.MlxNeuFlow(sd, ctx.dtype, iters),
            real=lambda: ctx.state_dict(key="model"),
            skeleton=lambda: NeuFlow().state_dict(),
            sources=mlx_net.SOURCES,
        )
        if mb := self.CACHE_MB.get(ctx.variant.id, self.CACHE_MB.get("*", 0)):  # held while this runner lives
            from fastforks._core.mlx import request_cache  # noqa: PLC0415 - MLX only

            request_cache(self, mb)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a padded pair."""
        from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - imports mlx

        return {"flow": to_torch(self.mlx(to_mx(image1.float()), to_mx(image2.float())))}


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": _MlxPortable,
}
# FASTFORKS_MLX_GRAPH=1 (diagnostics, tools/mlx_core_ab.py): the exported-graph port the hand-written network replaced
GRAPH_RUNNERS = {"portable_mlx": mlx_wrap.port(_OnePair, cache_mb=512)}
