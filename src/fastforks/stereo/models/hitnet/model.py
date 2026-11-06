"""HITNet runners: weights -> network and the portable / kernels forward passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.errors import WeightsError
from fastforks.shared.cuda_graphs import SmallShapeGraphs
from fastforks.stereo._runner import Forward, Runner

from .net import weights
from .net.hitnet import HITNet, Refine
from .net.pb import read_graph

if TYPE_CHECKING:
    from fastforks.stereo._runner import Context


@dataclass(frozen=True)
class Params:
    """Architecture of a variant (card `[params]`, identical to the numbers inside the authors' graph)."""

    in_ch: int
    channels: list[int]
    disparities: list[int]
    refine_channels: list[int]
    refine_dilations: list[list[int]]
    refine_sad_scale: list[float]
    pad_multiple: int
    pad_value: float
    pad_value_right: float
    cuda_graphs: bool  # option: kernels replays CUDA graphs for small repeated shapes

    def refine(self) -> list[Refine]:
        """The refinement stages (`refine_sad_scale` 0: no warp SAD)."""
        return [
            Refine(c, tuple(d), s or None)
            for c, d, s in zip(self.refine_channels, self.refine_dilations, self.refine_sad_scale, strict=True)
        ]


def _net(ctx: Context) -> HITNet:
    """The network with the weights of the graph file."""
    p = ctx.params
    net = HITNet(p.in_ch, p.channels, p.disparities, p.refine())
    path = ctx.weights()[0]
    try:
        state = weights.state_dict(read_graph(path), len(p.channels) - 1)
    except (KeyError, ValueError) as e:  # a graph without the expected nodes, or one the reader cannot decode
        msg = f"{path}: {e}"
        raise WeightsError(msg) from e
    net.load_state_dict(state)
    return net.eval().to(ctx.device)


class _HitnetRunner(Runner):
    Params = Params

    def padded(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Pad as the graph does (the core does not: `pad = "none"`); returns the images and the original H, W.

        Below and to the right up to a multiple, with the graph's constant for each image (the Middlebury graph pads
        the right image with +1).
        """
        h, w = left.shape[-2:]
        m = self.ctx.params.pad_multiple
        widths = (0, -w % m, 0, -h % m)
        p = self.ctx.params
        # the graph pads after scaling to [-1, 1]; the network takes 0..255
        return (
            F.pad(left, widths, value=(p.pad_value + 1.0) * 127.5),
            F.pad(right, widths, value=(p.pad_value_right + 1.0) * 127.5),
            h,
            w,
        )


class Portable(_HitnetRunner):
    """Plain PyTorch: float32 (CPU), or float16 convolutions with float32 hypotheses and warps (CUDA)."""

    dtypes = ("float32", "float16")

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        self.half = ctx.dtype == "float16"
        if ctx.device.type in ("cuda", "cpu"):  # oneDNN runs channels_last faster (no 3-D convs)
            self.net.to(memory_format=torch.channels_last)

    def _forward(self, net: HITNet, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left, right, h, w = self.padded(left, right)
        with torch.autocast(left.device.type, dtype=torch.float16, enabled=self.half):
            return net(left, right)[:, 0, :h, :w]

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of the pair (padded and cropped here)."""
        return {"disparity": self._forward(self.net, left, right)}

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        return Forward(self.net, self._forward)


_GRAPH_PIXELS = 500_000  # fixed: launch-bound up to KITTI 375x1242 (batch x H x W of the raw input), measured
_GRAPHS = 2  # fixed: graph cache policy


class Kernels(_HitnetRunner):
    """Our kernels: fused fp16-accumulate CUTLASS convolutions, cost / SAD kernels without intermediate volumes.

    A repeated input shape up to `_GRAPH_PIXELS` replays a CUDA graph (`fastforks.shared.cuda_graphs`):
    the eager pass is launch-bound there. The padding runs inside the feature-head kernel.
    """

    def __init__(self, ctx: Context) -> None:
        """Build the network and load the kernels."""
        super().__init__(ctx)
        from .kernels import KernelsHITNet  # noqa: PLC0415 - needs the kernel libraries

        self.net = _net(ctx)
        self.max = KernelsHITNet(self.net, ctx.device)
        self._graphs = SmallShapeGraphs(
            self._forward, pixels=_GRAPH_PIXELS, graphs=_GRAPHS, enabled=ctx.params.cuda_graphs
        )

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        self._graphs.release()

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of the pair (padded and cropped here)."""
        return {"disparity": self._graphs(left, right)}

    def _forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        h, w = left.shape[-2:]
        m, p = self.ctx.params.pad_multiple, self.ctx.params
        pads = ((p.pad_value + 1.0) * 127.5, (p.pad_value_right + 1.0) * 127.5)  # as `padded`, applied in the kernel
        return self.max(left, right, (h + -h % m, w + -w % m), pads)[:, 0, :h, :w]


class MetalKernels(Portable):
    """Apple MPS: the portable network with the level-init cost and the warp SADs as Metal kernels."""

    module = Runner.module  # the Metal kernels are not exportable

    def __init__(self, ctx: Context) -> None:
        """Build the network and switch its costs to the Metal kernels."""
        super().__init__(ctx)
        from fastforks._core import mps  # noqa: PLC0415 - MPS only
        from fastforks.kernels import stereo_metal  # noqa: PLC0415 - needs the kernel library
        from fastforks.kernels.stereo_metal.modules import HitnetCosts  # noqa: PLC0415

        mps.adapt(self.net)
        costs = HitnetCosts(stereo_metal.load(ctx.device))
        for m in self.net.modules():
            for name in ("init_cost", "tile_sad", "refine_sad"):
                if name in vars(m):
                    setattr(m, name, getattr(costs, name))


RUNNERS = {"portable": Portable, "kernels": Kernels, "kernels_mps": MetalKernels}
