"""FlowSeek runners: weights -> network and the portable forward pass."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import check_fits
from fastforks._core.half import half_weights
from fastforks.flow._runner import Forward, Runner
from fastforks.kernels import flow_upsample, flowseek_corr
from fastforks.shared import mlx_wrap

from .net import FlowSeek

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks._core.kernels import CubinModule
    from fastforks.flow._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[options]`."""

    iters: int  # flow updates after the initial regression (upstream `iters`; its "S" is T with 12)


class Portable(Runner):
    """Plain PyTorch; on CUDA in fp16 the networks run under autocast, motion bases and correlation in fp32.

    The card says `pad = "none"`: the network pads itself (upstream pads after Depth Anything has seen the unpadded
    images and the motion bases are made at the image size).
    """

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = FlowSeek()
        # upstream stores each shortcut batch norm twice (`bn3` and `downsample.1`, the same module); net.py has one
        sd = {k: v for k, v in ctx.state_dict().items() if ".downsample.1." not in k}
        check_fits(self.net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
        self.net.load_state_dict(sd)
        self.net.fp16 = ctx.dtype == "float16"
        self.net.eval().to(ctx.device)
        if ctx.device.type == "cuda" and self.net.fp16:
            # channels-last weights: cuDNN's fp16 kernels skip the NCHW<->NHWC conversion around each conv
            # (375x1242 85.3 -> 77.8 ms, peak 688 -> 629 MB RTX 5060 Ti, Windows); float32 stays NCHW
            self.net.to(memory_format=torch.channels_last)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a batch."""
        return {"flow": self.net(image1, image2, self.ctx.params.iters)}

    def module(self) -> torch.nn.Module:
        """The network; returns the flow."""
        iters = self.ctx.params.iters
        return Forward(self.net, lambda net, image1, image2: net(image1, image2, iters))


def _lookup(k: CubinModule, f1: torch.Tensor, f2: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Prepare native-width channels-last feature maps for the four pyramid levels."""
    b, c, h, w = f1.shape
    if c != 384 or f2.shape != f1.shape:  # noqa: PLR2004 - FlowSeek-T architecture
        msg = "FlowSeek-T needs two matching 384-channel feature maps"
        raise ValueError(msg)
    left = f1.to(dtype=torch.float16, memory_format=torch.channels_last)
    levels = [f2.to(dtype=torch.float16, memory_format=torch.channels_last)]
    for _ in range(3):
        levels.append(F.avg_pool2d(levels[-1], 2))

    def lookup(coords: torch.Tensor) -> torch.Tensor:
        if coords.shape != (b, 2, h, w) or coords.device != left.device or coords.dtype != torch.float32:
            msg = "FlowSeek coordinates must be float32 BCHW pixel coordinates on the feature device"
            raise ValueError(msg)
        out = left.new_empty((b, h, w, 324))
        k.launch("lookup", (h * w, b, 4), (128, 1, 1), left, *levels, coords.contiguous(), out, h, w, 1 / math.sqrt(c))
        return out.permute(0, 3, 1, 2)

    return lookup


class Kernels(Portable):
    """Portable FP16 network with native-width on-demand correlation and fused convex flow upsampling."""

    def __init__(self, ctx: Context) -> None:
        """Build the network and load the CUDA correlation kernel."""
        super().__init__(ctx)
        half_weights(self.net)  # reuse cast weights across the depth encoder and recurrent updates
        self.k = flowseek_corr.load(ctx.device)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a batch, with on-demand correlation."""
        return {
            "flow": self.net(
                image1, image2, self.ctx.params.iters, lambda a, b: _lookup(self.k, a, b), flow_upsample.convex
            )
        }

    module = Runner.module


class _MlxPortable(Runner):
    """Apple MLX: the network written in MLX operations (`mlx_net`) from the checkpoint's tensors, one pair per call."""

    batch = 1  # one pair per call (`mlx_net`)
    dtypes = ("float32", "float16")
    Params = Params
    module = Runner.module  # the MLX network is not exportable
    CACHE_MB: ClassVar[dict[str, int]] = {"*": 1024}  # the buffer cache this mode asks for, by variant ('*': all)

    def __init__(self, ctx: Context) -> None:
        """The MLX network from the conversion cache (the first load converts the checkpoint's weights)."""
        super().__init__(ctx)
        from fastforks.shared import mlx_port  # noqa: PLC0415 - imports mlx

        from . import mlx_net  # noqa: PLC0415 - imports mlx

        iters, path = ctx.params.iters, ctx.weights()[0]
        self.mlx = mlx_port.recorded(
            ctx,
            mlx_net.LAYOUT,
            lambda sd: mlx_net.MlxFlowSeek(sd, ctx.dtype, iters),
            real=ctx.state_dict,
            skeleton=lambda: mlx_port.header_skeleton(path),
            sources=mlx_net.SOURCES,
        )
        if mb := self.CACHE_MB.get(ctx.variant.id, self.CACHE_MB.get("*", 0)):  # held while this runner lives
            from fastforks._core.mlx import request_cache  # noqa: PLC0415 - MLX only

            request_cache(self, mb)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of an (unpadded) pair."""
        from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - imports mlx

        return {"flow": to_torch(self.mlx(to_mx(image1.float()), to_mx(image2.float())))}


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": _MlxPortable,
}
# FASTFORKS_MLX_GRAPH=1 (diagnostics, tools/mlx_core_ab.py): the exported-graph port the hand-written network replaced
GRAPH_RUNNERS = {"portable_mlx": mlx_wrap.port(Portable, cache_mb=1024)}
