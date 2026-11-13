"""SEA-RAFT runners: weights -> network and the portable forward pass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import check_fits
from fastforks.flow._runner import Forward, Runner

from .net import MEDIUM, SEARAFT, SMALL

if TYPE_CHECKING:
    from fastforks.flow._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` and `[options]`."""

    size: Literal["small", "medium"]  # upstream S (resnet18 encoders) / M (resnet34 encoders)
    iters: int  # flow updates after the initial regression (upstream `iters`; its "L" is M with 12)
    scale: Literal[0, -1]  # upstream config/eval `scale`: 0 = input resolution, -1 = half (its Spring checkpoints)


def _net(size: str) -> SEARAFT:
    return SEARAFT(MEDIUM if size == "medium" else SMALL)


class _Scaled(Runner):
    """Upstream `calc_flow`: the network sees the pair resized by `2^scale`; the flow is resized back and rescaled."""

    def __init__(self, ctx: Context) -> None:
        """Check the card's `scale`."""
        super().__init__(ctx)
        if ctx.params.scale not in (0, -1):
            msg = f"{ctx.variant.ref}: card scale must be 0 or -1, not {ctx.params.scale!r}"
            raise ValueError(msg)

    def resize(self, x: torch.Tensor) -> torch.Tensor:
        """Halve the pair (bilinear) when the card says `scale = -1`."""
        if self.ctx.params.scale == 0:
            return x
        return F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)

    def restore(self, flow: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Upsample the half-resolution flow (bilinear) and double it; an odd size repeats the last row / column."""
        if self.ctx.params.scale == 0:
            return flow
        up = F.interpolate(flow.float(), scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        return F.pad(up, (0, size[1] - up.shape[-1], 0, size[0] - up.shape[-2]), mode="replicate")


class Portable(_Scaled):
    """Plain PyTorch; on CUDA in fp16 the encoders and the update block run under autocast, the correlation in fp32."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx.params.size)
        sd = ctx.state_dict()
        check_fits(self.net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
        self.net.load_state_dict(sd)
        self.net.fp16 = ctx.dtype == "float16"
        self.net.eval().to(ctx.device)
        # Channels-last on CUDA: -10 % (cuDNN's NHWC convolutions without layout copies; RAFT is slower this way).
        self.layout = torch.channels_last if ctx.device.type == "cuda" else torch.contiguous_format
        self.net.to(memory_format=self.layout)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a padded batch."""
        image1, image2 = (x.contiguous(memory_format=self.layout) for x in (image1, image2))
        return {"flow": self.net(image1, image2, self.ctx.params.iters)}

    def module(self) -> torch.nn.Module:
        """The network; returns the flow."""
        iters = self.ctx.params.iters
        return Forward(self.net, lambda net, image1, image2: net(image1, image2, iters))


class Kernels(_Scaled):
    """Our CUDA engine (`kernels/engine.py`): no correlation volume, fused conv epilogues, folded ConvNeXt blocks.

    The lookup computes the correlation on the fly from a pooled fmap2 pyramid (O(HW) memory); fp16 activations with
    fp16-accumulate convolutions, fp32 correlation sums and flow state.
    """

    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the engine."""
        super().__init__(ctx)
        from fastforks.kernels import cudnn_hacc, cutlass_conv, raft_flow  # noqa: PLC0415 - needs the kernel libraries

        from .kernels.engine import Engine  # noqa: PLC0415

        d = ctx.device
        sd = ctx.state_dict()
        net = _net(ctx.params.size)
        check_fits(net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
        self.engine = Engine(
            sd, net.cfg, d, raft_flow.load(d), cutlass_conv.load(d), cudnn_hacc.load(d), ctx.params.iters
        )

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a padded batch, pair by pair."""
        return {"flow": self.engine(image1, image2)}


class MetalKernels(Portable):
    """Apple MPS: the portable network with the correlation volume replaced by a Metal window lookup (`flow_metal`).

    The pyramid keeps only fmap1 and the halved fmap2 levels (O(HW) memory instead of O((HW)^2)); each lookup takes
    its dot products on the fly. Encoders and update block as portable.
    """

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


RUNNERS = {"portable": Portable, "kernels": Kernels, "kernels_mps": MetalKernels}
