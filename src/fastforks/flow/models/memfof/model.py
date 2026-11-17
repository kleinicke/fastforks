"""MEMFOF runners: weights -> network and the forward pass of a pair."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import check_fits
from fastforks.flow._runner import Forward, Runner

from .net.model import MEMFOF

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks._core.kernels import CubinModule
    from fastforks.flow._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` and `[options]`."""

    iters: int  # flow updates after the initial regression (upstream `iters`)
    scale: Literal[0, 1]  # upstream `get_val_scale`: 1 = the frames are upsampled 2x (its KITTI / Sintel evaluation)


def _net(ctx: Context) -> MEMFOF:
    net = MEMFOF()
    sd = ctx.state_dict()
    check_fits(net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
    net.load_state_dict(sd)
    return net.eval().to(ctx.device)


class _Scaled(Runner):
    """Upstream `scale_and_forward_flow`: the network sees the pair resized by `2^scale`, the flow is resized back."""

    def __init__(self, ctx: Context) -> None:
        """Check the card's `scale`."""
        super().__init__(ctx)
        if ctx.params.scale not in (0, 1):
            msg = f"{ctx.variant.ref}: card scale must be 0 or 1, not {ctx.params.scale!r}"
            raise ValueError(msg)

    def resize(self, x: torch.Tensor) -> torch.Tensor:
        """Double the pair (bilinear) when the card says `scale = 1`."""
        if self.ctx.params.scale == 0:
            return x
        return F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)

    def restore(self, flow: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:  # noqa: ARG002 - 2x then 0.5x: same size
        """Halve the flow field (bilinear) and its values when the card says `scale = 1`."""
        if self.ctx.params.scale == 0:
            return flow
        return F.interpolate(flow.float(), scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5


class Portable(_Scaled):
    """Plain PyTorch; on CUDA in fp16 the encoders and the update block run under autocast, the correlation in fp32."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        self.fp16 = ctx.dtype == "float16"

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a padded batch."""
        return {"flow": self.net.forward_pair(image1, image2, self.ctx.params.iters, fp16=self.fp16)}

    def module(self) -> torch.nn.Module:
        """The network; returns the flow."""
        iters, fp16 = self.ctx.params.iters, self.fp16
        return Forward(self.net, lambda net, image1, image2: net.forward_pair(image1, image2, iters, fp16=fp16))


def _kernel_lookup(k: CubinModule, f1: torch.Tensor, f2: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """The 4-level 9 x 9 windows of the correlation computed on the fly by `raft_flow`: O(HW) memory, no volume.

    Upstream's pyramid halves fmap2 bilinearly (align_corners=False, scale 0.5), which is 2 x 2 average pooling, and
    pooling commutes with the dot product, so only the pooled fmap2 pyramid is kept (fp16). Operands are the fp16
    features with fp32 sums and bilinear weights; the windows are written in fp16, which the update block casts them to.
    """
    b, c, h, w = f1.shape
    a = f1.to(torch.float16).permute(0, 2, 3, 1).contiguous()
    pyr, x = [], f2.to(torch.float16)
    for _ in range(4):
        pyr.append(x.permute(0, 2, 3, 1).contiguous())
        x = F.avg_pool2d(x, 2)
    side = 4 * 81

    def lookup(coords: torch.Tensor) -> torch.Tensor:
        xy = coords.permute(0, 2, 3, 1).contiguous()
        out = torch.empty(b, h, w, side, dtype=torch.float16, device=coords.device)
        for i in range(b):  # one warp per pixel, 8 warps per block
            k.launch(
                "corr_lookup_c1024_r4",
                (-(-h * w // 8), 1, 1),
                (256, 1, 1),
                a[i],
                *(p[i] for p in pyr),
                h,
                w,
                xy[i],
                out[i],
                side,
                1 / math.sqrt(c),
            )
        return out.permute(0, 3, 1, 2)

    return lookup


class Kernels(Portable):
    """The portable fp16 pass with the correlation windows computed on the fly (`raft_flow`): no all-pairs volume."""

    def __init__(self, ctx: Context) -> None:
        """Build the network and bind the lookup kernel."""
        super().__init__(ctx)
        from fastforks.kernels import raft_flow  # noqa: PLC0415 - needs the kernel library

        self.lookup = functools.partial(_kernel_lookup, raft_flow.load(ctx.device))

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a padded batch."""
        iters = self.ctx.params.iters
        return {"flow": self.net.forward_pair(image1, image2, iters, corr_fn=self.lookup, fp16=True)}

    def module(self) -> torch.nn.Module:
        """No exportable module: the lookup is a kernel launch."""
        return Runner.module(self)


RUNNERS = {"portable": Portable, "kernels": Kernels}
