"""RIFE runners: Practical-RIFE's `IFNet_HDv3` (releases 4.25 / 4.26) and its `inference` call.

As upstream `RIFE_HDv3.Model.inference`: the two frames in `[0, 1]` are concatenated, five IFBlocks refine the
bidirectional flow and the fusion mask from 1/16 (lite: 1/32) to full resolution (`scale` 1), and the frame is the
mask-weighted blend of both frames warped to the time. Upstream's `inference_video.py` pads to a multiple of 128 with
zeros at the right and bottom; the card's `[io] pad` does the same.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from fastforks._core.checkpoint import check_fits
from fastforks._core.half import fp16_accumulate
from fastforks.interpolation._runner import Forward, Runner
from fastforks.kernels import cudnn_hacc
from fastforks.shared.cuda_graphs import GraphCache

from .net import IFNet
from .net.IFNet_HDv3 import ResConv

if TYPE_CHECKING:
    from fastforks.interpolation._runner import Context

_TRAINING_ONLY = ("teacher.", "caltime.")  # in the checkpoints, unused at inference (upstream loads with strict=False)


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params` + `[options]`."""

    scales: tuple[int, ...]  # the five IFBlocks' downscale factors (upstream `scale_list` at `scale` 1)
    c4: int  # width of the last IFBlock
    cuda_graphs: bool


class Portable(Runner):
    """Plain PyTorch in the card's dtype (the whole network; the warps sample in float32)."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = IFNet(c4=ctx.params.c4)
        sd = {k: v for k, v in ctx.state_dict().items() if not k.startswith(_TRAINING_ONLY)}
        check_fits(self.net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
        self.net.load_state_dict(sd)
        self.net.eval().to(ctx.device, getattr(torch, ctx.dtype))

    def _forward(self, x: torch.Tensor, timestep: float | torch.Tensor) -> torch.Tensor:
        return self.net(x, timestep, list(self.ctx.params.scales))[2][-1]

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor, timestep: float) -> dict[str, torch.Tensor]:
        """The frame at `timestep` of a padded batch."""
        dtype = getattr(torch, self.ctx.dtype)
        return {"frame": self._forward(torch.cat((image1, image2), 1).to(dtype), timestep)}

    def module(self) -> torch.nn.Module:
        """The network; returns the middle frame (time 0.5)."""
        scales, dtype = list(self.ctx.params.scales), getattr(torch, self.ctx.dtype)
        return Forward(
            self.net, lambda net, image1, image2: net(torch.cat((image1, image2), 1).to(dtype), 0.5, scales)[2][-1]
        )


def _hacc_conv(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    return cudnn_hacc.conv_half_acc(x, conv.weight, conv.bias, conv.padding, conv.stride, conv.dilation)  # ty: ignore[invalid-argument-type]


def _hacc_deconv(deconv: torch.nn.ConvTranspose2d, x: torch.Tensor) -> torch.Tensor:
    y = cudnn_hacc.deconv_half_acc(x, deconv.weight, deconv.padding, deconv.stride)  # ty: ignore[invalid-argument-type]
    return y.add_(deconv.bias.view(1, -1, 1, 1))  # ty: ignore[unresolved-attribute]


def _res_conv(m: ResConv, x: torch.Tensor) -> torch.Tensor:
    return m.relu(m.conv(x) + x)  # beta folded into the conv


class Kernels(Portable):
    """float16, channels-last, fp16-accumulate convs and transposed convs (`cudnn_hacc`), one CUDA graph per shape."""

    module = Runner.module  # fp16-accumulate convolutions are cuDNN plans, not exportable operators

    def __init__(self, ctx: Context) -> None:
        """Build the network, fold the residual scales, route the convolutions through `cudnn_hacc`."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        for m in self.net.modules():
            if isinstance(m, ResConv):
                beta = m.beta.data.view(-1)
                m.conv.weight.data.mul_(beta.view(-1, 1, 1, 1))
                m.conv.bias.data.mul_(beta)  # ty: ignore[unresolved-attribute]
                m.forward = functools.partial(_res_conv, m)  # ty: ignore[invalid-assignment]  # per-instance override
            elif type(m) is torch.nn.Conv2d:
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
                m.forward = functools.partial(_hacc_conv, m)  # ty: ignore[invalid-assignment]
            elif type(m) is torch.nn.ConvTranspose2d:
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
                m.forward = functools.partial(_hacc_deconv, m)  # ty: ignore[invalid-assignment]
        after = cudnn_hacc.capture_after()
        self.graphs = GraphCache(self._eager, graphs=1, capture_after=after) if ctx.params.cuda_graphs else None

    def _eager(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        with fp16_accumulate():
            return self._forward(x, timestep)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor, timestep: float) -> dict[str, torch.Tensor]:
        """The frame at `timestep` of a padded batch (the time is a graph input: one graph for every time)."""
        x = torch.cat((image1, image2), 1).to(torch.float16).contiguous(memory_format=torch.channels_last)
        t = torch.full((x.shape[0], 1, 1, 1), timestep, dtype=x.dtype, device=x.device)
        out = None if self.graphs is None else self.graphs.run((x.device, tuple(x.shape)), x, t)
        return {"frame": self._eager(x, t) if out is None else out}

    def release_graphs(self) -> None:
        """Free the CUDA graph and its memory pool."""
        if self.graphs is not None:
            self.graphs.release()


RUNNERS = {"portable": Portable, "kernels": Kernels}
