"""RAFT-Stereo runners: weights -> network and the portable / kernels forward passes."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from fastforks._core.checkpoint import check_fits
from fastforks.stereo._runner import Forward, Runner

from .net.raft_stereo import RAFTStereo

if TYPE_CHECKING:
    from fastforks.stereo._runner import Context


@dataclass(frozen=True)
class Params:
    """Load options (card `[options]`)."""

    iters: int


def _args(*, half: bool = False, cuda: bool = False) -> types.SimpleNamespace:
    """The released (default) architecture."""
    return types.SimpleNamespace(
        hidden_dims=[128] * 3,
        n_downsample=2,
        n_gru_layers=3,
        corr_levels=4,
        corr_radius=4,
        context_norm="batch",
        shared_backbone=False,
        slow_fast_gru=False,
        corr_implementation="reg_gather",  # the CUDA-extension lookup in plain PyTorch
        mixed_precision=half,
        channels_last_loop=cuda,  # NHWC update loop (cuDNN tensor-core layout)
        fnet_sequential=True,  # one image at a time through the feature encoder: lower peak memory
    )


def check_weights(sd: dict[str, torch.Tensor], what: str) -> None:
    """Raise `CheckpointMismatchError` unless `sd` has exactly the keys and shapes of the network (built on meta)."""
    with torch.device("meta"):
        own = RAFTStereo(_args()).state_dict()
    check_fits(own, {k.removeprefix("module."): v for k, v in sd.items()}, what)


def _disparity(net: RAFTStereo, left: torch.Tensor, right: torch.Tensor, iters: int) -> torch.Tensor:
    _, flow = net(left, right, iters=iters, test_mode=True)
    return -flow[:, 0]


class Portable(Runner):
    """Plain PyTorch: upstream network with a gather-based correlation lookup, folded BatchNorm, fp16 weights."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        args = _args(half=ctx.dtype == "float16", cuda=ctx.device.type == "cuda")
        self.net = RAFTStereo(args)
        self.net.load_state_dict(ctx.state_dict())
        self.net.eval().to(ctx.device)
        if args.mixed_precision:
            self.net.half_conv_weights()  # folds BatchNorm, then fp16 conv weights once (autocast recasts every call)
        else:
            self.net.fold_batchnorm()

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded batch."""
        return {"disparity": _disparity(self.net, left, right, self.ctx.params.iters)}

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        iters = self.ctx.params.iters
        return Forward(self.net, lambda net, left, right: _disparity(net, left, right, iters))


class Kernels(Runner):
    """Our engine: fused kernels for scaling + padding, correlation, GRUs and convolutions (fp16)."""

    io = "raw"  # the `prep_image` kernel scales, pads and converts the layout in one pass
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the engine."""
        super().__init__(ctx)
        from fastforks.kernels import cudnn_hacc, cutlass_conv, raft  # noqa: PLC0415 - needs the kernel libraries

        from .kernels.engine import Engine  # noqa: PLC0415

        d = ctx.device
        sd = ctx.state_dict()
        check_weights(sd, f"{ctx.variant.ref} checkpoint")  # the converter reads the tensors it needs by name
        self.engine = Engine(sd, d, raft.load(d), cutlass_conv.load(d), cudnn_hacc.load(d), ctx.params.iters)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of one `[1, 3, H, W]` pair in `[0, 1]`."""
        return {"disparity": self.engine(left, right)[None]}


class MetalKernels(Portable):
    """Apple MPS: the portable network (fp16) with Metal kernels for the ConvGRUs and the correlation lookup."""

    module = Runner.module  # the Metal kernels are not exportable

    def __init__(self, ctx: Context) -> None:
        """Build the network and switch it to the Metal kernels."""
        super().__init__(ctx)
        from fastforks._core import mps  # noqa: PLC0415 - MPS only

        mps.adapt(self.net)  # the faster / fp16-capable transposed convolutions, as for the portable runner
        from fastforks.kernels import stereo_metal  # noqa: PLC0415 - needs the kernel library

        from .kernels.metal import patch  # noqa: PLC0415

        patch(self.net, stereo_metal.load(ctx.device))


RUNNERS = {"portable": Portable, "kernels": Kernels, "kernels_mps": MetalKernels}
