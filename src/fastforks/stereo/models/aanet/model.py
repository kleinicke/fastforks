"""AANet / AANet+ runners: weights -> network and the portable / kernels forward passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks._core.half import pure_half
from fastforks.kernels import cudnn_hacc
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import SmallShapeGraphs
from fastforks.stereo._runner import Forward, Runner

from .net.aanet import AANet

if TYPE_CHECKING:
    from fastforks.stereo._runner import Context

# Intermediate-supervision heads (extra outputs of the last fusion block) that the network drops with
# `no_intermediate_supervision`; upstream loads with `strict=False` for them, we drop exactly these.
INTERMEDIATE = ("aggregation.fusions.5.fuse_layers.", "aggregation.final_conv.")


@dataclass(frozen=True)
class Params:
    """Architecture of a variant (variant `params`).

    Attributes:
        plus: AANet+ (GA-Net feature pyramid + hourglass refinement) instead of AANet (AANet feature extractor + FPN
            + StereoDRNet refinement).
        max_disp: Disparity range of the full-resolution cost volume.
    """

    plus: bool
    max_disp: int
    cuda_graphs: bool  # option: kernels replays CUDA graphs for small repeated shapes


def _net(ctx: Context) -> AANet:
    kw = (
        {"feature_type": "ganet", "feature_pyramid": True, "refinement_type": "hourglass"}
        if ctx.params.plus
        else {"feature_type": "aanet", "feature_pyramid_network": True, "refinement_type": "stereodrnet"}
    )
    net = AANet(
        ctx.params.max_disp,
        num_downsample=2,
        no_feature_mdconv=False,
        feature_similarity="correlation",
        aggregation_type="adaptive",
        num_scales=3,
        num_fusions=6,
        num_stage_blocks=1,
        num_deform_blocks=3,
        no_intermediate_supervision=True,
        mdconv_dilation=2,
        deformable_groups=2,
        **kw,
    )
    load_strict(net, ctx.state_dict(), f"{ctx.variant.ref} checkpoint", unused=INTERMEDIATE)
    return net.eval().to(ctx.device)


def forward(net: AANet, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Upstream's `inference.py` forward: the last (full-resolution) disparity `[B, H, W]`."""
    disp = net(left, right)[-1]
    if disp.shape[-1] < left.shape[-1]:
        disp = F.interpolate(disp[:, None], left.shape[-2:], mode="bilinear")[:, 0] * (left.shape[-1] / disp.shape[-1])
    return disp


class Portable(Runner):
    """Plain PyTorch: float32 or fp16 autocast on CUDA (the deformable convolution is `grid_sample` + matmuls).

    fp16 AANet runs channels-last; AANet+ stays NCHW like upstream: its full-resolution hourglass refinement and GA-Net
    features ran 7-15 % slower channels-last. On MPS fp16 is the whole network in fp16, NCHW (`pure_half`): autocast
    there ran slower than fp32 (AANet 1163 ms against 940 ms at 994x1482, pure fp16 736 ms).
    """

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        self.half = ctx.dtype == "float16"
        self.pure = self.half and ctx.device.type == "mps"
        cl = self.half and not ctx.params.plus and not self.pure
        self.fmt = torch.channels_last if cl else torch.contiguous_format
        self.net.to(memory_format=self.fmt)
        if self.pure:
            pure_half(self.net, channels_last=False)

    def _run(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if not self.half:
            return forward(self.net, left, right)
        if self.pure:
            return forward(self.net, left, right).float()
        with torch.autocast(left.device.type, dtype=torch.float16):
            return forward(self.net, left.contiguous(memory_format=self.fmt), right.contiguous(memory_format=self.fmt))

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded batch."""
        return {"disparity": self._run(left, right)}

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        return Forward(self.net, lambda _net, left, right: self._run(left, right))


_GRAPH_PIXELS = 500_000  # fixed: batch x padded H x W; launch-bound up to KITTI 375x1242, measured
_GRAPHS = 2  # fixed: graph cache policy


class Kernels(Runner):
    """Our kernels: fp16-accumulate convolutions, conv + BatchNorm + activation fused into one epilogue, fp16.

    A repeated input shape up to `_GRAPH_PIXELS` replays a CUDA graph (`fastforks.shared.cuda_graphs`): the eager
    pass is launch-bound there.
    """

    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network and the kernel state."""
        super().__init__(ctx)
        from .kernels.inference import forward_kernels, prepare_kernels  # noqa: PLC0415 - needs the kernel libraries

        self.net = _net(ctx)
        prepare_kernels(self.net, ctx.device)
        self._forward = forward_kernels
        self._graphs = SmallShapeGraphs(
            self._run,
            pixels=_GRAPH_PIXELS,
            graphs=_GRAPHS,
            enabled=ctx.params.cuda_graphs,
            capture_after=cudnn_hacc.capture_after(),
        )

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        self._graphs.release()

    def _run(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self._forward(self.net, left, right)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        return {"disparity": self._graphs(left, right)}


class MetalKernels(Portable):
    """Apple MPS: the portable network with the deformable convolutions' sampling as one Metal kernel.

    float16 weights and NCHW, no autocast (its casts took a quarter of the time).
    """

    module = Runner.module  # the Metal kernel is not exportable

    def __init__(self, ctx: Context) -> None:
        """Build the network and swap in the kernel."""
        super().__init__(ctx)
        from fastforks._core import mps  # noqa: PLC0415 - MPS only
        from fastforks.kernels import stereo_metal  # noqa: PLC0415 - needs the Metal library
        from fastforks.kernels.stereo_metal.modules import DeformConv  # noqa: PLC0415

        mps.adapt(self.net)
        self.net.half().to(memory_format=torch.contiguous_format)
        dc = DeformConv(stereo_metal.load(ctx.device))
        for m in self.net.modules():
            if "modulated_deform_conv" in vars(m):
                object.__setattr__(m, "modulated_deform_conv", dc)  # a plain attribute

    def _run(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return forward(self.net, left.half(), right.half()).float()  # a float32 disparity


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "kernels_mps": MetalKernels,
    "portable_mlx": mlx_wrap.port(Portable, whole="disparity"),
}
