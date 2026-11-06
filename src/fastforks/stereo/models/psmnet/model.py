"""PSMNet runners: weights -> network and the portable / kernels forward passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from fastforks.stereo._runner import Forward, Runner

from .net.stackhourglass import PSMNet
from .regress import upsample_softmax_regress

if TYPE_CHECKING:
    from fastforks.stereo._runner import Context


@dataclass(frozen=True)
class Params:
    """Architecture of a variant (card `[params]` + variant `params`)."""

    max_disp: int


def _net(ctx: Context) -> PSMNet:
    net = PSMNet(ctx.params.max_disp)
    net.load_state_dict(ctx.state_dict(key="state_dict"))
    return net.eval().to(ctx.device)


def cost_volume(net: PSMNet, ref: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Upstream's concatenation volume `[B, 2C, D, H, W]` in the features' dtype (upstream: float32)."""
    b, c, h, w = ref.shape
    volume = ref.new_zeros((b, 2 * c, net.maxdisp // 4, h, w))
    for i in range(min(net.maxdisp // 4, w)):  # disparities >= width: no overlap, stays zero
        volume[:, :c, i, :, i:] = ref[..., i:]
        volume[:, c:, i, :, i:] = tgt[..., : w - i]
    return volume


def forward(net: PSMNet, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Upstream's test forward (`pred3`) with the row-chunked regression tail; disparity `[B, H, W]` float32.

    Inputs are cast to the network's dtype (channels-last on CUDA); the regression runs in float32.
    """
    p = next(net.parameters())
    fmt = torch.channels_last if p.is_cuda else torch.contiguous_format
    left, right = (x.to(p.dtype).contiguous(memory_format=fmt) for x in (left, right))
    volume = cost_volume(net, net.feature_extraction(left), net.feature_extraction(right))
    cost0 = net.dres0(volume)
    del volume
    cost0 = net.dres1(cost0) + cost0
    out1, pre1, post1 = net.dres2(cost0, None, None)
    out1 = out1 + cost0
    out2, _, post2 = net.dres3(out1, pre1, post1)
    out2 = out2 + cost0
    out3, _, _ = net.dres4(out2, pre1, post2)
    out3 = out3 + cost0
    del cost0
    cost3 = net.classif3(out3) + (net.classif2(out2) + net.classif1(out1))  # upstream's summation order
    return upsample_softmax_regress(cost3, net.maxdisp, left.shape[2], left.shape[3])


class Portable(Runner):
    """Plain PyTorch: float32 (CPU) or pure fp16 weights, NHWC features (CUDA), regression in float32."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx).to(ctx.torch_dtype)
        if ctx.device.type == "cuda":  # NHWC for the 2D feature network (the 3D cost-volume convs stay NCDHW)
            self.net.feature_extraction.to(memory_format=torch.channels_last)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded batch."""
        return {"disparity": forward(self.net, left, right)}

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        return Forward(self.net, forward)


class Kernels(Runner):
    """Our kernels: fp16-accumulate convolutions, conv + BatchNorm + activation fused into one epilogue, fp16."""

    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network and the kernel state."""
        super().__init__(ctx)
        from fastforks.shared.cnn_kernels import prepare_generic  # noqa: PLC0415 - needs the kernel libraries

        self.net = _net(ctx)
        if not prepare_generic(self.net, ctx.device):
            msg = "kernels: no conv + BatchNorm run was fused (the module tree changed)"
            raise RuntimeError(msg)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        return {"disparity": forward(self.net, left, right)}


RUNNERS = {"portable": Portable, "kernels": Kernels}
