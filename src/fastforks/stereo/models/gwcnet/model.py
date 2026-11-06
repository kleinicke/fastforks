"""GwcNet runners: weights -> network and the portable / kernels forward passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from fastforks.stereo._runner import Forward, Runner

from .net.gwcnet import GwcNet
from .net.submodule import groupwise_correlation
from .regress import upsample_softmax_regress

if TYPE_CHECKING:
    from fastforks.stereo._runner import Context


@dataclass(frozen=True)
class Params:
    """Architecture of a variant (card `[params]` + variant `params`)."""

    concat_volume: bool
    max_disp: int


def _net(ctx: Context) -> GwcNet:
    net = GwcNet(ctx.params.max_disp, use_concat_volume=ctx.params.concat_volume)
    net.load_state_dict(ctx.state_dict(key="model"))
    return net.eval().to(ctx.device)


def cost_volume(net: GwcNet, fl: dict[str, torch.Tensor], fr: dict[str, torch.Tensor]) -> torch.Tensor:
    """Upstream's group-wise correlation (+ concatenation) volume `[B, G (+ 2C), D, H, W]`, built in place.

    Same values as `torch.cat((build_gwc_volume(...), build_concat_volume(...)), 1)`, without the two separate volumes
    alive next to the concatenated one.
    """
    # NCHW features: the group mean over a channels-last product is a strided reduction, 8x slower than contiguous
    gl, gr = fl["gwc_feature"].contiguous(), fr["gwc_feature"].contiguous()
    cl, cr = (fl["concat_feature"], fr["concat_feature"]) if net.use_concat_volume else (gl[:, :0], gr[:, :0])
    b, _, h, w = gl.shape
    g, c = net.num_groups, cl.shape[1]
    volume = gl.new_zeros([b, g + 2 * c, net.maxdisp // 4, h, w])
    for i in range(min(net.maxdisp // 4, w)):  # disparities >= width: no overlap, stays zero
        v = volume[:, :, i, :, i:]
        v[:, :g] = groupwise_correlation(gl[..., i:], gr[..., : w - i], g)
        if c:
            v[:, g : g + c] = cl[..., i:]
            v[:, g + c :] = cr[..., : w - i]
    return volume


def forward(net: GwcNet, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Upstream's test forward with the row-chunked regression tail; disparity `[B, H, W]` float32.

    Inputs are cast to the network's dtype (channels-last on CUDA); the regression runs in float32.
    """
    p = next(net.parameters())
    fmt = torch.channels_last if p.is_cuda else torch.contiguous_format
    left, right = (x.to(p.dtype).contiguous(memory_format=fmt) for x in (left, right))
    fl, fr = net.feature_extraction(left), net.feature_extraction(right)
    volume = cost_volume(net, fl, fr)
    del fl, fr
    cost0 = net.dres0(volume)
    del volume
    cost0 = net.dres1(cost0) + cost0
    out3 = net.dres4(net.dres3(net.dres2(cost0)))
    del cost0
    return upsample_softmax_regress(net.classif3(out3), net.maxdisp, left.shape[2], left.shape[3])


class Portable(Runner):
    """Plain PyTorch: float32 (CPU) or pure fp16 weights, NHWC (CUDA), regression in float32."""

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
        prepare_generic(self.net, ctx.device)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        return {"disparity": forward(self.net, left, right)}


RUNNERS = {"portable": Portable, "kernels": Kernels}
