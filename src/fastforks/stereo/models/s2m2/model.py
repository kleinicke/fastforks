"""S2M2 runners: weights -> network, the upstream blurred-border padding, and the portable / kernels forward passes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks.stereo._runner import Crop, Forward, Runner

from .net import rewrites
from .net.s2m2 import S2M2

if TYPE_CHECKING:
    from fastforks.stereo._runner import Context


@dataclass(frozen=True)
class Params:
    """Architecture of a variant (card `[params]` + variant `params`)."""

    feature_channels: int
    num_transformer: int
    dim_expansion: int
    use_positivity: bool
    refine_iter: int


_UNUSED = ("feature_flow_attn.q_proj.", "feature_flow_attn.k_proj.", "upsample_mask_4x.", "disp_init.layer_norm1.")


def _net(ctx: Context) -> S2M2:
    p: Params = ctx.params
    net = S2M2(
        feature_channels=p.feature_channels,
        dim_expansion=p.dim_expansion,
        num_transformer=p.num_transformer,
        use_positivity=p.use_positivity,
        refine_iter=p.refine_iter,
    )
    # Strict, unlike upstream's `my_load_state_dict` (which keeps the random init of a tensor whose shape differs and
    # ignores unknown keys), except for the tensors every released checkpoint carries but the network does not use.
    load_strict(net, ctx.state_dict(key="state_dict"), f"{ctx.variant.ref} checkpoint", unused=_UNUSED)
    return net.eval().to(ctx.device)


def _maps(disp: torch.Tensor, occ: torch.Tensor, conf: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"disparity": disp[:, 0], "confidence": conf[:, 0], "occlusion": occ[:, 0]}


class _S2M2Runner(Runner):
    Params = Params

    def pad(self, x: torch.Tensor) -> tuple[torch.Tensor, Crop]:
        """Upstream padding: centred, with a blurred copy of the image in the border."""
        h, w = x.shape[-2:]
        padded = image_pad(x, self.ctx.io.pad_multiple)
        return padded, Crop((padded.shape[-2] - h) // 2, (padded.shape[-1] - w) // 2, h, w)


class Portable(_S2M2Runner):
    """Plain PyTorch (upstream network with our rewrites), autocast to the card's CUDA dtype."""

    dtypes = ("float32", "float16", "bfloat16")

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        # CUDA: no 1x1-conv-before-upsample commute (3 % slower, 4 % larger there)
        rewrites.prepare(self.net, upconv=ctx.device.type == "cpu")
        if ctx.device.type == "cpu":  # oneDNN runs channels_last convolutions faster (S2M2 has no 3-D convs)
            self.net.to(memory_format=torch.channels_last)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity, confidence and occlusion of a padded batch."""
        if self.ctx.dtype == "float32":
            return _maps(*self.net(left, right))
        with torch.autocast(self.ctx.device.type, dtype=self.ctx.torch_dtype):
            return _maps(*self.net(left, right))

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        return Forward(self.net, lambda net, left, right: net(left, right)[0][:, 0])


class Kernels(_S2M2Runner):
    """Our kernels: fp16 with fp16-accumulate convolutions and fused attention / matching kernels."""

    def __init__(self, ctx: Context) -> None:
        """Build the network and bind the fast forward."""
        super().__init__(ctx)
        from .kernels.inference import prepare_kernels  # noqa: PLC0415 - needs the kernel libraries

        self.net = _net(ctx)
        prepare_kernels(self.net, hacc=True, kernels=True)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity, confidence and occlusion of a padded pair."""
        return _maps(*self.net.forward_fast(left, right, full_outputs=True))  # ty: ignore[call-non-callable] - bound by prepare_kernels


RUNNERS = {"portable": Portable, "kernels": Kernels}


def image_pad(img: torch.Tensor, factor: int) -> torch.Tensor:
    """Centre-pad `[B, C, H, W]` to multiples of `factor`, filling the border with a blurred image (upstream code)."""
    h, w = img.shape[-2:]
    hn, wn = math.ceil(h / factor) * factor, math.ceil(w / factor) * factor
    top, left = (hn - h) // 2, (wn - w) // 2
    padded = F.pad(img, (left, wn - w - left, top, hn - h - top))
    size = (max(h // factor, 1), max(w // factor, 1))  # >= 1: small images
    # MPS implements adaptive pooling only for divisible sizes; the pooled map is tiny, so it is computed on the CPU
    src = padded.float().cpu() if padded.device.type == "mps" else padded.float()
    coarse = F.adaptive_avg_pool2d(src, size).to(padded.device)
    fill = F.interpolate(coarse, size=(hn, wn), mode="bilinear")
    fill[:, :, top : top + h, left : left + w] = img
    return fill
