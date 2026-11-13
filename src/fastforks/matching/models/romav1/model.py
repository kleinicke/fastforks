"""RoMa v1 runners: weights -> network and the portable forward pass.

Upstream `RegressionMatcher.match` of `roma_outdoor` (one direction, the first image's half of upstream's symmetric
call): both images are resized (bicubic, antialiased) to a `coarse_size` square and normalised with the ImageNet mean
/ std for the DINOv2 + VGG19-BN encoder, the Gaussian-process match, the transformer classifier and the refiners at
strides 16 to 1, then to a `fine_size` square for the second refinement pass at strides 8 to 1 (`fine_size = 0`:
none). The certainty is upstream's: sigmoid of the refined logit minus half the negative part of the coarse logit
(`attenuate_cert`), 0 where the warp leaves the second image. Warp and certainty are resized (bilinear) to the first
image's input size; the warp goes from normalised coordinates to pixels of the second image (pixel centres at
integers). Upstream resizes PIL images with torchvision's bicubic `Resize`; this runner resizes the float tensors.

On CUDA the validated dtype is float16 as upstream (`amp_dtype=torch.float16`): DINOv2 in fp16 weights, the VGG, the
projections, the transformer classifier and the refiner blocks under fp16 autocast (upstream's
`get_autocast_params` enables autocast on CUDA for every region); the Gaussian process, the refiners' output convs
and the warps stay float32, the classifier's soft-argmax runs in fp16 outside autocast (as upstream). float32 is
upstream with every autocast off. On CUDA the eval BatchNorms are folded into their convs, the VGG and refiner
activations are channels-last and the conv / linear weights of the fp16 regions are stored in fp16.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from fastforks._core.checkpoint import check_fits
from fastforks.matching._runner import Forward, Runner
from fastforks.shared.depth_anything_v2 import sdpa_attention, use_fast_vit, vit_large
from fastforks.shared.depth_anything_v2.net.dinov2 import DinoVisionTransformer

from .net.roma import ConvRefiner, RoMa, TransformerDecoder, imagenet

if TYPE_CHECKING:
    from fastforks.matching._runner import Context

PATCH = 14  # fixed: DINOv2's patch size (the coarse size)
FINE_MULTIPLE = 8  # fixed: the refiners' largest stride in the upsampling pass


@dataclass(frozen=True)
class Params:
    """Load options."""

    coarse_size: int  # side of the square of the coarse pass (upstream coarse_res); a multiple of 14
    fine_size: int  # side of the square of the upsampling pass (upstream upsample_res); 0 = none; a multiple of 8


def resize(img: torch.Tensor, size: int) -> torch.Tensor:
    """Bicubic, antialiased resize to a `size` square, then the ImageNet normalisation (upstream's test transform)."""
    out = F.interpolate(img, (size, size), mode="bicubic", align_corners=False, antialias=True)
    return imagenet(out)


def forward(net: RoMa, image1: torch.Tensor, image2: torch.Tensor, params: Params) -> dict[str, torch.Tensor]:
    """`{"warp": [B, 2, H1, W1], "certainty": [B, H1, W1]}` of `[0, 1]` RGB `[B, 3, H1, W1]`, `[B, 3, H2, W2]`."""
    a, b = resize(image1, params.coarse_size), resize(image2, params.coarse_size)
    fine = params.fine_size > 0
    a_hi = resize(image1, params.fine_size) if fine else None
    b_hi = resize(image2, params.fine_size) if fine else None
    warp, certainty = net(a, b, a_hi, b_hi)
    warp, certainty = warp.float(), certainty.float()
    (h1, w1), (h2, w2) = image1.shape[-2:], image2.shape[-2:]
    if warp.shape[-2:] != (h1, w1):
        warp = F.interpolate(warp, (h1, w1), mode="bilinear", align_corners=False)
        certainty = F.interpolate(certainty[:, None], (h1, w1), mode="bilinear", align_corners=False)[:, 0]
    size = torch.tensor((w2, h2), dtype=warp.dtype, device=warp.device).view(1, 2, 1, 1)
    return {"warp": (warp + 1) * (size / 2) - 0.5, "certainty": certainty}


def dinov2() -> nn.Module:
    """Upstream's DINOv2 ViT-L/14 (img 518, LayerScale, MLP FFN) with SDPA attention, no weights yet."""
    with mock.patch.object(DinoVisionTransformer, "init_weights", return_value=None):  # every tensor is assigned
        vit = vit_large(img_size=518, patch_size=PATCH, init_values=1.0, ffn_layer="mlp", block_chunks=0)
    sdpa_attention(vit)  # upstream's dinov2 attention is `scaled_dot_product_attention`
    return vit


def _fold(seq: nn.Sequential | nn.ModuleList) -> None:
    for i, (conv, bn) in enumerate(itertools.pairwise(seq)):
        if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
            seq[i], seq[i + 1] = fuse_conv_bn_eval(conv, bn), nn.Identity()


def fold_batch_norms(net: RoMa) -> None:
    """Eval BatchNorms into the preceding conv: the VGG layers, the projections and the refiner blocks."""
    _fold(net.encoder.cnn.layers)
    for m in net.modules():
        if isinstance(m, nn.Sequential):
            _fold(m)


def cast_autocast_weights(net: RoMa, dtype: torch.dtype) -> None:
    """Conv / linear weights inside the autocast regions in `dtype` (autocast casts them on every call anyway).

    The VGG, the projections, the transformer classifier and the refiners up to their float32 output convs; norms
    stay float32.
    """
    regions: list[nn.Module] = [net.encoder.cnn, *net.decoder.proj.values()]
    regions += [m for m in net.modules() if isinstance(m, TransformerDecoder)]
    skip: set[nn.Module] = set()
    for m in net.modules():
        if isinstance(m, ConvRefiner):
            regions.append(m)
            skip.add(m.out_conv)  # after upstream's `d.float()`
    for region in regions:
        for m in region.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)) and m not in skip:
                m.to(dtype)


class Portable(Runner):
    """Plain PyTorch; see the module docstring for the fp16 regions on CUDA."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        p = ctx.params
        if p.coarse_size % PATCH or p.fine_size % FINE_MULTIPLE:
            msg = (
                f"{ctx.variant.ref}: coarse_size must be a multiple of {PATCH} and fine_size of {FINE_MULTIPLE}, "
                f"not {p.coarse_size}, {p.fine_size}"
            )
            raise ValueError(msg)
        net = RoMa(dinov2())
        sd = ctx.state_dict(0) | {f"encoder.dinov2.{k}": v for k, v in ctx.state_dict(1).items()}
        check_fits(net.state_dict(), sd, f"{ctx.variant.ref} checkpoints")
        net.load_state_dict(sd)
        net.eval().requires_grad_(requires_grad=False).to(ctx.device)
        if ctx.dtype == "float16":
            amp = torch.float16
            net.encoder.dinov2.to(amp)  # upstream: the ViT in fp16 weights, no autocast
            net.encoder.dino_dtype = amp
            for m in net.modules():
                if hasattr(m, "amp"):
                    setattr(m, "amp", amp)  # noqa: B010 - the autocast regions (VGG, decoder, refiners)
        if ctx.device.type == "cuda":  # not on CPU: the float32 reference parity
            fold_batch_norms(net)
            net.encoder.cnn.to(memory_format=torch.channels_last)
            for m in net.modules():
                if isinstance(m, ConvRefiner):
                    m.to(memory_format=torch.channels_last)
            if ctx.dtype == "float16":
                cast_autocast_weights(net, torch.float16)
        self.net = net

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Warp and certainty of a batch."""
        return forward(self.net, image1, image2, self.ctx.params)

    def module(self) -> torch.nn.Module:
        """The network with the resizing; returns `{"warp", "certainty"}`."""
        params = self.ctx.params
        return Forward(self.net, lambda net, image1, image2: forward(net, image1, image2, params))


class Kernels(Portable):
    """The portable pass with DINOv2 on the fused fp16 ViT path (`FastViT`: flash attention, fused block GEMMs)."""

    def __init__(self, ctx: Context) -> None:
        """Build the network and route the DINOv2 pass through `FastViT`."""
        super().__init__(ctx)
        vit = cast("DinoVisionTransformer", self.net.encoder.dinov2)
        use_fast_vit(vit, attn="hh", cache_pos=False)  # upstream: the fp16 table

    def module(self) -> torch.nn.Module:
        """No exportable module: DINOv2 runs kernel launches."""
        return Runner.module(self)


RUNNERS = {"portable": Portable, "kernels": Kernels}
