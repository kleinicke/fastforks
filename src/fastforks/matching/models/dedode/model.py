"""DeDoDe v2 runners: weights -> detector + descriptor, and the portable forward pass.

As upstream's README / demos (`detect_from_path`, `describe_keypoints_from_path`, `DualSoftMaxMatcher.match`): each
image is resized to a `size` x `size` square (bicubic, antialiased, clamped to [0, 1]: PIL's `resize` of upstream's
`read_image`, without its rounding to uint8) and ImageNet-normalised; the DeDoDe-L v2 detector's logits give a softmax
over all pixels, reweighted by the inverse square root of their local density (51-tap Gaussian, upstream's
`increase_coverage`), and the `max_keypoints` best pixels are the keypoints. The descriptor (DeDoDe-B, or -G with a
frozen DINOv2 ViT-L/14) samples a 256-d description at each (bilinear); the matches are the mutual maxima of the dual
softmax of the normalised descriptions (inverse temperature 20) above `threshold`, scored by that probability.
Keypoints are mapped back to input pixels (pixel centres at integers: upstream's `to_pixel_coords` minus 0.5).

On CUDA the validated dtype is float16 as upstream: the VGG encoders and the refiners under float16 autocast, the
DINOv2 of -G with float16 weights and input; logits, keypoint sampling and matching stay float32. On CUDA the eval
BatchNorms are folded into their convs and the convolutions run channels-last (cuDNN's depthwise 5 x 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from fastforks.matching._runner import Forward, Runner
from fastforks.shared.depth_anything_v2 import DINOv2

from .net.dedode import (
    VGG,
    ConvRefiner,
    Descriptor,
    Detector,
    VGGDINOv2,
    describe_keypoints,
    dual_softmax_match,
    sample_keypoints,
)

if TYPE_CHECKING:
    from fastforks.matching._runner import Context

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # fixed: upstream's `normalizer`
INV_TEMP = 20.0  # fixed: upstream README / demos


@dataclass(frozen=True)
class Params:
    """Variant parameters and load options."""

    descriptor: str  # "B" (VGG19) or "G" (VGG19 + frozen DINOv2 ViT-L/14)
    threshold: float  # dual-softmax probability a match needs (upstream demos 0.01)
    size: int  # side of the square the images are resized to (upstream 784)
    max_keypoints: int  # keypoints per image (upstream demos 10000)


Described = tuple[torch.Tensor, torch.Tensor]  # normalised keypoints [N, 2], descriptions [N, 256]


def fold_batch_norms(net: nn.Module) -> None:
    """Eval BatchNorms into the conv before them (VGG blocks, refiner blocks)."""
    for seq in [m for m in net.modules() if isinstance(m, (nn.Sequential, nn.ModuleList))]:
        layers = list(seq)
        for i in range(len(layers) - 1):
            conv, bn = layers[i], layers[i + 1]
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                seq[i], seq[i + 1] = fuse_conv_bn_eval(conv, bn), nn.Identity()


class Matcher(nn.Module):
    """DeDoDe on an RGB pair: `[1, 3, H1, W1]`, `[1, 3, H2, W2]` in `[0, 1]` -> keypoints, index pairs, scores."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, params: Params) -> None:
        """Build the networks (no weights)."""
        super().__init__()
        self.params = params
        self.detector = Detector()
        self.dinov2 = DINOv2("vitl") if params.descriptor == "G" else None
        self.descriptor = Descriptor(self.dinov2)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

    def set_amp(self, amp: torch.dtype | None) -> None:
        """Upstream's autocast dtype (float16) for the encoders and refiners; `None` = float32."""
        for m in self.modules():
            if isinstance(m, (VGG, ConvRefiner)):
                m.amp = amp
        encoder = self.descriptor.encoder
        if isinstance(encoder, VGGDINOv2) and amp is not None:
            encoder.dinov2[0].to(amp)
            encoder.dino_dtype = amp

    def prepare(self, image: torch.Tensor) -> torch.Tensor:
        """The normalised `size` x `size` image."""
        s = self.params.size
        x = F.interpolate(image, size=(s, s), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        return (x - self.mean) / self.std

    def describe(self, image: torch.Tensor) -> Described:
        """Keypoints (normalised coordinates) and their descriptions of one image."""
        x = self.prepare(image)
        keypoints, _ = sample_keypoints(self.detector(x), self.params.max_keypoints)
        return keypoints, describe_keypoints(self.descriptor(x), keypoints)

    @staticmethod
    def to_input(keypoints: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Normalised keypoints in input pixels (pixel centres at integers)."""
        h, w = image.shape[-2:]
        return (keypoints + 1) / 2 * torch.tensor([w, h], device=keypoints.device) - 0.5

    def match(
        self, image1: torch.Tensor, image2: torch.Tensor, d1: Described, d2: Described
    ) -> dict[str, torch.Tensor]:
        """The pair's matches from both images' `describe` results."""
        pairs, scores = dual_softmax_match(d1[1], d2[1], INV_TEMP, self.params.threshold)
        return {
            "keypoints1": self.to_input(d1[0], image1),
            "keypoints2": self.to_input(d2[0], image2),
            "pairs": pairs,
            "scores": scores,
        }

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """The pair's features and matches (batch 1)."""
        return self.match(image1, image2, self.describe(image1), self.describe(image2))


def load_matcher(ctx: Context) -> Matcher:
    """The matcher with the variant's weights (detector, descriptor, DINOv2 for -G), eval, on `ctx.device`."""
    p = ctx.params
    net = Matcher(p)
    net.detector.load_state_dict(ctx.state_dict(0), strict=True)
    net.descriptor.load_state_dict(ctx.state_dict(1), strict=True)
    if net.dinov2 is not None:
        net.dinov2.load_state_dict(ctx.state_dict(2), strict=True)
    net.eval().requires_grad_(requires_grad=False).to(ctx.device)
    if ctx.device.type == "cuda":  # not on CPU: the float32 reference parity
        fold_batch_norms(net)
        for m in net.modules():
            if isinstance(m, ConvRefiner):
                m.to(memory_format=torch.channels_last)
        net.detector.encoder.to(memory_format=torch.channels_last)
        net.descriptor.encoder.to(memory_format=torch.channels_last)
    net.set_amp(torch.float16 if ctx.dtype == "float16" else None)
    return net


class Portable(Runner):
    """Plain PyTorch; see the module docstring for the float16 regions on CUDA."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the networks."""
        super().__init__(ctx)
        self.net = load_matcher(ctx)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Keypoints of both images, the index pairs of their matches and the match probabilities."""
        return self.net(image1, image2)

    def describe(self, image: torch.Tensor) -> Described:
        """Keypoints and descriptions of one image: the per-image stage."""
        return self.net.describe(image)

    def match(self, image1: torch.Tensor, image2: torch.Tensor, described: tuple[Described, Described]) -> dict:
        """`__call__` with the two images' `describe` results."""
        return self.net.match(image1, image2, *described)

    def module(self) -> torch.nn.Module:
        """Detector, descriptor and matcher (data-dependent match counts: compiles with graph breaks, no export)."""
        return Forward(self.net, lambda net, image1, image2: net(image1, image2))


RUNNERS = {"portable": Portable}
