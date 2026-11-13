"""LoMa runners: the single checkpoint -> DaD detector, DeDoDe-G descriptor and the matcher, and the portable forward.

As upstream's `LoMa.match` on image paths (its MegaDepth / ScanNet benchmarks): the DaD detector sees the image
resized to a long side of `long_side` (aspect kept, both sides floored to a multiple of 8; bicubic, antialiased,
clamped to [0, 1]: PIL's `resize` of upstream's `load_image` without its rounding to uint8), ImageNet-normalised; its
logits give a softmax over all pixels, non-maximum-suppressed in 3 x 3 windows, and the `max_keypoints` best pixels,
refined to sub-pixel by the softmax of their 3 x 3 logits, are the keypoints. The DeDoDe-G descriptor (VGG19 + a
frozen DINOv2 ViT-L/14) sees the image resized to a `size` x `size` square, not normalised (upstream's
`read_image`), and samples a 256-d description at each keypoint (bilinear). Nine LightGlue-style transformer layers
(rotary self-attention, cross-attention) refine the descriptions of both images, and the matches are the mutual
maxima of the row-times-column softmax of their projected similarity above `threshold`, scored by that product.
Keypoints are mapped back to input pixels (pixel centres at integers: upstream's `to_pixel_coords` minus 0.5).

On CUDA the validated dtype is bfloat16 as upstream on Ampere and newer (float16 where upstream falls back to it):
the VGG encoders and the refiners under autocast, the DINOv2 with weights and input in that dtype, the matcher under
autocast; logits, keypoint sampling, the description sampling and the match filter stay float32. On CUDA the eval
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
from fastforks.shared.dedode import (
    DETECTOR_S,
    VGG,
    VGG11_BN,
    ConvRefiner,
    Descriptor,
    Detector,
    VGGDINOv2,
    describe_keypoints,
)
from fastforks.shared.depth_anything_v2 import DINOv2

from .net.dad import sample_keypoints
from .net.loma import LoMaNet, filter_matches

if TYPE_CHECKING:
    from fastforks.matching._runner import Context

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # fixed: DaD's `normalizer`
NMS_SIZE = 3  # fixed: upstream `DaD.Cfg.nms_size`
SUBPIXEL_TEMP = 0.5  # fixed: upstream `DaD.Cfg.subpixel_temp`
STRIDE = 8  # fixed: upstream `load_image` floors both sides to a multiple of 8
PREFIXES = {  # checkpoint prefix -> (submodule, the prefix it keeps)
    "_detector.": "detector",
    "_descriptor.encoder.vgg.": "descriptor.encoder.vgg",
    "_descriptor.encoder.frozen_dinov2.dinov2_vitl14.": "dinov2",
    "_descriptor.decoder.": "descriptor.decoder",
}


@dataclass(frozen=True)
class Params:
    """Variant parameters and load options."""

    dim: int  # description and embedding width (loma-b: 256)
    heads: int  # attention heads (loma-b: 4)
    layers: int  # transformer layers (loma-b: 9)
    threshold: float  # match score (product of the two softmaxes) a match needs (upstream 0.1)
    long_side: int  # the detector's image: long side (upstream 1024), aspect kept
    size: int  # the descriptor's image: side of the square (upstream 784)
    max_keypoints: int  # keypoints per image (upstream default 2048; its benchmarks 4096)


Described = tuple[torch.Tensor, torch.Tensor]  # normalised keypoints [N, 2], descriptions [N, dim]


def fold_batch_norms(net: nn.Module) -> None:
    """Eval BatchNorms into the conv before them (VGG blocks, refiner blocks)."""
    for seq in [m for m in net.modules() if isinstance(m, (nn.Sequential, nn.ModuleList))]:
        layers = list(seq)
        for i in range(len(layers) - 1):
            conv, bn = layers[i], layers[i + 1]
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                seq[i], seq[i + 1] = fuse_conv_bn_eval(conv, bn), nn.Identity()


class Matcher(nn.Module):
    """LoMa on an RGB pair: `[1, 3, H1, W1]`, `[1, 3, H2, W2]` in `[0, 1]` -> keypoints, index pairs, scores."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, params: Params) -> None:
        """Build the networks (no weights)."""
        super().__init__()
        self.params = params
        self.detector = Detector(DETECTOR_S, VGG11_BN)
        self.dinov2 = DINOv2("vitl")
        self.descriptor = Descriptor(self.dinov2)
        self.matcher = LoMaNet(params.dim, params.dim, params.heads, params.layers)
        self.amp: torch.dtype | None = None
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

    def set_amp(self, amp: torch.dtype | None) -> None:
        """Upstream's autocast dtype for the encoders, refiners and the matcher; `None` = float32."""
        self.amp = amp
        for m in self.modules():
            if isinstance(m, (VGG, ConvRefiner)):
                m.amp = amp
        encoder = self.descriptor.encoder
        if isinstance(encoder, VGGDINOv2):
            encoder.dinov2[0].to(amp or torch.float32)
            encoder.dino_dtype = amp or torch.float32

    @staticmethod
    def _resize(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(image, size=size, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)

    def describe(self, image: torch.Tensor) -> Described:
        """Keypoints (normalised coordinates) and their descriptions of one image."""
        p = self.params
        h, w = image.shape[-2:]
        scale = p.long_side / max(h, w)
        det = self._resize(image, (int(scale * h) // STRIDE * STRIDE, int(scale * w) // STRIDE * STRIDE))
        logits = self.detector((det - self.mean) / self.std)
        keypoints = sample_keypoints(logits, p.max_keypoints, NMS_SIZE, SUBPIXEL_TEMP)
        grid = self.descriptor(self._resize(image, (p.size, p.size)))
        return keypoints, describe_keypoints(grid, keypoints)

    @staticmethod
    def to_input(keypoints: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Normalised keypoints in input pixels (pixel centres at integers)."""
        h, w = image.shape[-2:]
        return (keypoints + 1) / 2 * torch.tensor([w, h], device=keypoints.device) - 0.5

    def match(
        self, image1: torch.Tensor, image2: torch.Tensor, d1: Described, d2: Described
    ) -> dict[str, torch.Tensor]:
        """The pair's matches from both images' `describe` results."""
        device = image1.device.type
        with torch.autocast(device, enabled=self.amp is not None, dtype=self.amp or torch.float16):
            scores = self.matcher(d1[0][None], d2[0][None], d1[1][None], d2[1][None])
        m0, mscores0 = filter_matches(scores.float(), self.params.threshold)
        idx = torch.nonzero(m0 > -1)[:, 0]
        return {
            "keypoints1": self.to_input(d1[0], image1),
            "keypoints2": self.to_input(d2[0], image2),
            "pairs": torch.stack([idx, m0[idx]], dim=-1),
            "scores": mscores0[idx],
        }

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """The pair's features and matches (batch 1)."""
        return self.match(image1, image2, self.describe(image1), self.describe(image2))


def split_checkpoint(sd: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Upstream's single state dict by submodule (`PREFIXES` values and `"matcher"`) -> that module's state dict."""
    parts: dict[str, dict[str, torch.Tensor]] = {name: {} for name in [*PREFIXES.values(), "matcher"]}
    for k, v in sd.items():
        for prefix, name in PREFIXES.items():
            if k.startswith(prefix):
                parts[name][k[len(prefix) :]] = v
                break
        else:
            parts["matcher"][k] = v
    return parts


def load_matcher(ctx: Context) -> Matcher:
    """The matcher with the variant's weights (one file: detector, descriptor, DINOv2, matcher) on `ctx.device`."""
    p = ctx.params
    net = Matcher(p)
    parts = split_checkpoint(ctx.state_dict(0))
    for name, sd in parts.items():
        net.get_submodule(name).load_state_dict(sd, strict=True)
    net.eval().requires_grad_(requires_grad=False).to(ctx.device)
    if ctx.device.type == "cuda":  # not on CPU: the float32 reference parity
        fold_batch_norms(net)
        for m in net.modules():
            if isinstance(m, ConvRefiner):
                m.to(memory_format=torch.channels_last)
        net.detector.encoder.to(memory_format=torch.channels_last)
        net.descriptor.encoder.to(memory_format=torch.channels_last)
    net.set_amp(None if ctx.dtype == "float32" else ctx.torch_dtype)
    return net


class Portable(Runner):
    """Plain PyTorch; see the module docstring for the autocast regions on CUDA."""

    dtypes = ("float32", "float16", "bfloat16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the networks."""
        super().__init__(ctx)
        self.net = load_matcher(ctx)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Keypoints of both images, the index pairs of their matches and the match scores."""
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
