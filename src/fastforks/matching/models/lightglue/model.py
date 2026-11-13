"""LightGlue runners: weights -> extractor + matcher, and the portable forward pass.

Upstream `match_pair` with `extractor.extract`: each image is resized so its longest side is `long_side` (kornia
`resize`: Gaussian blur before downscaling, then bilinear), features are detected there (ALIKED: score threshold, at
most `max_keypoints` by score; DISK: NMS 5, the best `max_keypoints` heatmap maxima), their keypoints are mapped back
to the input image (pixel centres at integers) and LightGlue matches them with upstream's defaults (adaptive depth and
width, match threshold 0.1). Returns both keypoint sets, the index pairs of the matches and their scores.

The LightGlue attention runs in float16 on CUDA, as upstream's (`flash=True`: SDPA on half q, k, v); the rest is
float32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from fastforks.matching._runner import Forward, Runner
from fastforks.shared.lightglue import LightGlue, match_many, resize_long

from .net.aliked import ALIKED
from .net.disk import DISK

if TYPE_CHECKING:
    from fastforks.matching._runner import Context

_DISK_NMS = 5  # fixed: upstream DISK extractor `nms_window_size`
_ALIKED_NMS = 2  # fixed: upstream ALIKED `nms_radius`


@dataclass(frozen=True)
class Params:
    """Variant parameters and load options."""

    extractor: str  # "aliked" (aliked-n16) or "disk" (depth)
    detection_threshold: float  # upstream extractor default (ALIKED 0.2, DISK 0.0)
    max_keypoints: int  # per image
    long_side: int  # longest side the features are detected at; 0 = the input size


class Matcher(nn.Module):
    """Extractor + LightGlue: `[1, 3, H1, W1]`, `[1, 3, H2, W2]` in `[0, 1]` -> keypoints, index pairs, scores."""

    def __init__(self, params: Params) -> None:
        """Build the networks (no weights)."""
        super().__init__()
        self.params = params
        p = params
        if p.extractor == "aliked":
            self.extractor = ALIKED(
                model_name="aliked-n16",
                max_num_keypoints=p.max_keypoints,
                detection_threshold=p.detection_threshold,
                nms_radius=_ALIKED_NMS,
            )
        else:
            self.extractor = DISK(desc_dim=128)
        self.matcher = LightGlue(features=p.extractor)

    def extract(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Upstream `Extractor.extract`: features at `long_side`, keypoints in input pixels."""
        p = self.params
        h, w = image.shape[-2:]
        x = resize_long(image, p.long_side) if p.long_side > 0 else image
        scale = torch.tensor([x.shape[-1] / w, x.shape[-2] / h], device=x.device, dtype=x.dtype)
        if p.extractor == "aliked":
            feats = self.extractor({"image": x})
            kpts, desc = feats["keypoints"], feats["descriptors"]
        else:
            ((kp, _, desc),) = self.extractor(
                x,
                n=p.max_keypoints,
                window_size=_DISK_NMS,
                score_threshold=p.detection_threshold,
                pad_if_not_divisible=True,
            )
            kpts, desc = kp[None].to(x), desc[None]
        return {
            "keypoints": (kpts + 0.5) / scale - 0.5,
            "descriptors": desc.contiguous(),
            "image_size": torch.tensor([[w, h]], device=x.device, dtype=x.dtype),
        }

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """The pair's features and matches (batch 1)."""
        return self.match(self.extract(image1), self.extract(image2))

    def match(self, f1: dict[str, torch.Tensor], f2: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """LightGlue on the two images' `extract` results."""
        out = self.matcher({"image0": f1, "image1": f2})
        return {
            "keypoints1": f1["keypoints"][0],
            "keypoints2": f2["keypoints"][0],
            "pairs": out["matches"][0],
            "scores": out["scores"][0],
        }


def load_matcher(ctx: Context) -> Matcher:
    """The matcher with the variant's weights (file 0: LightGlue, file 1: the extractor), eval, on `ctx.device`."""
    net = Matcher(ctx.params)
    net.matcher.load_weights(ctx.state_dict(0))
    sd = ctx.state_dict(1, key="extractor" if ctx.params.extractor == "disk" else None)
    net.extractor.load_state_dict(sd, strict=True)
    return net.eval().requires_grad_(requires_grad=False).to(ctx.device)


class Portable(Runner):
    """Plain PyTorch, float32 (LightGlue's attention in float16 on CUDA, as upstream)."""

    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the networks."""
        super().__init__(ctx)
        self.net = load_matcher(ctx)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Keypoints of both images, the index pairs of their matches and the match scores."""
        return self.net(image1, image2)

    match_reads_images = False
    matches_many = True

    def describe(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The image's keypoints and descriptors: the per-image stage."""
        return self.net.extract(image)

    def match(self, image1: torch.Tensor, image2: torch.Tensor, described: tuple[Any, Any]) -> dict[str, torch.Tensor]:
        """`__call__` with the two images' `describe` results (the images are not read again)."""
        del image1, image2
        return self.net.match(*described)

    def match_many(self, described: list[tuple[Any, Any]]) -> list[dict[str, torch.Tensor]]:
        """`match` of many pairs in one packed LightGlue forward (`fastforks.shared.lightglue.match_many`)."""
        outs = match_many(self.net.matcher, described)
        return [
            {
                "keypoints1": f1["keypoints"][0],
                "keypoints2": f2["keypoints"][0],
                "pairs": o["matches"],
                "scores": o["scores"],
            }
            for (f1, f2), o in zip(described, outs, strict=True)
        ]

    def module(self) -> torch.nn.Module:
        """The extractor and matcher (data-dependent keypoint counts: compiles with graph breaks, does not export)."""
        return Forward(self.net, lambda net, image1, image2: net(image1, image2))


RUNNERS = {"portable": Portable}
