"""LoFTR runners: weights -> the vendored network, and the portable forward pass.

As upstream's MegaDepth / ScanNet loaders and demo: the RGB image is converted to grayscale (0.299 R + 0.587 G +
0.114 B, OpenCV's weights), resized so its longest side is `long_side` (rounded; 0 = the input size) and both sides
cut down to multiples of 8 (bilinear, as `cv2.resize`), then upstream's LoFTR: coarse matches (dual-softmax confidence
above `threshold`, `border_rm` cells off the border, mutual maxima) refined in a 5 x 5 window at 1/2 resolution. The
matched points are scaled back to the input (`x * W / W_resized`, as upstream's loaders' `scale0` / `scale1`); every
match is its own keypoint pair, so `pairs` is `[[0, 0], [1, 1], ...]` and `scores` the coarse confidence. Float32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from fastforks.matching._runner import Forward, Runner

from .net.loftr import LoFTR

if TYPE_CHECKING:
    from fastforks.matching._runner import Context

GRAY = (0.299, 0.587, 0.114)  # fixed: OpenCV's RGB -> gray weights (upstream reads the images with cv2 as grayscale)
DIVISOR = 8  # fixed: the coarse stride (upstream `df`)


@dataclass(frozen=True)
class Params:
    """Variant parameters and load options."""

    temp_bug_fix: bool  # the position encoding the weights were trained with (upstream issue #41)
    border_rm: int  # coarse cells removed at the border (upstream eval configs: outdoor 2, indoor 0)
    threshold: float  # coarse dual-softmax confidence (upstream 0.2)
    long_side: int  # longest side the images are matched at; 0 = the input size


def resized_size(h: int, w: int, long_side: int) -> tuple[int, int]:
    """Upstream `get_resized_wh` (longest side `long_side`, rounded) + `get_divisible_wh` (multiples of 8, cut down)."""
    if long_side > 0:
        s = long_side / max(h, w)
        h, w = round(h * s), round(w * s)
    return h // DIVISOR * DIVISOR, w // DIVISOR * DIVISOR


class Matcher(nn.Module):
    """LoFTR on an RGB pair: `[1, 3, H1, W1]`, `[1, 3, H2, W2]` in `[0, 1]` -> keypoints, index pairs, scores."""

    gray: torch.Tensor

    def __init__(self, params: Params) -> None:
        """Build the network (no weights)."""
        super().__init__()
        self.params = params
        self.loftr = LoFTR(temp_bug_fix=params.temp_bug_fix, thr=params.threshold, border_rm=params.border_rm)
        self.register_buffer("gray", torch.tensor(GRAY).view(1, 3, 1, 1), persistent=False)

    def prepare(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Grayscale at the matching size and the factors (x, y) back to input pixels."""
        h, w = image.shape[-2:]
        rh, rw = resized_size(h, w, self.params.long_side)
        if min(rh, rw) < DIVISOR * 2:
            msg = f"image of {w} x {h} px is too small for LoFTR (matched at {rw} x {rh}; at least 16 px per side)"
            raise ValueError(msg)
        g = (image * self.gray).sum(1, keepdim=True)
        if (rh, rw) != (h, w):
            g = F.interpolate(g, size=(rh, rw), mode="bilinear", align_corners=False)
        return g, torch.tensor([w / rw, h / rh], device=image.device)

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """The pair's matches (batch 1)."""
        (g1, s1), (g2, s2) = self.prepare(image1), self.prepare(image2)
        out = self.loftr(g1, g2)
        n = out["confidence"].shape[0]
        idx = torch.arange(n, device=image1.device)
        return {
            "keypoints1": out["keypoints0"] * s1,
            "keypoints2": out["keypoints1"] * s2,
            "pairs": torch.stack([idx, idx], -1),
            "scores": out["confidence"],
        }


def load_matcher(ctx: Context) -> Matcher:
    """The matcher with the variant's weights (a Lightning checkpoint; the indoor one prefixes `matcher.`)."""
    net = Matcher(ctx.params)
    sd = ctx.state_dict(0, key="state_dict", strip="matcher.")
    net.loftr.load_state_dict(sd, strict=True)
    return net.eval().requires_grad_(requires_grad=False).to(ctx.device)


class Portable(Runner):
    """Plain PyTorch, float32."""

    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = load_matcher(ctx)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """The matched points of both images, their index pairs and confidences."""
        return self.net(image1, image2)

    def module(self) -> torch.nn.Module:
        """The matcher (data-dependent match counts: compiles with graph breaks, does not export)."""
        return Forward(self.net, lambda net, image1, image2: net(image1, image2))


RUNNERS = {"portable": Portable}
