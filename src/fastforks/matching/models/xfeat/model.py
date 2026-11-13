"""XFeat runners: weights -> extractor + matcher, and the portable forward pass.

Upstream `XFeat.detectAndCompute` per image (the image is bilinearly resized down to a multiple of 32, keypoints are
the heatmap maxima after a 5x5 NMS above `detection_threshold`, ranked by score, at most `max_keypoints`, mapped back
to the input pixels), then either upstream's mutual nearest neighbours of the descriptors (`match_xfeat`: no
similarity threshold; the score of a match is its cosine similarity) or LighterGlue (`match_lighterglue`: kornia's
LightGlue with XFeat's config = fastforks.shared.lightglue, match threshold 0.1). With `long_side > 0` each image is
first resized so its longest side is `long_side` (upstream LightGlue's resize; upstream XFeat does not resize).
Returns both keypoint sets, the index pairs of the matches and their scores. Float32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from fastforks.matching._runner import Forward, Runner
from fastforks.shared.lightglue import LightGlue, resize_long

from .net.xfeat import XFeat

if TYPE_CHECKING:
    from fastforks.matching._runner import Context

# upstream modules/lighterglue.py `default_conf_xfeat` (kornia LightGlue; `add_laf`, `scale_coef` are kornia-only, off)
LIGHTERGLUE = {
    "input_dim": 64,
    "descriptor_dim": 96,
    "add_scale_ori": False,
    "n_layers": 6,
    "num_heads": 1,
    "flash": True,
    "mp": False,
    "depth_confidence": -1,
    "width_confidence": 0.95,
    "filter_threshold": 0.1,
}


@dataclass(frozen=True)
class Params:
    """Variant parameters and load options."""

    matcher: str  # "mnn" (mutual nearest neighbours) or "lighterglue"
    detection_threshold: float  # upstream default 0.05
    max_keypoints: int  # per image (upstream `top_k`)
    long_side: int  # longest side the features are detected at; 0 = the input size (upstream)


class Matcher(nn.Module):
    """XFeat + matcher: `[1, 3, H1, W1]`, `[1, 3, H2, W2]` in `[0, 1]` -> keypoints, index pairs, scores."""

    def __init__(self, params: Params) -> None:
        """Build the networks (no weights)."""
        super().__init__()
        self.params = params
        self.xfeat = XFeat(top_k=params.max_keypoints, detection_threshold=params.detection_threshold)
        self.matcher = LightGlue(features=None, **LIGHTERGLUE) if params.matcher == "lighterglue" else None

    def extract(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Upstream `detectAndCompute` at `long_side`: keypoints and `image_size` of that resized image."""
        p = self.params
        x = resize_long(image, p.long_side) if p.long_side > 0 else image
        d = self.xfeat.detectAndCompute(x)[0]
        return {
            "keypoints": d["keypoints"],
            "descriptors": d["descriptors"],
            "image_size": torch.tensor(x.shape[-1:-3:-1]),
        }

    @staticmethod
    def to_input(kpts: torch.Tensor, size: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Keypoints of the resized image in input pixels (pixel centres at integers, as LightGlue's `extract`)."""
        h, w = image.shape[-2:]
        if (int(size[0]), int(size[1])) == (w, h):
            return kpts
        scale = torch.tensor([float(size[0]) / w, float(size[1]) / h], device=kpts.device, dtype=kpts.dtype)
        return (kpts + 0.5) / scale - 0.5

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """The pair's features and matches (batch 1)."""
        f1, f2 = self.extract(image1), self.extract(image2)
        if self.matcher is None:
            idx0, idx1 = self.xfeat.match(f1["descriptors"], f2["descriptors"], min_cossim=-1)
            pairs = torch.stack([idx0, idx1], dim=-1)
            scores = (f1["descriptors"][idx0] * f2["descriptors"][idx1]).sum(-1)
        else:
            dev = f1["keypoints"].device
            out = self.matcher(
                {
                    f"image{i}": {
                        "keypoints": f["keypoints"][None],
                        "descriptors": f["descriptors"][None],
                        "image_size": f["image_size"].to(dev)[None],
                    }
                    for i, f in enumerate((f1, f2))
                }
            )
            pairs, scores = out["matches"][0], out["scores"][0]
        k1 = self.to_input(f1["keypoints"], f1["image_size"], image1)
        k2 = self.to_input(f2["keypoints"], f2["image_size"], image2)
        return {"keypoints1": k1, "keypoints2": k2, "pairs": pairs, "scores": scores}


def load_matcher(ctx: Context) -> Matcher:
    """The matcher with the variant's weights (file 0: XFeat, file 1: LighterGlue), eval, on `ctx.device`."""
    net = Matcher(ctx.params)
    net.xfeat.net.load_state_dict(ctx.state_dict(0), strict=True)
    if net.matcher is not None:  # upstream LighterGlue: drop the `matcher.` prefix, then LightGlue's renames
        sd = ctx.state_dict(1)  # also holds a training-time XFeat copy (`extractor.*`) upstream ignores (strict=False)
        net.matcher.load_weights(
            {k.replace("matcher.", ""): v for k, v in sd.items() if not k.startswith("extractor.")}
        )
    return net.eval().requires_grad_(requires_grad=False).to(ctx.device)


class Portable(Runner):
    """Plain PyTorch, float32."""

    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the networks."""
        super().__init__(ctx)
        self.net = load_matcher(ctx)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Keypoints of both images, the index pairs of their matches and the match scores."""
        return self.net(image1, image2)

    def module(self) -> torch.nn.Module:
        """Extractor and matcher (data-dependent keypoint counts: compiles with graph breaks, does not export)."""
        return Forward(self.net, lambda net, image1, image2: net(image1, image2))


RUNNERS = {"portable": Portable}
