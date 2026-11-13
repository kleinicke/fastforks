"""FlexGlue runners: ALIKED keypoints, the DINOv3 ConvNeXt descriptor and the sparse matcher, and the portable forward.

As upstream's `FlexGlue.match_pair(image0, image1, detector=ALIKED(origin="corner"), sparse=True)` (its README usage):
ALIKED-n16 detects up to `max_keypoints` keypoints (score threshold 0.2, NMS radius 2) on the image resized to a long
side of `long_side` (kornia `resize`, antialiased, as LightGlue's extractor). The descriptor sees the image squashed
to `size` x `size` (bilinear, no antialiasing), ImageNet-normalised, and gives stride-8 features. Nine transformer
layers (rotary self-attention, cross-attention) turn both feature maps into cross features; the last layer's head
predicts, for each keypoint, its point in the other image (a 1.5-entmax readout around the best cell plus a sub-cell
offset) and a confidence. Matches are the mutual nearest neighbours of (keypoint, predicted point) pairs, kept when
both points lie inside the images, the geometric mean of both confidences is at least `threshold` and the mean
distance of the two predictions to their keypoints is below `max_distance_px` (pixels of the descriptor input).
Keypoints are returned in input pixels with pixel centres at integers (upstream's corner keypoints minus 0.5).

On CUDA the default dtype is float16 as upstream (`mp="float16"`): the transformer layers under float16 autocast,
everything else float32; bfloat16 likewise, float32 has no autocast (the CPU reference).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from fastforks.matching._runner import Forward, Runner
from fastforks.shared.aliked import ALIKED
from fastforks.shared.deform_conv import modulated_deform_conv
from fastforks.shared.lightglue import resize_long

from .net.descriptor import Descriptor
from .net.flexglue import FlexGlueNet, inside, mutual_nn, sq_dist_gemm

if TYPE_CHECKING:
    from fastforks.matching._runner import Context

ALIKED_THRESHOLD = 0.2  # fixed: upstream ALIKED `detection_threshold` (LightGlue extractor default)
ALIKED_NMS = 2  # fixed: upstream ALIKED `nms_radius`
STRIDE = 8  # fixed: upstream `FlexGlue.stride`, descriptor input pixels per feature cell
HEAD = "matcher.heads.9."  # fixed: the last layer's head (9 layers), the one sparse matching reads


@dataclass(frozen=True)
class Params:
    """Variant parameters and load options."""

    threshold: float  # confidence (geometric mean of both directions) a match needs (upstream 0.1)
    max_distance_px: float  # mean distance of both predictions to their keypoints, descriptor input pixels (3.0)
    max_keypoints: int  # ALIKED keypoints per image (upstream README 2048)
    long_side: int  # ALIKED's image: long side (LightGlue extractor resize 1024), aspect kept; 0 = input size
    size: int  # the descriptor's image: side of the square (upstream 512)


Described = tuple[torch.Tensor, torch.Tensor]  # keypoints [N, 2] in input pixels, upstream corner origin; features


class Matcher(nn.Module):
    """FlexGlue on an RGB pair: `[1, 3, H1, W1]`, `[1, 3, H2, W2]` in `[0, 1]` -> keypoints, index pairs, scores."""

    def __init__(self, params: Params) -> None:
        """Build the networks (no weights)."""
        super().__init__()
        self.params = params
        self.extractor = ALIKED(
            modulated_deform_conv,
            model_name="aliked-n16",
            max_num_keypoints=params.max_keypoints,
            detection_threshold=ALIKED_THRESHOLD,
            nms_radius=ALIKED_NMS,
        )
        self.descriptor = Descriptor()
        self.matcher = FlexGlueNet()
        self.amp: torch.dtype | None = None

    def describe(self, image: torch.Tensor) -> Described:
        """ALIKED keypoints (input pixels, corner origin as upstream) and the dense features of one image."""
        p = self.params
        h, w = image.shape[-2:]
        x = resize_long(image, p.long_side) if p.long_side > 0 else image
        scale = torch.tensor([x.shape[-1] / w, x.shape[-2] / h], device=x.device, dtype=x.dtype)
        kpts = self.extractor({"image": x})["keypoints"][0]
        sq = F.interpolate(image, (p.size, p.size), mode="bilinear", align_corners=False)
        return (kpts + 0.5) / scale, self.descriptor(sq)  # the patched LightGlue's `origin="corner"`

    def match(
        self, image1: torch.Tensor, image2: torch.Tensor, d1: Described, d2: Described
    ) -> dict[str, torch.Tensor]:
        """The pair's matches from both images' `describe` results."""
        (k0, f0), (k1, f1) = d1, d2
        sizes = [image.new_tensor([image.shape[-1], image.shape[-2]]) for image in (image1, image2)]
        resized = [f.new_tensor([STRIDE * f.shape[-1], STRIDE * f.shape[-2]]) for f in (f0, f1)]
        kpts = [k0[None], k1[None]]
        cf = self.matcher.decode(f0, f1, self.amp)
        p2d, conf = [], []
        for i, j in ((0, 1), (1, 0)):
            p2d_n, c = self.matcher.head(kpts[i] / sizes[i] * 2 - 1, cf[i], cf[j])
            p2d.append((p2d_n + 1) / 2 * sizes[j] * (resized[j] / sizes[j]))  # upstream `to_resized`
            conf.append(c)
        kr = [kpts[i] * (resized[i] / sizes[i]) for i in (0, 1)]
        valid = [inside(kr[i], resized[i], 0) & inside(p2d[i], resized[1 - i], 0) for i in (0, 1)]
        m0, scores0, valid0 = self.sparse_matches(kr, p2d, conf, resized, valid)
        idx = torch.nonzero(valid0[0])[:, 0]
        pairs = torch.stack([idx, m0[0, idx]], -1)
        return {"keypoints1": k0 - 0.5, "keypoints2": k1 - 0.5, "pairs": pairs, "scores": scores0[0, idx]}

    def sparse_matches(
        self,
        kr: list[torch.Tensor],
        p2d: list[torch.Tensor],
        conf: list[torch.Tensor],
        resized: list[torch.Tensor],
        valid: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Upstream `sparse_matches` (`gemm_nn`), direction 0 -> 1: matches, scores and their validity."""
        p = self.params
        c0, c1 = resized[0] / 2, resized[1] / 2  # centred on each image for float32 precision
        x = torch.cat([p2d[0] - c1, kr[0] - c0], -1)
        y = torch.cat([kr[1] - c1, p2d[1] - c0], -1)
        m0, _ = mutual_nn(sq_dist_gemm(x, y))
        take = m0.clamp(0)[..., None].expand(-1, -1, 2)
        dist0 = 0.5 * ((p2d[0] - kr[1].gather(1, take)).norm(dim=-1) + (kr[0] - p2d[1].gather(1, take)).norm(dim=-1))
        scores0 = (conf[0] * conf[1].gather(1, m0.clamp(0))).sqrt()
        valid0 = (m0 > -1) & (scores0 >= p.threshold) & valid[0] & valid[1].gather(1, m0.clamp(0))
        return m0, scores0, valid0 & (dist0 < p.max_distance_px)

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """The pair's features and matches (batch 1)."""
        return self.match(image1, image2, self.describe(image1), self.describe(image2))


def split_checkpoint(sd: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Upstream's checkpoint -> the descriptor's and the matcher's state dicts (the last head as `head.`).

    Skipped: the descriptor's layer-0 `descriptor_head`, the heads of layers 0-8 and the last head's `disambiguate`
    (covisibility) layer, which sparse matching does not read.
    """
    parts: dict[str, dict[str, torch.Tensor]] = {"descriptor": {}, "matcher": {}}
    for k, v in sd.items():
        if k.startswith(HEAD):
            if not k.startswith(HEAD + "disambiguate."):
                parts["matcher"]["head." + k[len(HEAD) :]] = v
        elif k.startswith("matcher.") and not k.startswith("matcher.heads."):
            parts["matcher"][k[len("matcher.") :]] = v
        elif k.startswith("extractor.") and not k.startswith("extractor.descriptor_head."):
            parts["descriptor"][k[len("extractor.") :]] = v
    return parts


def load_matcher(ctx: Context) -> Matcher:
    """The matcher with the variant's weights (file 0: FlexGlue, file 1: ALIKED-n16) on `ctx.device`."""
    net = Matcher(ctx.params)
    for name, sd in split_checkpoint(ctx.state_dict(0)).items():
        net.get_submodule(name).load_state_dict(sd, strict=True)
    net.extractor.load_state_dict(ctx.state_dict(1), strict=True)
    net.eval().requires_grad_(requires_grad=False).to(ctx.device)
    net.amp = None if ctx.dtype == "float32" else ctx.torch_dtype
    return net


class Portable(Runner):
    """Plain PyTorch; the transformer layers under float16 / bfloat16 autocast unless dtype is float32."""

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
        """Keypoints and dense features of one image: the per-image stage."""
        return self.net.describe(image)

    def match(self, image1: torch.Tensor, image2: torch.Tensor, described: tuple[Described, Described]) -> dict:
        """`__call__` with the two images' `describe` results."""
        return self.net.match(image1, image2, *described)

    def module(self) -> torch.nn.Module:
        """Extractor, descriptor and matcher (data-dependent match counts: compiles with graph breaks, no export)."""
        return Forward(self.net, lambda net, image1, image2: net(image1, image2))


RUNNERS = {"portable": Portable}
