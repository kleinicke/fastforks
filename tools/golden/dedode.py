"""DeDoDe v2 matching goldens from the reference implementation.

Reference: upstream Parskatt/DeDoDe (`REF_COMMIT`) `dedode_detector_L` + `dedode_descriptor_B` / `_G` with the card's
files and `DualSoftMaxMatcher.match(normalize=True, inv_temp=20, threshold=<card>)`, as the README; CPU float32 (the
CUDA-only autocasts are inactive on CPU; -G's DINOv2, which upstream casts to float16 whenever amp is on, is cast back
to float32). Input as upstream's `read_image`, in float: resized to the card's `size` square (bicubic, antialiased =
PIL's filter, clamped, not rounded to uint8) and ImageNet-normalised. Inputs: the RAFT goldens' pair
(tools/golden/raft_flow.py: a 384x512 astronaut.png crop and its affine warp) in [0, 1]. Stored as float32: `input`,
`input2` (RGB), `keypoints1` / `keypoints2` `[N, 2]` (upstream `to_pixel_coords` - 0.5: pixel centres at integers),
`pairs` `[K, 2]` and `scores` `[K]` (the dual-softmax probability).

    FASTFORKS_GOLDEN_PATH=... PYTHONPATH=<einops> python tools/golden/dedode.py DEDODE_CHECKOUT
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from _common import reference_tree, write
from raft_flow import CROP, IMAGE_SHA256, warp

from fastforks._core import registry

if TYPE_CHECKING:
    from collections.abc import Callable

FAMILY = "dedode"
UPSTREAM = "https://github.com/Parskatt/DeDoDe"
REF_COMMIT = "6d156183f4dc84cd704ae779eebc8350995c5b06"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def prepare(image: torch.Tensor, size: int) -> torch.Tensor:
    """`[3, H, W]` in [0, 1] -> `[1, 3, size, size]` normalised."""
    mean, std = (torch.tensor(v, device=image.device).view(1, 3, 1, 1) for v in (MEAN, STD))
    x = F.interpolate(image[None], size=(size, size), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    return (x - mean) / std


def build(
    files: list[Path], params: dict, *, amp: bool, device: torch.device
) -> Callable[[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
    """Upstream DeDoDe on `device`: `[3, H, W]` pair in [0, 1] -> keypoints (input pixels), pairs, scores.

    `params`: the card's params and options merged. The reference checkout must be on `sys.path`. `amp`: upstream's
    float16 autocast and float16 DINOv2 (its default on CUDA); off: float32.
    """
    import DeDoDe  # noqa: PLC0415  # reference checkout

    load = [torch.load(f, map_location="cpu", weights_only=True) for f in files]
    detector = DeDoDe.dedode_detector_L(device=device, weights=load[0])
    if params["descriptor"] == "G":
        descriptor = DeDoDe.dedode_descriptor_G(device=device, weights=load[1], dinov2_weights=load[2])
    else:
        descriptor = DeDoDe.dedode_descriptor_B(device=device, weights=load[1])
    if not amp:
        for m in [*detector.modules(), *descriptor.modules()]:
            if hasattr(m, "amp_dtype"):
                m.amp = False
        if params["descriptor"] == "G":  # upstream casts it to float16 whenever amp was on at construction
            dino = descriptor.encoder.frozen_dinov2
            dino.amp_dtype = torch.float32
            dino.dinov2_vitl14[0] = dino.dinov2_vitl14[0].float()
    thr, size, num = params["threshold"], int(params["size"]), int(params["max_keypoints"])

    @torch.inference_mode()
    def run(image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        out = []
        for t in (image1, image2):
            batch = {"image": prepare(t.to(device), size)}
            kpts = detector.detect(batch, num_keypoints=num)["keypoints"]
            out.append((kpts, descriptor.describe_keypoints(batch, kpts)["descriptions"]))
        p = DeDoDe.utils.dual_softmax_matcher(out[0][1], out[1][1], normalize=True, inv_temperature=20)
        inds = torch.nonzero(
            (p == p.max(dim=-1, keepdim=True).values) * (p == p.max(dim=-2, keepdim=True).values) * (p > thr)
        )  # upstream `DualSoftMaxMatcher.match` up to its indices (it returns the matched points)
        px = [(o[0][0] + 1) / 2 * torch.tensor([t.shape[-1], t.shape[-2]], device=device) - 0.5
              for o, t in zip(out, (image1, image2), strict=True)]  # fmt: skip
        return {
            "keypoints1": px[0],
            "keypoints2": px[1],
            "pairs": inds[:, 1:],
            "scores": p[inds[:, 0], inds[:, 1], inds[:, 2]],
        }

    return run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of Parskatt/DeDoDe")
    a = ap.parse_args()
    import skimage  # noqa: PLC0415

    png = Path(skimage.data.__file__).parent / "astronaut.png"
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's astronaut.png"
        raise SystemExit(msg)
    y0, x0, h, w = CROP
    img1 = np.ascontiguousarray(skimage.io.imread(png)[y0 : y0 + h, x0 : x0 + w, :3])
    img2 = warp(img1)
    t1, t2 = (torch.from_numpy(i).permute(2, 0, 1).float() / 255 for i in (img1, img2))
    sys.path.insert(0, str(reference_tree(a.checkout, REF_COMMIT)))
    for card, variant in registry.cards("matching").values():
        if card.family != FAMILY:
            continue
        params = dict(card.params) | dict(variant.params) | dict(card.options)
        size, num = int(params["size"]), int(params["max_keypoints"])
        run = build(registry.weights(card, variant), params, amp=False, device=torch.device("cpu"))
        maps = run(t1, t2)
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img1.astype(np.float32), "input2": img2.astype(np.float32)}
            | {k: v.numpy().astype(np.float32) for k, v in maps.items()},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, upstream dedode_detector_L(v2) + dedode_descriptor_{params['descriptor']} + "
            f"DualSoftMaxMatcher(normalize, inv_temp 20, threshold {params['threshold']}), {num} keypoints, "
            f"{size}x{size} input (torch bicubic antialias), skimage astronaut.png {h}x{w} crop + affine warp as "
            f"raft_flow.py, torch {torch.__version__}",
            task="matching",
        )
        print(f"  {variant.id}: {len(maps['pairs'])} matches")


if __name__ == "__main__":
    main()
