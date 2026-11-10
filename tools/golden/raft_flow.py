"""RAFT optical-flow goldens from torchvision itself.

Reference: torchvision 0.29.1 (`TORCHVISION_COMMIT`) `raft_large` / `raft_small(weights=None)` with the variant's
checkpoint loaded strictly, CPU float32, `num_flow_updates=12` (the runner's default `iters`), the last prediction.
Inputs: a 384x512 crop of scikit-image's `astronaut.png` (public domain, NASA; `IMAGE_SHA256`) as image1 and the same
image warped by the affine map `warp()` documents (rotation 1.5 deg about the centre, scale 1.02, shift (+3.4, -2.1) px;
bilinear, edge-clamped, rounded to uint8) as image2, both normalised to [-1, 1] as torchvision's transforms do.
Stored as float32: `input` (image1, RGB), `input2` (image2, RGB) and `flow` ([2, H, W]: dx, dy in pixels).

    FASTFORKS_GOLDEN_PATH=... uv run --with scikit-image python tools/golden/raft_flow.py [raft-small ...]
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import skimage
import torch
import torchvision
from _common import write
from torchvision.models.optical_flow import raft_large, raft_small

from fastforks._core import registry
from fastforks._core.checkpoint import load_state_dict

FAMILY = "raft"
UPSTREAM = "https://github.com/pytorch/vision"
TORCHVISION_VERSION = "0.29.1"
TORCHVISION_COMMIT = "1f1a920d685d46dfbe187934b785b7ecb7a182f2"
IMAGE_SHA256 = "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5"  # skimage data/astronaut.png
CROP = (64, 0, 384, 512)  # y0, x0, height, width: H and W multiples of 8 (torchvision's RAFT requires it)
ANGLE, SCALE, SHIFT = 1.5, 1.02, (3.4, -2.1)  # degrees (counter-clockwise), zoom, (dx, dy) pixels


def warp(img: np.ndarray) -> np.ndarray:
    """`img` moved by the affine map p' = c + SCALE * R(ANGLE) (p - c) + SHIFT (c: the image centre), as uint8.

    Each output pixel samples the input at the inverse map, bilinearly in float64 with edge clamping, then rounds.
    """
    h, w = img.shape[:2]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    a = math.radians(ANGLE)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    u, v = xs - cx - SHIFT[0], ys - cy - SHIFT[1]
    sx = cx + (math.cos(a) * u - math.sin(a) * v) / SCALE  # inverse rotation (y down: R(-a) undoes R(a))
    sy = cy + (math.sin(a) * u + math.cos(a) * v) / SCALE
    sx, sy = np.clip(sx, 0, w - 1), np.clip(sy, 0, h - 1)
    x0, y0 = np.minimum(np.floor(sx).astype(int), w - 2), np.minimum(np.floor(sy).astype(int), h - 2)
    fx, fy = (sx - x0)[..., None], (sy - y0)[..., None]
    f = img.astype(np.float64)
    out = (
        f[y0, x0] * (1 - fx) * (1 - fy)
        + f[y0, x0 + 1] * fx * (1 - fy)
        + f[y0 + 1, x0] * (1 - fx) * fy
        + f[y0 + 1, x0 + 1] * fx * fy
    )
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255 * 2 - 1


def main() -> None:
    if torchvision.__version__.split("+")[0] != TORCHVISION_VERSION:
        msg = f"torchvision {torchvision.__version__}, the goldens pin {TORCHVISION_VERSION}"
        raise SystemExit(msg)
    png = Path(skimage.data.__file__).parent / "astronaut.png"
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's astronaut.png"
        raise SystemExit(msg)
    y0, x0, h, w = CROP
    img1 = np.ascontiguousarray(skimage.io.imread(png)[y0 : y0 + h, x0 : x0 + w, :3])
    img2 = warp(img1)
    for card, variant in registry.cards("flow").values():
        if card.family != FAMILY or (sys.argv[1:] and variant.id not in sys.argv[1:]):
            continue
        size = variant.params.get("size", card.params.get("size"))
        model = (raft_large if size == "large" else raft_small)(weights=None).eval()
        (ckpt,) = registry.weights(card, variant)
        model.load_state_dict(load_state_dict(ckpt), strict=True)
        with torch.inference_mode():
            flow = model(_tensor(img1), _tensor(img2), num_flow_updates=12)[-1][0].numpy()
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img1.astype(np.float32), "input2": img2.astype(np.float32), "flow": flow.astype(np.float32)},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{TORCHVISION_COMMIT}",
            patch=None,
            settings=f"CPU float32, torchvision {torchvision.__version__} raft_{size} num_flow_updates=12, "
            f"skimage astronaut.png {h}x{w} crop + affine warp ({ANGLE} deg, x{SCALE}, shift {SHIFT}), "
            f"torch {torch.__version__}",
            task="flow",
        )
        print(f"  {variant.id}: mean flow ({flow[0].mean():.3f}, {flow[1].mean():.3f}), |max| {np.abs(flow).max():.2f}")


if __name__ == "__main__":
    main()
