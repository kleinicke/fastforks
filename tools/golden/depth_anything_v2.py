"""Depth Anything V2 goldens from the reference implementation.

Reference: upstream DepthAnything/Depth-Anything-V2 (`REF_COMMIT`), CPU float32, `DepthAnythingV2.infer_image` with
the default `input_size=518` (OpenCV bicubic resize, ImageNet normalisation, bilinear upsampling back). Run this with
a Python that has OpenCV installed.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/depth_anything_v2.py DAV2_CHECKOUT WEIGHTS_DIR [vits vitb vitl]

`WEIGHTS_DIR` holds `depth_anything_v2_vit{s,b,l}.pth` (directly or in `Depth-Anything-V2-{Small,Base,Large}/`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from _common import image, reference_tree, write

UPSTREAM = "https://github.com/DepthAnything/Depth-Anything-V2"
REF_COMMIT = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"
VARIANTS = {  # arch: (ref, upstream run.py model_configs, weights subdirectory)
    "vits": ("depth-anything-v2-s", {"features": 64, "out_channels": [48, 96, 192, 384]}, "Small"),
    "vitb": ("depth-anything-v2-b", {"features": 128, "out_channels": [96, 192, 384, 768]}, "Base"),
    "vitl": ("depth-anything-v2-l", {"features": 256, "out_channels": [256, 512, 1024, 1024]}, "Large"),
}


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT)))
    # upstream's depth_anything_v2/ has no __init__.py, so this script (same name, on sys.path) would shadow it
    sys.path.remove(str(Path(__file__).resolve().parent))
    from depth_anything_v2.dpt import DepthAnythingV2  # noqa: PLC0415 - reference checkout

    bgr = np.ascontiguousarray(image()[..., ::-1])  # infer_image takes cv2.imread's BGR
    for arch in sys.argv[3:] or VARIANTS:
        ref, cfg, sub = VARIANTS[arch]
        file = f"depth_anything_v2_{arch}.pth"
        path = next(p for p in (weights / file, weights / f"Depth-Anything-V2-{sub}" / file) if p.exists())
        net = DepthAnythingV2(encoder=arch, **cfg)
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        with torch.inference_mode():
            inv = net.eval().infer_image(bgr)
        write(
            f"{ref}@1",
            {"inverse_depth": inv},
            family="depth_anything_v2",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, infer_image(input_size=518), torch {torch.__version__}",
            task="depth",
        )
        print(f"  median inverse depth {np.median(inv):.4f}")


if __name__ == "__main__":
    main()
