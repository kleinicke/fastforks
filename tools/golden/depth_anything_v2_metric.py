"""Depth Anything V2 metric goldens from the reference implementation (upstream `metric_depth/`).

Reference: upstream DepthAnything/Depth-Anything-V2 (`REF_COMMIT`, the relative family's commit), CPU float32,
`metric_depth`'s `DepthAnythingV2(max_depth=...).infer_image` with the default `input_size=518` (OpenCV bicubic
resize, ImageNet normalisation, sigmoid head * max_depth, bilinear upsampling back). Run with a Python that has OpenCV
and no visible CUDA device (`infer_image` moves its input to CUDA when there is one).

    CUDA_VISIBLE_DEVICES= FASTFORKS_GOLDEN_PATH=... python tools/golden/depth_anything_v2_metric.py \
        DAV2_CHECKOUT WEIGHTS_DIR [hypersim-s vkitti-l ...]

`WEIGHTS_DIR` holds `depth_anything_v2_metric_{hypersim,vkitti}_vit{s,b,l}.pth` (directly or one level down).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from _common import image, reference_tree, write

UPSTREAM = "https://github.com/DepthAnything/Depth-Anything-V2"
REF_COMMIT = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"
ARCHS = {  # size: (arch, upstream metric_depth/run.py model_configs)
    "s": ("vits", {"features": 64, "out_channels": [48, 96, 192, 384]}),
    "b": ("vitb", {"features": 128, "out_channels": [96, 192, 384, 768]}),
    "l": ("vitl", {"features": 256, "out_channels": [256, 512, 1024, 1024]}),
}
MAX_DEPTH = {"hypersim": 20.0, "vkitti": 80.0}  # the metric_depth README / model cards


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT) / "metric_depth"))
    sys.path.remove(str(Path(__file__).resolve().parent))  # tools/golden/depth_anything_v2.py would shadow upstream's
    from depth_anything_v2.dpt import DepthAnythingV2  # noqa: PLC0415 - reference checkout

    bgr = np.ascontiguousarray(image()[..., ::-1])  # infer_image takes cv2.imread's BGR
    names = sys.argv[3:] or [f"{d}-{s}" for d in MAX_DEPTH for s in ARCHS]
    for name in names:
        ds, size = name.split("-")
        arch, cfg = ARCHS[size]
        file = f"depth_anything_v2_metric_{ds}_{arch}.pth"
        path = next(p for p in (weights / file, *weights.glob(f"*/{file}")) if p.exists())
        net = DepthAnythingV2(encoder=arch, **cfg, max_depth=MAX_DEPTH[ds])
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        with torch.inference_mode():
            depth = net.eval().infer_image(bgr)
        write(
            f"depth-anything-v2-metric-{name}@1",
            {"depth": depth},
            family="depth_anything_v2",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, metric_depth infer_image(input_size=518), max_depth {MAX_DEPTH[ds]:g}, "
            f"torch {torch.__version__}",
            task="depth",
        )
        print(f"  median depth {np.median(depth):.4f} m")


if __name__ == "__main__":
    main()
