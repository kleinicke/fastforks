"""MiDaS v3.1 goldens from the reference implementation.

Reference: upstream isl-org/MiDaS (`REF_COMMIT`) plus `patches/midas-timm.patch` (timm >= 0.9: skip the checkpoint's
`relative_position_index` buffers, which timm now recomputes, and alias `drop_path1`), CPU float32, the `run.py` recipe
for `dpt_beit_large_512` with `optimize=False`: `read_image` (RGB / 255), `load_model`'s transform ("minimal" resize to
512, multiple of 32, OpenCV bicubic; mean = std = 0.5), `process` (bicubic upsampling to the input size). Run this
with a Python that has the upstream dependencies (timm, opencv-python).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/midas.py MIDAS_CHECKOUT dpt_beit_large_512.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from _common import image, reference_tree, write

UPSTREAM = "https://github.com/isl-org/MiDaS"
REF_COMMIT = "454597711a62eabcbf7d1e89f3fb9f569051ac9b"
PATCH = "midas-timm.patch"


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT, PATCH)))
    sys.path.remove(str(Path(__file__).resolve().parent))  # this script (same name) would shadow upstream's midas/
    from midas.model_loader import load_model  # noqa: PLC0415  # reference checkout
    from run import process  # noqa: PLC0415  # reference checkout

    cpu = torch.device("cpu")
    model, transform, net_w, net_h = load_model(cpu, str(weights), "dpt_beit_large_512", optimize=False)
    rgb = image().astype(np.float32) / 255.0
    x = transform({"image": rgb})["image"]
    with torch.no_grad():
        inv = process(
            cpu, model, "dpt_beit_large_512", x, (net_w, net_h), rgb.shape[1::-1], optimize=False, use_camera=False
        )
    write(
        "midas-dpt-beit-large-512@1",
        {"inverse_depth": inv},
        family="midas",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=PATCH,
        settings=f"CPU float32, run.py dpt_beit_large_512 (optimize=False), torch {torch.__version__}",
        task="depth",
    )
    print(f"  network input {tuple(x.shape)}, median inverse depth {np.median(inv):.4f}")


if __name__ == "__main__":
    main()
