"""UniDepth V2 goldens from the reference implementation.

Reference: upstream lpiccinelli-eth/UniDepth (`REF_COMMIT`) + `PATCH` (the predicted intrinsics are mapped back with
the actual resize per axis, not the resize factor before the sides are rounded to multiples of 14), CPU float32,
`UniDepthV2.infer(rgb)` without a camera, `resolution_level` unset, bilinear output interpolation (the released demo
settings). Run this with a Python that has the upstream dependencies (einops, torchvision, huggingface_hub,
safetensors).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/unidepth_v2.py UNIDEPTH_CHECKOUT WEIGHTS_DIR

`WEIGHTS_DIR` holds the Hugging Face `lpiccinelli/unidepth-v2-vitl14` files (`config.json`, `model.safetensors`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from _common import image, reference_tree, write
from safetensors.torch import load_file

UPSTREAM = "https://github.com/lpiccinelli-eth/UniDepth"
REF_COMMIT = "8d8cfe4c7ee15297099983607febf0d4f32eb3d6"
PATCH = "unidepth-v2-per-axis-intrinsics.patch"


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT, PATCH)))
    from unidepth.models import UniDepthV2  # noqa: PLC0415  # reference checkout

    net = UniDepthV2(json.loads((weights / "config.json").read_text()))
    net.load_state_dict(load_file(weights / "model.safetensors"))
    net.interpolation_mode = "bilinear"
    rgb = torch.from_numpy(image()).permute(2, 0, 1)  # uint8 CHW, as the upstream demo
    with torch.inference_mode():
        out = net.eval().infer(rgb)
    maps = {
        "depth": out["depth"][0, 0].numpy(),
        "points": out["points"][0].permute(1, 2, 0).numpy(),
        "intrinsics": out["intrinsics"][0].numpy() - np.array([[0, 0, 0.5], [0, 0, 0.5], [0, 0, 0]]),  # -> OpenCV
        "confidence": out["confidence"][0, 0].numpy(),
    }
    write(
        "unidepth-v2-vitl14@1",
        maps,
        family="unidepth_v2",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=PATCH,
        settings=f"CPU float32, infer(rgb) without camera, resolution_level unset, torch {torch.__version__}",
        task="depth",
    )
    print(f"  median depth {np.median(maps['depth']):.4f}, K {maps['intrinsics'].round(2).tolist()}")


if __name__ == "__main__":
    main()
