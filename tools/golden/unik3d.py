"""UniK3D goldens from the reference implementation.

Reference: upstream lpiccinelli-eth/UniK3D (`REF_COMMIT`), unpatched, CPU float32: `UniK3D(config)` with the released
weights, `resolution_level` 9 and bilinear output interpolation (the demo settings), `infer(rgb)` without a camera.
Run this with a Python that has the upstream dependencies (einops, timm, torchvision, huggingface_hub, safetensors):

    FASTFORKS_GOLDEN_PATH=... python tools/golden/unik3d.py UNIK3D_CHECKOUT WEIGHTS_DIR

`WEIGHTS_DIR` holds the Hugging Face `lpiccinelli/unik3d-vitl` files (`config.json`, `model.safetensors`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from _common import image, reference_tree, write
from safetensors.torch import load_file

UPSTREAM = "https://github.com/lpiccinelli-eth/UniK3D"
REF_COMMIT = "29f7862f3c3ee79c53ef0153dfc901ce511df575"


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT)))
    from unik3d.models import UniK3D  # noqa: PLC0415  # reference checkout

    net = UniK3D(json.loads((weights / "config.json").read_text()))
    net.load_state_dict(load_file(weights / "model.safetensors"))
    net.resolution_level = 9
    net.interpolation_mode = "bilinear"
    rgb = torch.from_numpy(image()).permute(2, 0, 1)  # uint8 CHW, as the upstream demo
    with torch.inference_mode():
        out = net.eval().infer(rgb)
    maps = {
        "depth": out["depth"][0, 0].numpy(),
        "points": out["points"][0].permute(1, 2, 0).numpy(),
        "confidence": out["confidence"][0, 0].numpy(),
    }
    write(
        "unik3d-vitl@1",
        maps,
        family="unik3d",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=None,
        settings=f"CPU float32, infer(rgb) without camera, resolution_level 9, bilinear, torch {torch.__version__}",
        task="depth",
    )
    print(f"  median depth {np.median(maps['depth']):.4f}")


if __name__ == "__main__":
    main()
