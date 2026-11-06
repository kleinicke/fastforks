"""Depth Pro goldens from the reference implementation.

Reference: upstream apple/ml-depth-pro (`REF_COMMIT`), unpatched: `create_model_and_transforms` in float32 on the CPU,
`infer(transform(image), f_px=None)`. Intrinsics: the predicted focal length with the principal point at the image
centre. Run this with a Python that has timm (and pillow_heif, which upstream imports):

    FASTFORKS_GOLDEN_PATH=... python tools/golden/depth_pro.py ML_DEPTH_PRO_CHECKOUT depth_pro.pt
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch
from _common import image, reference_tree, write

UPSTREAM = "https://github.com/apple/ml-depth-pro"
REF_COMMIT = "9e65e4dbe9568d23c546fcec53302b10445e109e"


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    tree = reference_tree(repo, REF_COMMIT)
    sys.path.insert(0, str(tree / "src"))
    from depth_pro import create_model_and_transforms  # noqa: PLC0415
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT  # noqa: PLC0415

    torch.manual_seed(0)
    cfg = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(weights))
    net, transform = create_model_and_transforms(cfg, device=torch.device("cpu"), precision=torch.float32)
    net.eval()
    rgb = image()
    h, w = rgb.shape[:2]
    with torch.inference_mode():
        out = net.infer(transform(rgb), f_px=None)
    f = float(out["focallength_px"])
    k = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]])  # OpenCV pixel centres (upstream: w / 2)
    maps = {"depth": out["depth"].cpu().numpy(), "intrinsics": k}
    write(
        "depth-pro@1",
        maps,
        family="depth_pro",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=None,
        settings=f"CPU float32, infer(f_px=None), torch {torch.__version__}",
        task="depth",
    )
    print(f"  median depth {np.median(maps['depth']):.4f} f_px {f:.2f}")


if __name__ == "__main__":
    main()
