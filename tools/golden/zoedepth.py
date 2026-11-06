"""ZoeDepth goldens from the reference implementation.

Reference: upstream isl-org/ZoeDepth (`REF_COMMIT`) plus `patches/zoedepth-hub-timm.patch` (load the MiDaS hub code from
a local tree instead of GitHub; skip the checkpoint's BEiT `relative_position_index` buffers, which timm >= 0.9
recomputes), with the MiDaS hub code from `midas.py`'s reference (MiDaS + `patches/midas-timm.patch`). CPU float32,
`build_model(get_config("zoedepth_nk", "infer"))` with the local checkpoint and `model.infer(x)` defaults (pad_input,
flip augmentation) on the RGB image / 255. Run this with a Python that has the upstream dependencies (timm,
torchvision).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/zoedepth.py ZOEDEPTH_CHECKOUT MIDAS_CHECKOUT ZoeD_M12_NK.pt
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from _common import image, reference_tree, write

UPSTREAM = "https://github.com/isl-org/ZoeDepth"
REF_COMMIT = "d87f17b2f5fdcb174cf4fb115491f4a6c60de152"
PATCH = "zoedepth-hub-timm.patch"
MIDAS_COMMIT = "454597711a62eabcbf7d1e89f3fb9f569051ac9b"
MIDAS_PATCH = "midas-timm.patch"


def main() -> None:
    repo, midas, weights = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    os.environ["MIDAS_LOCAL_REPO"] = str(reference_tree(midas, MIDAS_COMMIT, MIDAS_PATCH))
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT, PATCH)))
    sys.path.remove(str(Path(__file__).resolve().parent))  # this script (same name) would shadow upstream's zoedepth/
    from zoedepth.models.builder import build_model  # noqa: PLC0415  # reference checkout
    from zoedepth.utils.config import get_config  # noqa: PLC0415  # reference checkout

    conf = get_config("zoedepth_nk", "infer")
    conf.pretrained_resource = f"local::{weights}"
    model = build_model(conf).eval()
    x = torch.from_numpy(image().astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    with torch.no_grad():
        depth = model.infer(x)[0, 0].numpy()
    write(
        "zoedepth-nk@1",
        {"depth": depth},
        family="zoedepth",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=PATCH,
        settings=f"CPU float32, zoedepth_nk infer config, model.infer (pad_input, flip), torch {torch.__version__}",
        task="depth",
    )
    print(f"  median depth {np.median(depth):.4f}")


if __name__ == "__main__":
    main()
