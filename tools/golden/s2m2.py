"""S2M2 goldens from the reference implementation.

Reference: upstream junhong-3dv/s2m2 at `REF_COMMIT` + `patches/s2m2-lookup-fix.patch` (the cost-volume lookup
convention the released weights were trained with; the lookup at `REF_COMMIT` is shifted by +0.5 px), CPU float32,
upstream padding.

    FASTFORKS_GOLDEN_PATH=... uv run python tools/golden/s2m2.py /path/to/s2m2-checkout WEIGHTS_DIR [s2m2-s ...]

The checkout only has to contain `REF_COMMIT`; the reference runs from a patched export of that commit.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch
from _common import pair, reference_tree, write

UPSTREAM = "https://github.com/junhong-3dv/s2m2"
REF_COMMIT = "8f3bd785df593619d955203c6f2c741bef6e31df"
PATCH = "s2m2-lookup-fix.patch"
SIZES = {"s2m2-s": (128, 1), "s2m2-m": (192, 2), "s2m2-l": (256, 3), "s2m2-xl": (384, 3)}


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    sys.modules["cv2"] = types.ModuleType("cv2")  # image_utils imports cv2 for file I/O only
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT, PATCH) / "src"))
    from s2m2.core.model.s2m2 import S2M2  # noqa: PLC0415 - reference checkout on sys.path
    from s2m2.core.utils.image_utils import image_crop, image_pad  # noqa: PLC0415 - reference checkout on sys.path

    left, right = (torch.from_numpy(x).permute(2, 0, 1)[None].float() for x in pair())
    h, w = left.shape[-2:]
    for ref in sys.argv[3:] or SIZES:
        ch, ntr = SIZES[ref]
        net = S2M2(feature_channels=ch, dim_expansion=1, num_transformer=ntr, use_positivity=True, refine_iter=3)
        net.my_load_state_dict(
            torch.load(weights / f"CH{ch}NTR{ntr}.pth", map_location="cpu", weights_only=True, mmap=True)["state_dict"]
        )
        with torch.inference_mode():
            disp, occ, conf = net.eval()(image_pad(left, 32), image_pad(right, 32))
        maps = {
            k: image_crop(v, (h, w))[0, 0].numpy()
            for k, v in (("disparity", disp), ("confidence", conf), ("occlusion", occ))
        }
        settings = f"CPU float32, torch {torch.__version__}"
        write(f"{ref}@1", maps, family="s2m2", upstream=f"{UPSTREAM}@{REF_COMMIT}", patch=PATCH, settings=settings)
        print(f"  median disparity {np.median(maps['disparity']):.4f}")


if __name__ == "__main__":
    main()
