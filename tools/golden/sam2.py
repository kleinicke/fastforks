"""SAM 2.1 (image) goldens from the reference implementation.

Reference: upstream facebookresearch/sam2 (`REF_COMMIT`), CPU float32, `build_sam2` (default post-processing settings)
+ `SAM2ImagePredictor.set_image` on the synthetic image, then `predict` with the fixed prompts of `tests/synth.py` (a
centre point with multimask output; a box + the point, single mask). Stored: the low-resolution mask logits (clamped
to +-32, as returned) and IoU predictions (`box_` prefix for the second prompt). Run this with a Python that has the
upstream dependencies (hydra-core, iopath).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/sam2.py SAM2_CHECKOUT WEIGHTS_DIR [tiny small base_plus large]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from _common import image, reference_tree, write

from synth import prompts

UPSTREAM = "https://github.com/facebookresearch/sam2"
REF_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
VARIANTS = {  # size: (ref, config, checkpoint)
    "tiny": ("sam2.1-hiera-tiny", "t"),
    "small": ("sam2.1-hiera-small", "s"),
    "base_plus": ("sam2.1-hiera-base-plus", "b+"),
    "large": ("sam2.1-hiera-large", "l"),
}


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT)))
    from sam2.build_sam import build_sam2  # noqa: PLC0415  # reference checkout
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: PLC0415  # reference checkout

    img = image()
    for key in sys.argv[3:] or VARIANTS:
        ref, cfg = VARIANTS[key]
        model = build_sam2(f"configs/sam2.1/sam2.1_hiera_{cfg}.yaml", str(weights / f"sam2.1_hiera_{key}.pt"), "cpu")
        pred = SAM2ImagePredictor(model)
        maps = {}
        with torch.inference_mode():
            pred.set_image(img)
            for prefix, p in prompts().items():
                _, iou, low = pred.predict(
                    point_coords=np.array(p["input_points"], np.float64),
                    point_labels=np.array(p["input_labels"]),
                    box=np.array(p["input_boxes"], np.float64) if "input_boxes" in p else None,
                    multimask_output=bool(p["multimask_output"]),
                )
                maps |= {f"{prefix}pred_masks": low, f"{prefix}iou_scores": iou}
        write(
            f"{ref}@1",
            maps,
            family="sam2",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=(
                "CPU float32, build_sam2 + SAM2ImagePredictor set_image + predict (point; box + point), "
                f"torch {torch.__version__}"
            ),
            task="promptable_segmentation",
        )
        print(f"  iou {maps['iou_scores']}, box iou {maps['box_iou_scores']}")


if __name__ == "__main__":
    main()
