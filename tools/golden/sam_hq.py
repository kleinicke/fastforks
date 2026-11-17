"""HQ-SAM goldens from the reference implementation.

Reference: upstream SysCV/sam-hq (`REF_COMMIT`), CPU float32, `SamPredictor.set_image` on the synthetic image, then
`predict` with the fixed prompts of `tests/synth.py` (a centre point with multimask output; a box + the point, single
mask) and upstream's default `hq_token_only=False`: one mask per prompt, SAM's mask (with multimask output the
best-scored of its three) plus the HQ mask. Stored: the low-resolution mask logits and IoU predictions (`box_` prefix
for the second prompt). Needs torchvision and timm (upstream's package imports its TinyViT encoder).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/sam_hq.py SAM_HQ_CHECKOUT WEIGHTS_DIR [vit_b vit_l vit_h]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from _common import image, reference_tree, write

from synth import prompts

UPSTREAM = "https://github.com/SysCV/sam-hq"
REF_COMMIT = "e696978d60352dc9a26b12631cd91781502c6546"
VARIANTS = {  # upstream registry key: (ref, checkpoint)
    "vit_b": ("sam-hq-vit-b", "sam_hq_vit_b.pth"),
    "vit_l": ("sam-hq-vit-l", "sam_hq_vit_l.pth"),
    "vit_h": ("sam-hq-vit-h", "sam_hq_vit_h.pth"),
}


def tree(repo: Path) -> None:
    """Put the pinned export of the checkout on `sys.path` (its package is `segment_anything`, like SAM's)."""
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT)))


def predictor(key: str, weights: Path) -> Any:  # noqa: ANN401 - upstream SamPredictor
    """Upstream `sam_model_registry[key]` with the file `weights`, loaded strictly (CPU, eval)."""
    from segment_anything import SamPredictor, sam_model_registry  # noqa: PLC0415  # reference checkout

    sam = sam_model_registry[key]()
    sam.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return SamPredictor(sam.eval())


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    tree(repo)
    img = image()
    for key in sys.argv[3:] or VARIANTS:
        ref, file = VARIANTS[key]
        pred = predictor(key, weights / file)
        maps = {}
        with torch.inference_mode():
            pred.set_image(img)
            for prefix, p in prompts().items():
                _, iou, low = pred.predict(
                    point_coords=np.array(p["input_points"], np.float64),
                    point_labels=np.array(p["input_labels"]),
                    box=np.array(p["input_boxes"], np.float64) if "input_boxes" in p else None,
                    multimask_output=bool(p["multimask_output"]),
                    hq_token_only=False,
                )
                maps |= {f"{prefix}pred_masks": low, f"{prefix}iou_scores": iou}
        write(
            f"{ref}@1",
            maps,
            family="sam_hq",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=(
                "CPU float32, SamPredictor set_image + predict (point; box + point; hq_token_only=False), "
                f"torch {torch.__version__}"
            ),
            task="promptable_segmentation",
        )
        print(f"  iou {maps['iou_scores']}, box iou {maps['box_iou_scores']}")


if __name__ == "__main__":
    main()
