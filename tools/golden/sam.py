"""SAM goldens from the reference implementation.

Reference: upstream facebookresearch/segment-anything (`REF_COMMIT`), CPU float32, `SamPredictor.set_image` on the
synthetic image, then `predict` with the fixed prompts of `tests/synth.py` (a centre point with multimask output; a
box + the point, single mask). Stored: the low-resolution mask logits and IoU predictions (`box_` prefix for the
second prompt). Needs torchvision (the `torchvision` extra / dev groups).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/sam.py SAM_CHECKOUT WEIGHTS_DIR [vit_b vit_l vit_h]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from _common import image, reference_tree, write

from synth import prompts

UPSTREAM = "https://github.com/facebookresearch/segment-anything"
REF_COMMIT = "dca509fe793f601edb92606367a655c15ac00fdf"
VARIANTS = {  # upstream registry key: (ref, checkpoint)
    "vit_b": ("sam-vit-b", "sam_vit_b_01ec64.pth"),
    "vit_l": ("sam-vit-l", "sam_vit_l_0b3195.pth"),
    "vit_h": ("sam-vit-h", "sam_vit_h_4b8939.pth"),
}


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT)))
    from segment_anything import SamPredictor, sam_model_registry  # noqa: PLC0415  # reference checkout

    img = image()
    for key in sys.argv[3:] or VARIANTS:
        ref, file = VARIANTS[key]
        sam = sam_model_registry[key]()
        sam.load_state_dict(torch.load(weights / file, map_location="cpu", weights_only=True))
        pred = SamPredictor(sam.eval())
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
            family="sam",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, SamPredictor set_image + predict (point; box + point), torch {torch.__version__}",
            task="promptable_segmentation",
        )
        print(f"  iou {maps['iou_scores']}, box iou {maps['box_iou_scores']}")


if __name__ == "__main__":
    main()
