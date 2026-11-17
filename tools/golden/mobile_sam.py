"""MobileSAM goldens from the reference implementation.

Reference: upstream ChaoningZhang/MobileSAM (`REF_COMMIT`, a segment-anything fork), CPU float32,
`SamPredictor.set_image` on the synthetic image, then `predict` with the fixed prompts of `tests/synth.py` (a centre
point with multimask output; a box + the point, single mask). Stored: the low-resolution mask logits and IoU
predictions (`box_` prefix for the second prompt). Needs timm and torchvision (upstream imports).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/mobile_sam.py MOBILESAM_CHECKOUT [WEIGHTS]

`WEIGHTS` defaults to the checkout's own `weights/mobile_sam.pt`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from _common import image, reference_tree, write

from synth import prompts

UPSTREAM = "https://github.com/ChaoningZhang/MobileSAM"
REF_COMMIT = "f706ad9c4eb7f219c00d9050e46328518ffb65d2"


def tree(repo: Path) -> Path:
    """Put the pinned export of the checkout on `sys.path` (its package is `mobile_sam`); returns it."""
    t = reference_tree(repo, REF_COMMIT)
    sys.path.insert(0, str(t))
    return t


def predictor(weights: Path) -> Any:  # noqa: ANN401 - upstream SamPredictor
    """Upstream `sam_model_registry["vit_t"]` with the file `weights`, loaded strictly (CPU, eval)."""
    from mobile_sam import SamPredictor, sam_model_registry  # noqa: PLC0415  # reference checkout

    sam = sam_model_registry["vit_t"]()
    sam.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return SamPredictor(sam.eval())


def main() -> None:
    t = tree(Path(sys.argv[1]))
    pred = predictor(Path(sys.argv[2]) if len(sys.argv) > 2 else t / "weights" / "mobile_sam.pt")  # noqa: PLR2004
    maps = {}
    with torch.inference_mode():
        pred.set_image(image())
        for prefix, p in prompts().items():
            _, iou, low = pred.predict(
                point_coords=np.array(p["input_points"], np.float64),
                point_labels=np.array(p["input_labels"]),
                box=np.array(p["input_boxes"], np.float64) if "input_boxes" in p else None,
                multimask_output=bool(p["multimask_output"]),
            )
            maps |= {f"{prefix}pred_masks": low, f"{prefix}iou_scores": iou}
    write(
        "mobile-sam@1",
        maps,
        family="mobile_sam",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=None,
        settings=f"CPU float32, SamPredictor set_image + predict (point; box + point), torch {torch.__version__}",
        task="promptable_segmentation",
    )
    print(f"  iou {maps['iou_scores']}, box iou {maps['box_iou_scores']}")


if __name__ == "__main__":
    main()
