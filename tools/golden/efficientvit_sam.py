"""EfficientViT-SAM goldens from the reference implementation.

Reference: upstream mit-han-lab/efficientvit (`REF_COMMIT`) with facebookresearch/segment-anything (`SAM_COMMIT`, its
prompt encoder and mask decoder), CPU float32, `EfficientViTSamPredictor.set_image` on the synthetic image, then
`predict` with the fixed prompts of `tests/synth.py` (a centre point with multimask output; a box + the point, single
mask). Stored: the low-resolution mask logits and IoU predictions (`box_` prefix for the second prompt). Needs
torchvision (the `torchvision` extra / dev groups).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/efficientvit_sam.py EFFICIENTVIT_CHECKOUT SAM_CHECKOUT WEIGHTS_DIR \
        [l0 l1 l2 xl0 xl1]
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from _common import image, reference_tree, write

from synth import prompts

UPSTREAM = "https://github.com/mit-han-lab/efficientvit"
REF_COMMIT = "de7d7733cc0329f391b33f1f459271562ec27bd5"
SAM_COMMIT = "dca509fe793f601edb92606367a655c15ac00fdf"  # facebookresearch/segment-anything (upstream's dependency)
VARIANTS = ("l0", "l1", "l2", "xl0", "xl1")  # upstream `efficientvit_sam_<key>`; ref `efficientvit-sam-<key>`


TRAINING_ONLY = ("onnxsim", "omegaconf")  # imported by upstream's package __init__ chain, unused at inference


class _Stub(types.ModuleType):
    """A module whose every public name imports (as `object`): `from <stub> import name` at upstream's import time."""

    def __getattr__(self, attr: str) -> type:
        if attr.startswith("__"):
            raise AttributeError(attr)
        return object


def trees(repo: Path, sam_repo: Path) -> None:
    """Put the pinned exports of both checkouts on `sys.path`; stand in for missing training-only imports."""
    sys.path[:0] = [str(reference_tree(repo, REF_COMMIT)), str(reference_tree(sam_repo, SAM_COMMIT))]
    for name in TRAINING_ONLY:
        if name not in sys.modules and importlib.util.find_spec(name) is None:
            sys.modules[name] = _Stub(name)


def predictor(key: str, weights: Path) -> Any:  # noqa: ANN401 - upstream EfficientViTSamPredictor
    """Upstream `create_efficientvit_sam_model(efficientvit-sam-<key>)` with the file `weights` (CPU, eval)."""
    from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor  # noqa: PLC0415
    from efficientvit.sam_model_zoo import create_efficientvit_sam_model  # noqa: PLC0415

    sam = create_efficientvit_sam_model(f"efficientvit-sam-{key}", pretrained=True, weight_url=str(weights))
    return EfficientViTSamPredictor(sam.eval())


def main() -> None:
    repo, sam_repo, weights = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    trees(repo, sam_repo)
    img = image()
    for key in sys.argv[4:] or VARIANTS:
        pred = predictor(key, weights / f"efficientvit_sam_{key}.pt")
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
            f"efficientvit-sam-{key}@1",
            maps,
            family="efficientvit_sam",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=(
                "CPU float32, EfficientViTSamPredictor set_image + predict (point; box + point), segment-anything@"
                f"{SAM_COMMIT[:7]}, torch {torch.__version__}"
            ),
            task="promptable_segmentation",
        )
        print(f"  iou {maps['iou_scores']}, box iou {maps['box_iou_scores']}")


if __name__ == "__main__":
    main()
