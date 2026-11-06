"""detectron2 Mask / Faster R-CNN goldens from the reference implementation.

Reference: upstream facebookresearch/detectron2 (`REF_COMMIT`), CPU float32, the model-zoo config + `DefaultPredictor`
(BGR uint8 input, shortest edge 800, test NMS) on scikit-image's `coffee.png` (CC0, Rachel Michetti; `IMAGE_SHA256`):
4 confident instances each (cup, spoons, dining table), so boxes, scores, labels and masks are all checked. The score
threshold `SCORE_THRESHOLD` (in golden.toml as the ref's `options`, which the test loads with) sits well clear of every
score, so fp16 GPU runs keep the same set. Stored: the input image (`input`, RGB), the detections' boxes, scores,
labels and (Mask R-CNN) pasted masks, as float32. The Python-only package is enough (no `_C` extension: CPU NMS /
ROIAlign come from torchvision). Run this with a Python that has the upstream dependencies (fvcore, iopath,
torchvision).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/rcnn.py DETECTRON2_CHECKOUT WEIGHTS_DIR COFFEE_PNG [mask faster]
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from _common import goldens, reference_tree, write
from PIL import Image

UPSTREAM = "https://github.com/facebookresearch/detectron2"
REF_COMMIT = "1e3e13bbf607b54f62205c4c33922521822fb298"
IMAGE_SHA256 = "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7"  # skimage 0.26 data/coffee.png
SCORE_THRESHOLD = 0.35  # fixed: >= 0.079 from every score (kept: Mask 0.468+, Faster 0.580+; next: 0.271, 0.212)
VARIANTS = {  # key: (ref, config, checkpoint)
    "mask": ("mask-rcnn-r50-fpn-3x-coco", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x", "mask_rcnn_R_50_FPN_3x"),
    "faster": ("faster-rcnn-r50-fpn-3x-coco", "COCO-Detection/faster_rcnn_R_50_FPN_3x", "faster_rcnn_R_50_FPN_3x"),
}


def main() -> None:
    repo, weights, png = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's coffee.png"
        raise SystemExit(msg)
    rgb = np.asarray(Image.open(png).convert("RGB"))
    tree = reference_tree(repo, REF_COMMIT)
    sys.path.insert(0, str(tree))
    from detectron2.config import get_cfg  # noqa: PLC0415  # reference checkout
    from detectron2.engine import DefaultPredictor  # noqa: PLC0415  # reference checkout

    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    for key in sys.argv[4:] or VARIANTS:
        ref, config, checkpoint = VARIANTS[key]
        cfg = get_cfg()
        cfg.merge_from_file(str(tree / "configs" / f"{config}.yaml"))
        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.WEIGHTS = str(weights / f"{checkpoint}.pkl")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD
        predictor = DefaultPredictor(cfg)
        with torch.inference_mode():
            inst = predictor(bgr)["instances"]
        maps = {
            "input": rgb,
            "boxes": inst.pred_boxes.tensor.numpy(),
            "scores": inst.scores.numpy(),
            "labels": inst.pred_classes.numpy(),
        }
        if inst.has("pred_masks"):
            maps["masks"] = inst.pred_masks.numpy()
        write(
            f"{ref}@1",
            {k: v.astype(np.float32) for k, v in maps.items()},
            family="rcnn",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, model-zoo config + DefaultPredictor (BGR uint8), skimage coffee.png, score "
            f"threshold {SCORE_THRESHOLD}, torch {torch.__version__}",
            task="detection",
        )
        store = goldens.load("rcnn", "detection")
        store["refs"][f"{ref}@1"]["options"] = [["score_threshold", SCORE_THRESHOLD]]  # the test loads with these
        goldens.dump("rcnn", store, "detection")
        print(f"  {ref}: {len(inst)} detections, scores {maps['scores']}, labels {maps['labels']}")


if __name__ == "__main__":
    main()
