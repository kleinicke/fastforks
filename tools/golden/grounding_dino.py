"""Grounding DINO goldens from the transformers implementation.

Reference: transformers `COMMIT` (v5.19.0) `GroundingDinoImageProcessor` (short side 800, long side at most 1333,
ImageNet mean / std), `BertTokenizer` (the card's vocabulary) on the caption `" . ".join(names) + " ."` and
`GroundingDinoForObjectDetection` with the card's weights, CPU float32, on scikit-image's `coffee.png` (`IMAGE_SHA256`)
with the 80 `coco-80` class names. Post-processing as the authors' COCO evaluation (`PostProcessCocoGrounding`): the
sigmoid token scores averaged over each class name's tokens, the best 100 (query, class) pairs, boxes centre / size ->
corners in image pixels. Pairs with a score above the variant's `THRESHOLDS` entry (mid-gap) are kept, sorted by score;
stored as in tools/golden/dfine.py.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/grounding_dino.py WEIGHTS_DIR COFFEE_PNG LABELS_TXT [tiny base]

WEIGHTS_DIR holds the card's weight files; LABELS_TXT is src/fastforks/_core/labels/coco-80.txt.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from _common import goldens, write
from PIL import Image

UPSTREAM = "https://github.com/huggingface/transformers"
COMMIT = "c4c4605f961877852b89b97478536bb3a26b6fd1"
IMAGE_SHA256 = "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7"  # skimage 0.26 data/coffee.png
THRESHOLDS = {"tiny": 0.4, "base": 0.37}  # per variant: mid-gap of the scores around the cut
TOP = 100  # the card's default max_detections


def main() -> None:
    from safetensors.torch import load_file  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        BertTokenizer,
        GroundingDinoConfig,
        GroundingDinoForObjectDetection,
        GroundingDinoImageProcessor,
    )

    weights, png, labels_txt = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's coffee.png"
        raise SystemExit(msg)
    names = [s for s in labels_txt.read_text().splitlines() if s]
    img = Image.open(png).convert("RGB")
    tok = BertTokenizer(str(weights / "bert-uncased.vocab.txt"))
    text = tok(" . ".join(names) + " .", return_tensors="pt")
    ids = text["input_ids"][0].tolist()
    classes = torch.zeros(len(names), len(ids))  # each name's tokens (rows sum to 1)
    at = 1
    for c, n in enumerate(names):
        k = len(tok(n, add_special_tokens=False)["input_ids"])
        classes[c, at : at + k] = 1 / k
        at += k + 1  # the name and its "."
    if at != len(ids) - 1:
        msg = "caption token spans do not add up"
        raise SystemExit(msg)
    pv = GroundingDinoImageProcessor()(images=img, return_tensors="pt")
    for key in sys.argv[4:] or THRESHOLDS:
        ref = f"grounding-dino-{key}"
        config = GroundingDinoConfig.from_json_file(str(weights / f"gdino-{key}.config.json"))
        sd = load_file(str(weights / f"gdino-{key}.model.safetensors"))
        model = GroundingDinoForObjectDetection.from_pretrained(None, config=config, state_dict=sd).eval()
        with torch.inference_mode():
            out = model(**text, pixel_values=pv["pixel_values"], pixel_mask=pv["pixel_mask"])
        prob = out.logits[0, :, : len(ids)].sigmoid() @ classes.T  # [Q, C]
        scores, flat = prob.flatten().topk(TOP)
        labels = flat % len(names)
        cx, cy, bw, bh = out.pred_boxes[0][flat // len(names)].unbind(-1)
        boxes = torch.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), -1)
        boxes = boxes * torch.tensor([img.width, img.height, img.width, img.height])
        threshold = THRESHOLDS[key]
        keep = scores > threshold
        maps = {
            "input": np.asarray(img),
            "boxes": boxes[keep].numpy(),
            "scores": scores[keep].numpy(),
            "labels": labels[keep].numpy(),
        }
        write(
            f"{ref}@1",
            {k: v.astype(np.float32) for k, v in maps.items()},
            family="grounding_dino",
            upstream=f"{UPSTREAM}@{COMMIT}",
            patch=None,
            settings="CPU float32, transformers GroundingDinoImageProcessor + GroundingDinoForObjectDetection, "
            f"skimage coffee.png, the coco-80 names as one caption, mean token scores, score threshold {threshold}, "
            f"torch {torch.__version__}",
            task="detection",
        )
        store = goldens.load("grounding_dino", "detection")
        store["refs"][f"{ref}@1"]["options"] = [["score_threshold", threshold]]  # the test loads with these
        goldens.dump("grounding_dino", store, "detection")
        n = int(keep.sum())
        print(f"  {ref}: {n} kept, scores around the cut {np.round(scores[max(0, n - 3) : n + 3].numpy(), 3)}")


if __name__ == "__main__":
    main()
