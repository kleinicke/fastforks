"""OWLv2 goldens from the transformers implementation.

Reference: transformers `COMMIT` (v5.19.0) `Owlv2ImageProcessor` (pad to a square, blur, resize to the variant's size,
CLIP normalisation), `CLIPTokenizer` (the card's `clip-bpe` vocabulary, padded to 16 tokens) and
`Owlv2ForObjectDetection` with the card's weights, CPU float32, on scikit-image's `coffee.png` (`IMAGE_SHA256`) with the
80 `coco-80` class names as the text queries (verbatim). Post-processing as `post_process_object_detection`: the best
query per box, a sigmoid, boxes centre / size -> corners times the padded side. Detections with a score above the
variant's `THRESHOLDS` entry (mid-gap) are kept, sorted by score; stored as in tools/golden/dfine.py.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/owlv2.py WEIGHTS_DIR COFFEE_PNG LABELS_TXT [b16 l14]

WEIGHTS_DIR holds the card's weight files; LABELS_TXT is src/fastforks/_core/labels/coco-80.txt.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from _common import goldens, write
from PIL import Image

UPSTREAM = "https://github.com/huggingface/transformers"
COMMIT = "c4c4605f961877852b89b97478536bb3a26b6fd1"
IMAGE_SHA256 = "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7"  # skimage 0.26 data/coffee.png
THRESHOLDS = {"b16": 0.21, "l14": 0.13}  # per variant: mid-gap of the scores around the cut


def main() -> None:
    from transformers import CLIPTokenizer, Owlv2Config, Owlv2ForObjectDetection, Owlv2ImageProcessor  # noqa: PLC0415

    weights, png, labels_txt = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's coffee.png"
        raise SystemExit(msg)
    names = [s for s in labels_txt.read_text().splitlines() if s]
    img = Image.open(png).convert("RGB")
    tok = CLIPTokenizer(str(weights / "clip-bpe.vocab.json"), str(weights / "clip-bpe.merges.txt"))
    text = tok(names, padding="max_length", max_length=16, truncation=True, return_tensors="pt")
    side = max(img.width, img.height)
    for key in sys.argv[4:] or THRESHOLDS:
        ref = f"owlv2-{key}"
        config = Owlv2Config(**json.loads((weights / f"{ref}.config.json").read_text()))
        size = config.vision_config.image_size
        proc = Owlv2ImageProcessor(size={"height": size, "width": size})
        x = proc(images=img, return_tensors="pt")["pixel_values"]
        model = Owlv2ForObjectDetection(config).eval()
        model.load_state_dict(_load(weights / f"{ref}.safetensors"))
        with torch.inference_mode():
            out = model(input_ids=text["input_ids"], attention_mask=text["attention_mask"], pixel_values=x)
        best, labels = out.logits[0].max(-1)
        scores = best.sigmoid()
        cx, cy, bw, bh = out.pred_boxes[0].unbind(-1)
        boxes = torch.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), -1) * side
        threshold = THRESHOLDS[key]
        keep = (scores > threshold).nonzero()[:, 0]
        keep = keep[scores[keep].argsort(descending=True, stable=True)]
        maps = {
            "input": np.asarray(img),
            "boxes": boxes[keep].numpy(),
            "scores": scores[keep].numpy(),
            "labels": labels[keep].numpy(),
        }
        write(
            f"{ref}@1",
            {k: v.astype(np.float32) for k, v in maps.items()},
            family="owlv2",
            upstream=f"{UPSTREAM}@{COMMIT}",
            patch=None,
            settings=f"CPU float32, transformers Owlv2ImageProcessor + Owlv2ForObjectDetection, {size} px, skimage "
            f"coffee.png, the coco-80 names as queries, score threshold {threshold}, torch {torch.__version__}",
            task="detection",
        )
        store = goldens.load("owlv2", "detection")
        store["refs"][f"{ref}@1"]["options"] = [["score_threshold", threshold]]  # the test loads with these
        goldens.dump("owlv2", store, "detection")
        near = np.sort(scores.numpy())[::-1]
        n = len(keep)
        print(f"  {ref}: {n} kept, scores around the cut {np.round(near[max(0, n - 3) : n + 3], 3)}")


def _load(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file  # noqa: PLC0415

    return load_file(str(path))


if __name__ == "__main__":
    main()
