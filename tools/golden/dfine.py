"""D-FINE goldens from the reference implementation.

Reference: Hugging Face transformers `DFineForObjectDetection` (`REF_VERSION`; the port of Peterande/D-FINE that the
`ustc-community/dfine-*-coco` weights are published for), CPU float32, on scikit-image's `coffee.png` (CC0, Rachel
Michetti; `IMAGE_SHA256`) resized to 640 x 640 with PIL bilinear, values in [0, 1] (the official deploy input), and
the RT-DETR postprocess (the 300 best of the 300 x 80 sigmoid scores, boxes centre / size -> corners times the image
size). Detections with a score of at least the variant's `THRESHOLDS` entry are kept, sorted by score (the ref's
`options` in golden.toml, which the test loads with). The threshold sits mid-gap, clear of the score gaps so fp16 GPU
runs keep the same set. Stored: the input image (`input`, RGB), boxes (input pixels), scores and labels (`coco-80`).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/dfine.py WEIGHTS_DIR COFFEE_PNG [n s m l x]

WEIGHTS_DIR holds one directory per size (`nano` ... `xlarge`) with the Hub files of `ustc-community/dfine-<size>-coco`.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from _common import goldens, write
from PIL import Image

UPSTREAM = "https://github.com/huggingface/transformers"
REF_VERSION = transformers.__version__
IMAGE_SHA256 = "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7"  # skimage 0.26 data/coffee.png
THRESHOLDS = {"n": 0.39, "s": 0.42, "m": 0.58, "l": 0.4, "x": 0.5}  # per variant: mid-gap of the scores around the cut
SIZES = {"n": "nano", "s": "small", "m": "medium", "l": "large", "x": "xlarge"}


def main() -> None:
    weights, png = Path(sys.argv[1]), Path(sys.argv[2])
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's coffee.png"
        raise SystemExit(msg)
    img = Image.open(png).convert("RGB")
    x = torch.from_numpy(np.asarray(img.resize((640, 640), Image.Resampling.BILINEAR)).copy()).permute(2, 0, 1)[None]
    for key in sys.argv[3:] or SIZES:
        ref = f"dfine-{key}-coco"
        model = transformers.DFineForObjectDetection.from_pretrained(weights / SIZES[key], dtype=torch.float32).eval()
        with torch.inference_mode():
            out = model(pixel_values=x / 255)
        scores, idx = out.logits.sigmoid()[0].flatten().topk(300)
        labels, query = idx % 80, idx // 80
        cx, cy, bw, bh = out.pred_boxes[0][query].unbind(-1)
        boxes = torch.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), -1) * torch.tensor(
            [img.width, img.height, img.width, img.height]
        )
        threshold = THRESHOLDS[key]
        keep = scores >= threshold
        maps = {
            "input": np.asarray(img),
            "boxes": boxes[keep].numpy(),
            "scores": scores[keep].numpy(),
            "labels": labels[keep].numpy(),
        }
        write(
            f"{ref}@1",
            {k: v.astype(np.float32) for k, v in maps.items()},
            family="dfine",
            upstream=f"{UPSTREAM}@v{REF_VERSION}",
            patch=None,
            settings=f"CPU float32, transformers DFineForObjectDetection + RT-DETR postprocess, 640x640 PIL bilinear, "
            f"skimage coffee.png, score threshold {threshold}, torch {torch.__version__}",
            task="detection",
        )
        store = goldens.load("dfine", "detection")
        store["refs"][f"{ref}@1"]["options"] = [["score_threshold", threshold]]  # the test loads with these
        goldens.dump("dfine", store, "detection")
        near = np.sort(scores.numpy())[::-1]
        n = int(keep.sum())
        print(f"  {ref}: {n} kept, scores around the cut {np.round(near[max(0, n - 2) : n + 3], 3)}")


if __name__ == "__main__":
    main()
