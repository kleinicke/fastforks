"""DEIM goldens (the `deim-*-coco` variants of the D-FINE family) from the official DEIM code.

Reference: Intellindust-AI-Lab/DEIM at `COMMIT` (`configs/deim_dfine/deim_hgnetv2_<size>_coco.yml`, the authors'
`model` checkpoints, `model.deploy()`), CPU float32, on scikit-image's `coffee.png` (`IMAGE_SHA256`) resized to
640 x 640 with PIL bilinear, values in [0, 1] (the official deploy input), and the RT-DETR postprocess (the 300 best of
the 300 x 80 sigmoid scores, boxes centre / size -> corners times the image size). Detections with a score of at least
the variant's `THRESHOLDS` entry (mid-gap) are kept, sorted by score; stored as in tools/golden/dfine.py.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/deim.py DEIM_SRC WEIGHTS_DIR COFFEE_PNG SIZE  # n s m l x

One size per process (DEIM's config registry keeps the first model's settings). DEIM_SRC is a checkout of the DEIM
repository at `COMMIT` (its imports need tensorboard and faster-coco-eval);
WEIGHTS_DIR holds `deim_dfine_hgnetv2_<size>_coco.pth` (the card's weight files).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from _common import goldens, write
from PIL import Image

UPSTREAM = "https://github.com/Intellindust-AI-Lab/DEIM"
COMMIT = "09d35d53d39ee3145a1e61e3a989b28b9468d1dd"
IMAGE_SHA256 = "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7"  # skimage 0.26 data/coffee.png
THRESHOLDS = {"n": 0.5, "s": 0.44, "m": 0.58, "l": 0.5, "x": 0.55}  # per variant: mid-gap of the scores around the cut


def main() -> None:
    src, weights, png = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's coffee.png"
        raise SystemExit(msg)
    sys.path.insert(0, str(src))
    from engine.core import YAMLConfig  # noqa: PLC0415

    img = Image.open(png).convert("RGB")
    x = torch.from_numpy(np.asarray(img.resize((640, 640), Image.Resampling.BILINEAR)).copy()).permute(2, 0, 1)[None]
    for key in sys.argv[4:5]:  # one size per process: DEIM's config registry keeps the first model's settings
        ref = f"deim-{key}-coco"
        cfg = YAMLConfig(str(src / f"configs/deim_dfine/deim_hgnetv2_{key}_coco.yml"))
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        model = cfg.model
        model.load_state_dict(torch.load(weights / f"deim_dfine_hgnetv2_{key}_coco.pth", map_location="cpu")["model"])
        model = model.deploy().eval()
        with torch.inference_mode():
            out = model(x.float() / 255)
        scores, idx = out["pred_logits"].sigmoid()[0].flatten().topk(300)
        labels, query = idx % 80, idx // 80
        cx, cy, bw, bh = out["pred_boxes"][0][query].unbind(-1)
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
            upstream=f"{UPSTREAM}@{COMMIT}",
            patch=None,
            settings=f"CPU float32, DEIM deploy model + RT-DETR postprocess, 640x640 PIL bilinear, skimage coffee.png, "
            f"score threshold {threshold}, torch {torch.__version__}",
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
