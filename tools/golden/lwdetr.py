"""LW-DETR goldens from the reference implementation.

Reference: Hugging Face transformers `LwDetrForObjectDetection` (`REF_VERSION`; the port of Atten4Vis/LW-DETR that the
`AnnaZhang/lwdetr_*_60e_coco` weights are published for) with its `DeformableDetrImageProcessor` as the weights'
`preprocessor_config.json` sets it (640 x 640, bilinear, ImageNet mean / std), CPU float32, on scikit-image's
`coffee.png` (`IMAGE_SHA256`), and the authors' `PostProcess` (the `num_select` best of the Q x 91 sigmoid
scores, boxes centre / size -> corners times the image size); the COCO category ids are mapped to `coco-80` and the
unused ids dropped. Detections with a score of at least the variant's threshold are kept, sorted by score. The
threshold is `THRESHOLD`, else the middle of the widest gap between the scores in [0.3, 0.7] (stored as the ref's
`options`, which the test loads with), so fp16 GPU runs keep the same set. Stored: the input image (`input`, RGB),
boxes (input pixels), scores and labels.

    FASTFORKS_GOLDEN_PATH=... FASTFORKS_WEIGHTS_PATH=... python tools/golden/lwdetr.py COFFEE_PNG [tiny small ...]
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from _common import goldens, write
from PIL import Image
from safetensors.torch import load_file

from fastforks._core import registry
from fastforks.shared.seginfer import untie_to_checkpoint

UPSTREAM = "https://github.com/huggingface/transformers"
REF_VERSION = transformers.__version__
IMAGE_SHA256 = "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7"  # skimage 0.26 data/coffee.png
UNUSED_IDS = (12, 26, 29, 30, 45, 66, 68, 69, 71, 83)  # COCO category ids without a class
SIZES = ("tiny", "small", "medium", "large", "xlarge")


def coco80(ids: torch.Tensor) -> torch.Tensor:
    """COCO category ids -> `coco-80` labels (-1: id 0 or an unused id)."""
    table, nxt = torch.full((91,), -1, dtype=torch.int64), 0
    for i in range(1, 91):
        if i not in UNUSED_IDS:
            table[i], nxt = nxt, nxt + 1
    return table[ids]


def reference(ref: str) -> tuple[torch.nn.Module, int]:
    """The transformers model of the card's pinned files and the variant's `num_select`."""
    card, variant = registry.resolve("detection", ref)
    cfg_path, weights = registry.weights(card, variant, allow_download=False)
    cfg = transformers.LwDetrConfig.from_json_file(str(cfg_path))
    sd = load_file(str(weights))
    net = transformers.LwDetrForObjectDetection.from_pretrained(None, config=cfg, state_dict=sd)
    untie_to_checkpoint(net, sd)
    return net.float().eval(), int(variant.params["num_select"])


def processor(size: str) -> Any:  # noqa: ANN401 - a transformers image processor
    """The image processor of the weights' `preprocessor_config.json` (from the local Hub cache)."""
    return transformers.AutoImageProcessor.from_pretrained(f"AnnaZhang/lwdetr_{size}_60e_coco", local_files_only=True)


def detect(net: torch.nn.Module, num_select: int, img: np.ndarray, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """The authors' PostProcess on `net(x)` for an `img` of the original size; all `num_select` kept."""
    h, w = img.shape[:2]
    out = net(pixel_values=x)
    logits = out.logits.float()[0]
    scores, idx = logits.sigmoid().flatten().topk(num_select)
    labels, query = coco80(idx.cpu() % logits.shape[-1]).to(idx.device), idx // logits.shape[-1]
    cx, cy, bw, bh = out.pred_boxes.float()[0][query].unbind(-1)
    boxes = torch.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), -1)
    keep = labels >= 0
    return {"boxes": boxes[keep] * boxes.new_tensor([w, h, w, h]), "scores": scores[keep], "labels": labels[keep]}


# tiny: a duplicate query of the cup (float32 0.385, below the widest gap) scores 0.64 with the fp16 kernels, so the
# cut goes above the second detection (float32 0.739; GPU 0.66-0.75) and keeps the one clear detection
THRESHOLD = {"tiny": 0.78}


def mid_gap(scores: np.ndarray) -> float:
    """The middle of the widest gap between the sorted scores inside [0.3, 0.7] (rounded to 3 digits)."""
    s = np.sort(np.concatenate([[0.3, 0.7], scores[(scores > 0.3) & (scores < 0.7)]]))  # noqa: PLR2004
    i = int(np.argmax(np.diff(s)))
    return round(float(s[i] + s[i + 1]) / 2, 3)


def main() -> None:
    png = Path(sys.argv[1])
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's coffee.png"
        raise SystemExit(msg)
    img = np.asarray(Image.open(png).convert("RGB"))
    for size in sys.argv[2:] or SIZES:
        x = processor(size)(images=img, return_tensors="pt")["pixel_values"]
        ref = f"lw-detr-{size}-coco"
        net, num_select = reference(ref)
        with torch.inference_mode():
            det = detect(net, num_select, img, x)
        scores = det["scores"].numpy()
        threshold = THRESHOLD.get(size) or mid_gap(scores)
        keep = scores >= threshold
        maps = {"input": img, **{k: v.numpy()[keep] for k, v in det.items()}}
        write(
            f"{ref}@1",
            {k: v.astype(np.float32) for k, v in maps.items()},
            family="lwdetr",
            upstream=f"{UPSTREAM}@v{REF_VERSION}",
            patch=None,
            settings=f"CPU float32, transformers LwDetrForObjectDetection + DeformableDetrImageProcessor 640x640, "
            f"the authors' PostProcess (num_select {num_select}), skimage coffee.png, score threshold {threshold}, "
            f"torch {torch.__version__}",
            task="detection",
        )
        store = goldens.load("lwdetr", "detection")
        store["refs"][f"{ref}@1"]["options"] = [["score_threshold", threshold]]  # the test loads with these
        goldens.dump("lwdetr", store, "detection")
        near = np.sort(scores)[::-1]
        n = int(keep.sum())
        print(f"  {ref}: {n} kept at {threshold}, scores around the cut {np.round(near[max(0, n - 2) : n + 3], 3)}")


if __name__ == "__main__":
    main()
