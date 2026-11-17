"""RF-DETR goldens from the pinned original network and PostProcess.

Usage: FASTFORKS_GOLDEN_PATH=... python tools/golden/rfdetr.py RF_DETR_CHECKOUT COFFEE_PNG [variant ids]
The original package's optional dependencies must be installed in this reference environment. The PNG is scikit-image's
`coffee.png` (CC0, `IMAGE_SHA256`). Resize/normalization use upstream's torchvision functions as RFDETR.predict does
(square, no antialiasing); PostProcess keeps the `num_select` best (query, class) pairs. Detections above a per-variant
threshold are kept (the ref's `options` in golden.toml): the largest set of best detections whose scores, and the cut
below them, are at least `MARGIN` apart, so fp16 GPU runs keep the same detections in the same order. The first view of
the image (`VIEWS`: whole, top three quarters) with such a set is used. Stored: the input image (`input`, RGB), boxes
(input pixels), scores, labels (`coco-80`, unused COCO ids dropped) and, for Seg, the masks at the input size.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from _common import goldens, reference_tree, write
from PIL import Image
from torchvision.transforms import functional as TF

from fastforks._core import registry
from fastforks.detection.models.rfdetr.model import class_map

FAMILY = "rfdetr"
COMMIT = "5b22dfc0940697ce1abdfa681a5f7392b3eddbfb"
UPSTREAM = "https://github.com/roboflow/rf-detr"
IMAGE_SHA256 = "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7"  # skimage 0.26 data/coffee.png
MARGIN = 0.03  # min score gap between kept detections and at the cut (fp16 GPU score error <= 0.02)
FLOOR = 0.3  # kept detections score above this
VIEWS = {"whole": (slice(None), slice(None)), "top": (slice(0, 300), slice(None))}  # of 400 x 600
CLASSES = {  # variant id -> upstream class
    "rf-detr-n-coco": "RFDETRNano",
    "rf-detr-s-coco": "RFDETRSmall",
    "rf-detr-m-coco": "RFDETRMedium",
    "rf-detr-l-coco": "RFDETRLarge",
    "rf-detr-seg-n-coco": "RFDETRSegNano",
    "rf-detr-seg-s-coco": "RFDETRSegSmall",
    "rf-detr-seg-m-coco": "RFDETRSegMedium",
    "rf-detr-seg-l-coco": "RFDETRSegLarge",
    "rf-detr-seg-xl-coco": "RFDETRSegXLarge",
    "rf-detr-seg-2xl-coco": "RFDETRSeg2XLarge",
}


def reference(checkout: Path, name: str, checkpoint: Path, device: str = "cpu") -> Any:  # noqa: ANN401 - upstream wrapper
    """Build the original package at the pinned commit with the author's checkpoint."""
    tree = reference_tree(checkout, COMMIT)
    if str(tree / "src") not in sys.path:
        sys.path.insert(0, str(tree / "src"))
    import rfdetr  # noqa: PLC0415 - isolated pinned source

    model = getattr(rfdetr, name)(pretrain_weights=str(checkpoint), device=device)
    model.model.model.to(device).eval()  # predict() switches to eval before running
    return model


def raw(model: Any, image: torch.Tensor) -> dict[str, torch.Tensor]:  # noqa: ANN401 - upstream wrapper
    """The original predict tensor path: square resize without antialiasing, normalization, the network."""
    size = model.model.resolution
    x = TF.normalize(TF.resize(image, [size, size], antialias=False), model.means, model.stds)
    net = model.model.inference_model
    if net is None:
        net = model.model.model
    out = net(x.to(next(net.parameters()).dtype))
    if isinstance(out, tuple):  # the public inference(dtype=...) path exports (boxes, logits[, masks])
        out = dict(zip(("pred_boxes", "pred_logits", "pred_masks"), out, strict=False))
    return {k: v.float() for k, v in out.items() if k in {"pred_boxes", "pred_logits", "pred_masks"}}


def predict(model: Any, image: torch.Tensor, threshold: float) -> dict[str, torch.Tensor]:  # noqa: ANN401 - upstream wrapper
    """Original PostProcess above `threshold`, labels mapped to `coco-80` (unused COCO ids dropped)."""
    target = torch.tensor([image.shape[-2:]], device=image.device)
    r = model.model.postprocess(raw(model, image), target_sizes=target, score_threshold=threshold)[0]
    labels = class_map(91).to(r["labels"].device)[r["labels"]]  # 90 COCO ids + id 0
    keep = (labels >= 0) & (r["scores"] > threshold)
    out = {"boxes": r["boxes"][keep], "scores": r["scores"][keep], "labels": labels[keep]}
    if "masks" in r:
        out["masks"] = r["masks"][keep][:, 0]
    return out


def gap_threshold(scores: np.ndarray) -> float | None:
    """Mid-gap threshold under the largest best-first set (scores > `FLOOR`) separated by `MARGIN`; None if none."""
    s = np.sort(scores)[::-1]
    best = None
    for k in range(1, int((s > FLOOR).sum()) + 1):
        if np.diff(s[: k + 1]).max() > -MARGIN:  # a near tie inside the set or at the cut
            continue
        best = round(float(s[k - 1] + s[k]) / 2, 4)
    return best


def main() -> None:
    """Write the original CPU float32 results and their pinned provenance."""
    checkout, png = map(Path, sys.argv[1:3])
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's coffee.png"
        raise SystemExit(msg)
    whole = np.asarray(Image.open(png).convert("RGB"))
    for ref in sys.argv[3:] or CLASSES:
        card, variant = registry.cards("detection")[ref]
        (checkpoint,) = registry.weights(card, variant)
        model = reference(checkout, CLASSES[ref], checkpoint)
        for view in VIEWS:
            image = np.ascontiguousarray(whole[VIEWS[view]])
            x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255
            with torch.inference_mode():
                r = model.model.postprocess(raw(model, x), target_sizes=torch.tensor([image.shape[:2]]))[0]
            threshold = gap_threshold(r["scores"][class_map(91)[r["labels"]] >= 0].numpy())
            if threshold is not None:
                break
        else:
            msg = f"{ref}: no view with detections {MARGIN} apart"
            raise SystemExit(msg)
        with torch.inference_mode():
            out = predict(model, x, threshold)
        maps = {"input": image, **{k: v.float().numpy() for k, v in out.items()}}
        write(
            variant.ref,
            {k: v.astype(np.float32) for k, v in maps.items()},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{COMMIT}",
            patch=None,
            settings=f"CPU float32 eval mode, original RF-DETR network and PostProcess; torchvision resize "
            f"antialias=False, ImageNet normalization; skimage coffee.png ({view}), score threshold {threshold}; "
            f"torch {torch.__version__}",
            task="detection",
        )
        store = goldens.load(FAMILY, "detection")
        store["refs"][variant.ref]["options"] = [["score_threshold", threshold]]  # the test loads with these
        goldens.dump(FAMILY, store, "detection")
        print(f"  {ref}: {view}, {len(out['scores'])} kept at {threshold}: {out['scores'].tolist()}")


if __name__ == "__main__":
    main()
