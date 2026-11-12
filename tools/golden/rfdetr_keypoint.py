"""RF-DETR Keypoint goldens from the pinned original network and PostProcess.

Usage: python tools/golden/rfdetr_keypoint.py RF_DETR_CHECKOUT ASTRONAUT_PNG
The original package's optional dependencies must be installed in this reference environment. The PNG is the
public-domain NASA astronaut image distributed by scikit-image, hash-checked below. Resize/normalization use
upstream's torchvision functions, with the same defaults as RFDETR.predict (no antialiasing, threshold 0.5).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from _common import reference_tree, write
from PIL import Image
from torchvision.transforms import functional as TF

from fastforks._core import registry

FAMILY = "rfdetr_keypoint"
COMMIT = "5b22dfc0940697ce1abdfa681a5f7392b3eddbfb"
UPSTREAM = "https://github.com/roboflow/rf-detr"
IMAGE_SHA256 = "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5"


def reference(checkout: Path, checkpoint: Path, device: str = "cpu") -> Any:  # noqa: ANN401 - upstream wrapper
    """Build the original package at the pinned commit with the author's checkpoint."""
    tree = reference_tree(checkout, COMMIT)
    sys.path.insert(0, str(tree / "src"))
    from rfdetr import RFDETRKeypointPreview  # noqa: PLC0415 - isolated pinned source

    model = RFDETRKeypointPreview(pretrain_weights=str(checkpoint), device=device, num_classes=1)
    # The public predict() switches to eval before running. The constructor leaves 13 training query groups active.
    model.model.model.to(device).eval()
    return model


def predict(model: Any, image: torch.Tensor, threshold: float = 0.5) -> dict[str, torch.Tensor]:  # noqa: ANN401 - upstream wrapper
    """The original predict tensor path and PostProcess, retaining its COCO person class."""
    size = model.model.resolution
    x = TF.normalize(TF.resize(image, [size, size], antialias=False), model.means, model.stds)
    net = model.model.inference_model
    if net is None:
        net = model.model.model
    dtype = next(net.parameters()).dtype
    raw = net(x.to(dtype))
    if isinstance(raw, tuple):  # the public inference(dtype=...) path exports these same three tensors
        raw = dict(zip(("pred_boxes", "pred_logits", "pred_keypoints"), raw, strict=True))
    target = torch.tensor([image.shape[-2:]], device=image.device)
    result = model.model.postprocess(raw, target_sizes=target)[0]
    keep = (result["labels"] == 1) & (result["scores"] > threshold)
    points = result["keypoints"][keep]
    return {
        "boxes": result["boxes"][keep],
        "scores": result["scores"][keep],
        "keypoints": points[..., :2],
        "keypoint_scores": points[..., 2],
    }


def main() -> None:
    """Write the original CPU float32 result and its pinned provenance."""
    checkout, png = map(Path, sys.argv[1:3])
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = "expected scikit-image's public-domain astronaut.png"
        raise SystemExit(msg)
    image = np.asarray(Image.open(png).convert("RGB"))
    card, variant = registry.cards("pose")["rf-detr-keypoint-preview"]
    (checkpoint,) = registry.weights(card, variant)
    model = reference(checkout, checkpoint)
    x = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
    with torch.inference_mode():
        out = predict(model, x)
    maps = {"input": image, **{k: v.cpu().numpy() for k, v in out.items()}}
    write(
        variant.ref,
        maps,
        family=FAMILY,
        upstream=f"{UPSTREAM}@{COMMIT}",
        patch=None,
        settings="CPU float32 eval mode, original RF-DETR network and PostProcess; "
        "torchvision resize antialias=False, "
        "ImageNet normalization, uncertainty trace_alpha=0.2, person threshold=0.5; "
        f"NASA astronaut.png; torch {torch.__version__}",
        task="pose",
    )


if __name__ == "__main__":
    main()
