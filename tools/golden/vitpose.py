"""ViTPose / ViTPose++ goldens from the reference implementation.

Reference: HF transformers `VitPoseForPoseEstimation` + `VitPoseImageProcessor` (`REF_VERSION`), CPU float32, on
scikit-image's `astronaut.png` (public domain, NASA; `IMAGE_SHA256`) with the person boxes `BOXES` (the second one
reaches past the image, so the crop's zero border is checked too). ViTPose++ runs with `dataset_index = 0` (COCO).
The image goes to the processor as float [0, 1] with `do_rescale=False`: its scipy crop then stays float, as the
library's (with uint8 input it would round the crop). Stored: the input image (`input`, RGB), the boxes (`boxes`,
`x0, y0, x1, y1`; the processor takes `x, y, w, h`), the keypoints (input pixels), their scores, and `scores` (the
mean keypoint score per box, the library's person score; the reference gives none), as float32.

    FASTFORKS_GOLDEN_PATH=... uv run --with transformers==4.57.1 --with scipy --with scikit-image \
        python tools/golden/vitpose.py [vitpose-base-simple ...]
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import skimage
import torch
import transformers
from _common import write
from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor

UPSTREAM = "https://github.com/huggingface/transformers"
REF_VERSION = "4.57.1"
IMAGE_SHA256 = "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5"  # skimage data/astronaut.png
BOXES = np.array([[120.0, 20.0, 440.0, 511.0], [250.0, 150.0, 600.0, 560.0]], dtype=np.float32)
REFS = {  # ref: (hub repo, revision)
    "vitpose-base-simple": ("usyd-community/vitpose-base-simple", "a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45"),
    "vitpose-base": ("usyd-community/vitpose-base", "95be2991424e646950d656bb7fc15ec9be119700"),
    "vitpose-plus-small": ("usyd-community/vitpose-plus-small", "0c30b6534bb621af0162b481176742577264e36e"),
    "vitpose-plus-base": ("usyd-community/vitpose-plus-base", "92be54d7a29e42fad47b6e2ca01dd9e685a61e0d"),
    "vitpose-plus-large": ("usyd-community/vitpose-plus-large", "e211df377e49c89dba508ab3e83ddef13c0832b4"),
    "vitpose-plus-huge": ("usyd-community/vitpose-plus-huge", "9f36d7aec1800d23e97f10c2e74393aee92aa53f"),
}


def main() -> None:
    if transformers.__version__ != REF_VERSION:
        msg = f"transformers {transformers.__version__}, the goldens pin {REF_VERSION}"
        raise SystemExit(msg)
    png = Path(skimage.data.__file__).parent / "astronaut.png"
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's astronaut.png"
        raise SystemExit(msg)
    img = skimage.io.imread(png)[..., :3]
    xywh = np.concatenate([BOXES[:, :2], BOXES[:, 2:] - BOXES[:, :2]], 1)
    for ref in sys.argv[1:] or REFS:
        repo, rev = REFS[ref]
        proc = VitPoseImageProcessor.from_pretrained(repo, revision=rev)
        model = VitPoseForPoseEstimation.from_pretrained(repo, revision=rev).eval()
        x = proc(img.astype(np.float32) / 255, boxes=[xywh.tolist()], do_rescale=False, return_tensors="pt")
        kw = (
            {"dataset_index": torch.zeros(len(BOXES), dtype=torch.long)}
            if model.config.backbone_config.num_experts > 1
            else {}
        )
        with torch.inference_mode():
            out = model(x["pixel_values"], **kw)
        res = proc.post_process_pose_estimation(out, boxes=[xywh.tolist()])[0]
        kp = np.stack([r["keypoints"].numpy() for r in res])
        ks = np.stack([r["scores"].numpy() for r in res])
        maps = {"input": img, "boxes": BOXES, "keypoints": kp, "keypoint_scores": ks, "scores": ks.mean(1)}
        write(
            f"{ref}@1",
            {k: np.asarray(v, np.float32) for k, v in maps.items()},
            family="vitpose",
            upstream=f"{UPSTREAM}@v{REF_VERSION}",
            patch=None,
            settings="CPU float32, VitPoseImageProcessor (float input, do_rescale=False) + "
            f"post_process_pose_estimation, dataset_index 0, skimage astronaut.png, 2 boxes, torch {torch.__version__}",
            task="pose",
        )
        print(f"  {ref}: scores {np.round(ks.mean(1), 3)}")


if __name__ == "__main__":
    main()
