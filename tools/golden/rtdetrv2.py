"""RT-DETRv2 goldens from the reference implementation.

Reference: upstream lyuwenyu/RT-DETR (`REF_COMMIT`, `rtdetrv2_pytorch`), CPU float32, the official deploy path
(`references/deploy/rtdetrv2_torch.py`: `YAMLConfig` model in deploy mode + `RTDETRPostProcessor`, image resized to
640 x 640 with PIL bilinear) on scikit-image's `coffee.png` (CC0, Rachel Michetti; `IMAGE_SHA256`). Detections with a
score of at least the variant's `THRESHOLDS` entry are kept, sorted by score (the ref's `options` in golden.toml, which
the test loads with). The threshold sits mid-gap, clear of the score gaps so fp16 GPU runs keep the same set. Stored:
the input image (`input`, RGB), boxes (input pixels), scores and labels (`coco-80`), as float32. Run this with a
Python that
has the upstream dependencies (torch, torchvision, pyyaml, scipy, pycocotools, PIL).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/rtdetrv2.py RT_DETR_CHECKOUT WEIGHTS_DIR COFFEE_PNG [s m ms l x]
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
from _common import goldens, reference_tree, write
from PIL import Image

UPSTREAM = "https://github.com/lyuwenyu/RT-DETR"
REF_COMMIT = "df2a96c8c3a098567a1149bf0248da6efe6d495d"
IMAGE_SHA256 = "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7"  # skimage 0.26 data/coffee.png
THRESHOLDS = {"s": 0.67, "m": 0.68, "ms": 0.6, "l": 0.45, "x": 0.5}  # per variant: mid-gap of the scores around the cut
VARIANTS = {  # key: (ref, config, checkpoint)
    "s": ("rtdetrv2-s-coco", "rtdetrv2_r18vd_120e_coco.yml", "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"),
    "m": ("rtdetrv2-m-coco", "rtdetrv2_r34vd_120e_coco.yml", "rtdetrv2_r34vd_120e_coco_ema.pth"),
    "ms": ("rtdetrv2-ms-coco", "rtdetrv2_r50vd_m_7x_coco.yml", "rtdetrv2_r50vd_m_7x_coco_ema.pth"),
    "l": ("rtdetrv2-l-coco", "rtdetrv2_r50vd_6x_coco.yml", "rtdetrv2_r50vd_6x_coco_ema.pth"),
    "x": ("rtdetrv2-x-coco", "rtdetrv2_r101vd_6x_coco.yml", "rtdetrv2_r101vd_6x_coco_from_paddle.pth"),
}


def main() -> None:
    repo, weights, png = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's coffee.png"
        raise SystemExit(msg)
    img = Image.open(png).convert("RGB")
    tree = reference_tree(repo, REF_COMMIT) / "rtdetrv2_pytorch"
    sys.path.insert(0, str(tree))
    for name in (
        "torch.utils.tensorboard",
        "faster_coco_eval",
        "faster_coco_eval.utils",
        "faster_coco_eval.utils.pytorch",
        "faster_coco_eval.core",
        "faster_coco_eval.core.mask",
    ):
        sys.modules[name] = MagicMock()  # training / evaluation imports of the config package, unused here
    for base in ("FasterCocoDetection", "FasterCocoEvaluator"):
        setattr(sys.modules["faster_coco_eval.utils.pytorch"], base, type(base, (), {}))
    from src.core import YAMLConfig  # noqa: PLC0415

    x = torch.from_numpy(np.asarray(img.resize((640, 640), Image.BILINEAR)).copy()).permute(2, 0, 1)[None] / 255
    for key in sys.argv[4:] or VARIANTS:
        ref, config, checkpoint = VARIANTS[key]
        cfg = YAMLConfig(str(tree / "configs" / "rtdetrv2" / config))
        ckpt = torch.load(weights / checkpoint, map_location="cpu", weights_only=True)
        cfg.model.load_state_dict(ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"])
        model, post = cfg.model.deploy().eval(), cfg.postprocessor.deploy()
        with torch.inference_mode():
            labels, boxes, scores = post(model(x), torch.tensor([[img.width, img.height]]))
        threshold = THRESHOLDS[key]
        keep = scores[0] >= threshold
        maps = {
            "input": np.asarray(img),
            "boxes": boxes[0][keep].numpy(),
            "scores": scores[0][keep].numpy(),
            "labels": labels[0][keep].numpy(),
        }
        write(
            f"{ref}@1",
            {k: v.astype(np.float32) for k, v in maps.items()},
            family="rtdetrv2",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, official deploy model + postprocessor, 640x640 PIL bilinear, skimage coffee.png, "
            f"score threshold {threshold}, torch {torch.__version__}",
            task="detection",
        )
        store = goldens.load("rtdetrv2", "detection")
        store["refs"][f"{ref}@1"]["options"] = [["score_threshold", threshold]]  # the test loads with these
        goldens.dump("rtdetrv2", store, "detection")
        near = np.sort(scores[0].numpy())[::-1]
        n = int(keep.sum())
        print(f"  {ref}: {n} kept, scores around the cut {np.round(near[max(0, n - 2) : n + 3], 3)}")


if __name__ == "__main__":
    main()
