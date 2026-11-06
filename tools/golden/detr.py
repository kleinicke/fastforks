"""DETR goldens from the reference implementation.

Reference: upstream facebookresearch/detr (`REF_COMMIT`), CPU float32, the official evaluation path (`hubconf`
network, `RandomResize([800], max_size=1333)` + ImageNet normalisation, `PostProcess`) on scikit-image's `coffee.png`
(CC0, Rachel Michetti; `IMAGE_SHA256`). The unused COCO ids are dropped and the others mapped to the 80 contiguous
`coco-80` labels (the library's output convention). The score threshold `SCORE_THRESHOLD` (in golden.toml as the
ref's `options`, which the test loads with) sits clear of every score so fp16 GPU runs keep the same set. Stored: the
input image (`input`, RGB), the kept detections' boxes (input pixels), scores and labels, as float32. Run this with a
Python that has the upstream dependencies (torch, torchvision, scipy, pycocotools, PIL).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/detr.py DETR_CHECKOUT WEIGHTS_DIR COFFEE_PNG [r50 r101]
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from _common import goldens, reference_tree, write
from PIL import Image

UPSTREAM = "https://github.com/facebookresearch/detr"
REF_COMMIT = "29901c51d7fe8712168b8d0d64351170bc0f83e0"
IMAGE_SHA256 = "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7"  # skimage 0.26 data/coffee.png
SCORE_THRESHOLD = 0.5
UNUSED_IDS = {0, 12, 26, 29, 30, 45, 66, 68, 69, 71, 83}  # COCO ids without a class (and the padding id 0)
USED_IDS = [i for i in range(91) if i not in UNUSED_IDS]
VARIANTS = {  # key: (ref, hubconf function, checkpoint)
    "r50": ("detr-r50-coco", "detr_resnet50", "detr-r50-e632da11.pth"),
    "r101": ("detr-r101-coco", "detr_resnet101", "detr-r101-2c7b67e5.pth"),
}


def main() -> None:
    repo, weights, png = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's coffee.png"
        raise SystemExit(msg)
    img = Image.open(png).convert("RGB")
    tree = reference_tree(repo, REF_COMMIT)
    sys.path.insert(0, str(tree))
    import hubconf  # noqa: PLC0415
    from datasets import transforms as T  # noqa: PLC0415
    from models.detr import PostProcess  # noqa: PLC0415

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    x = transform(img, None)[0][None]
    for key in sys.argv[4:] or VARIANTS:
        ref, fn, checkpoint = VARIANTS[key]
        model = getattr(hubconf, fn)(pretrained=False)
        model.load_state_dict(torch.load(weights / checkpoint, map_location="cpu", weights_only=True)["model"])
        model.eval()
        with torch.inference_mode():
            res = PostProcess()(model(x), torch.tensor([[img.height, img.width]]))[0]
        ids = res["labels"]
        keep = torch.isin(ids, torch.tensor(USED_IDS)) & (res["scores"] >= SCORE_THRESHOLD)
        order = res["scores"][keep].argsort(descending=True)
        label = torch.tensor([USED_IDS.index(i) for i in ids[keep][order].tolist()])
        maps = {
            "input": np.asarray(img),
            "boxes": res["boxes"][keep][order].numpy(),
            "scores": res["scores"][keep][order].numpy(),
            "labels": label.numpy(),
        }
        write(
            f"{ref}@1",
            {k: v.astype(np.float32) for k, v in maps.items()},
            family="detr",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, official eval transform + PostProcess, skimage coffee.png, score threshold "
            f"{SCORE_THRESHOLD}, torch {torch.__version__}",
            task="detection",
        )
        store = goldens.load("detr", "detection")
        store["refs"][f"{ref}@1"]["options"] = [["score_threshold", SCORE_THRESHOLD]]  # the test loads with these
        goldens.dump("detr", store, "detection")
        all_scores = np.sort(res["scores"][torch.isin(ids, torch.tensor(USED_IDS))].numpy())[::-1][:12]
        print(f"  {ref}: {len(label)} kept, top scores {np.round(all_scores, 3)}, labels {maps['labels']}")


if __name__ == "__main__":
    main()
