"""Fast-ACVNet goldens from the reference implementation.

Reference: upstream gangweix/Fast-ACVNet (`REF_COMMIT`), CPU float32, max disparity 192, test mode, upstream test
preprocessing: ImageNet normalization, then zeros on top and on the right up to a multiple of 32
(`datasets/kitti_dataset_1215.py`). Upstream builds its backbone with timm (`mobilenetv2_100`); run this with a Python
that has timm. timm is shimmed: no ImageNet download (the checkpoints hold the
backbone), and an identity stem `act1` for timm >= 0.6 (it folds the ReLU6 into `bn1`; same math).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/fast_acvnet.py FAST_ACVNET_CHECKOUT WEIGHTS_DIR [VARIANT_ID ...]

`WEIGHTS_DIR` holds upstream's two Drive folders (`Fast-ACVNet/<name>.ckpt`, `Fast-ACVNet+/<name>.ckpt`). The checkout
only has to contain `REF_COMMIT`; the reference runs from an export of that commit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import timm  # reference environment only
import torch
import torch.nn.functional as F
from _common import pair, reference_tree, write

UPSTREAM = "https://github.com/gangweix/Fast-ACVNet"
REF_COMMIT = "61d5375e1816f124f4417cea506eeaddac1504e4"
MAX_DISP = 192
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
FILES = {"generalization": "generalization", "sceneflow": "sceneflow", "kitti12": "kitti_2012", "kitti15": "kitti_2015"}
VARIANTS = tuple(f"fast-acvnet{p}-{v}" for p in ("", "-plus") for v in FILES)


def _patch_timm() -> None:
    create = timm.create_model

    def create_model(*a: object, **kw: object) -> torch.nn.Module:
        model = create(*a, **{**kw, "pretrained": False})
        if not hasattr(model, "act1"):
            model.act1 = torch.nn.Identity()
        return model

    timm.create_model = create_model


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT)))
    _patch_timm()
    from models import __models__  # noqa: PLC0415  # reference checkout on sys.path

    mean, std = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)
    left, right = ((torch.from_numpy(x).permute(2, 0, 1)[None].float() / 255 - mean) / std for x in pair())
    h, w = left.shape[2:]
    pads = (0, (-w) % 32, (-h) % 32, 0)
    lp, rp = F.pad(left, pads), F.pad(right, pads)
    for vid in sys.argv[3:] or VARIANTS:
        plus = vid.startswith("fast-acvnet-plus-")
        net = __models__["Fast_ACVNet_plus" if plus else "Fast_ACVNet"](MAX_DISP, False)  # noqa: FBT003 - upstream signature (attention_weights_only)
        ckpt = weights / ("Fast-ACVNet+" if plus else "Fast-ACVNet") / f"{FILES[vid.rsplit('-', 1)[1]]}.ckpt"
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)["model"]
        net.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()}, strict=True)
        with torch.inference_mode():
            disp = net.eval()(lp, rp)[-1]
        disp = disp[0, pads[2] :, :w].numpy()
        write(
            f"{vid}@1",
            {"disparity": disp},
            family="fast_acvnet",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, max disparity {MAX_DISP}, torch {torch.__version__}",
        )
        print(f"  median disparity {np.median(disp):.4f}")


if __name__ == "__main__":
    main()
