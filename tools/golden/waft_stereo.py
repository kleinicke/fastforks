"""WAFT-Stereo goldens from the reference implementation.

Reference: upstream princeton-vl/WAFT-Stereo (`REF_COMMIT`), the SynLarge DAv2-S / DAv2-B checkpoints, CPU float32,
upstream `WAFT.inference(sample, size=None)` (the path of its evaluation and demo scripts; it pads internally). Upstream
imports timm, peft, einops, torchvision and cv2; run this with a Python that has them.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/waft_stereo.py WAFT_CHECKOUT WEIGHTS_DIR [s b ...]

`WEIGHTS_DIR` holds upstream's `SynLarge/DAv2<S|B>-4.pth` files. The checkout only has to contain `REF_COMMIT`. The
iterative modules call `timm.create_model(..., pretrained=True)` for an ImageNet initialisation that the checkpoint
overwrites; the reference builds them without the download (`pretrained=False`).
"""

from __future__ import annotations

import functools
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from _common import pair, reference_tree, write

UPSTREAM = "https://github.com/princeton-vl/WAFT-Stereo"
REF_COMMIT = "c534320860f4bbb2c0df896e3ccb95de49284aec"
VARIANTS = {"s": "DAv2S-4", "b": "DAv2B-4"}  # ref suffix -> checkpoint / config name under SynLarge/


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    tree = reference_tree(repo, REF_COMMIT)
    sys.path[:0] = [str(tree)]
    os.chdir(
        tempfile.mkdtemp(prefix="ff-waft-cwd-")
    )  # no depth-anything-ckpts/ here: the checkpoint holds the DAv2 weights
    import timm  # noqa: PLC0415 - the reference's own dependency

    vars(timm)["create_model"] = functools.partial(timm.create_model, pretrained=False)  # no ImageNet download
    from algorithms.waft import WAFT  # noqa: PLC0415  # reference checkout on sys.path
    from bridgedepth.config import get_cfg  # noqa: PLC0415  # reference checkout on sys.path

    left, right = (torch.from_numpy(x).permute(2, 0, 1)[None].float() for x in pair())
    for suffix in sys.argv[3:] or VARIANTS:
        name = VARIANTS[suffix]
        cfg = get_cfg()
        cfg.merge_from_file(str(tree / "configs" / "SynLarge" / f"{name}.yaml"))
        cfg.freeze()
        net = WAFT(cfg)
        ckpt = torch.load(weights / "SynLarge" / f"{name}.pth", map_location="cpu", weights_only=False)
        missing = net.load_state_dict(ckpt["model"], strict=False)
        if missing.missing_keys or missing.unexpected_keys:  # nothing silently skipped
            raise RuntimeError(missing)
        with torch.inference_mode():
            disp = net.eval().inference({"img1": left, "img2": right}, size=None)["disp_pred"]
        disp = disp[0].numpy()
        write(
            f"waft-stereo-{suffix}@1",
            {"disparity": disp},
            family="waft_stereo",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, 3 delta iterations, inference(size=None), torch {torch.__version__}",
        )
        print(f"  median disparity {np.median(disp):.4f}")


if __name__ == "__main__":
    main()
