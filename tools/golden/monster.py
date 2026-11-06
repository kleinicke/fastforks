"""MonSter goldens from the reference implementation.

Reference: upstream Junda24/MonSter (`REF_COMMIT`), CPU float32, 32 iterations, test mode, upstream
`InputPadder(divis_by=32)` (evaluate_stereo.py defaults). Upstream imports matplotlib, opt_einsum, timm and cv2; run
this with a Python that has them.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/monster.py MONSTER_CHECKOUT WEIGHTS_DIR [sceneflow mix-all]

`WEIGHTS_DIR` holds upstream's `sceneflow.pth` / `mix_all.pth`. The checkout only has to contain `REF_COMMIT`; the
reference runs from an export of that commit, in an empty working directory. Upstream's constructor loads
`./pretrained/depth_anything_v2_<encoder>.pth` (the Depth Anything V2 weights, all overwritten by the strict
checkpoint load); that one load returns the state dict of a freshly built Depth Anything V2 instead.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from _common import pair, reference_tree, write

UPSTREAM = "https://github.com/Junda24/MonSter"
REF_COMMIT = "49016c9a72b4260c0d119c77bb14e78a132cea8c"
VARIANTS = {"sceneflow": "sceneflow.pth", "mix-all": "mix_all.pth"}  # ref name -> checkpoint file
ITERS = 32


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    tree = reference_tree(repo, REF_COMMIT)
    here = Path(__file__).resolve().parent  # its depth_anything_v2.py would shadow upstream's namespace package
    sys.path[:] = [str(tree / "Depth-Anything-V2-list3"), str(tree / "core"), str(tree)] + [
        p for p in sys.path if Path(p or ".").resolve() != here
    ]
    os.chdir(tempfile.mkdtemp(prefix="ff-monster-cwd-"))
    from core.monster import Monster  # noqa: PLC0415  # reference checkout on sys.path
    from depth_anything_v2.dpt import DepthAnythingV2  # noqa: PLC0415 - reference checkout on sys.path
    from utils.utils import InputPadder  # noqa: PLC0415  # reference checkout on sys.path

    cfg = {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}  # upstream's vitl config
    load = torch.load

    def load_pretrained(f, *a, **k):  # noqa: ANN001, ANN002, ANN003, ANN202 - torch.load signature
        if str(f).startswith("./pretrained/depth_anything_v2_"):
            return DepthAnythingV2(**cfg).state_dict()
        return load(f, *a, **k)

    left, right = (torch.from_numpy(x).permute(2, 0, 1)[None].float() for x in pair())
    padder = InputPadder(left.shape, divis_by=32)
    lp, rp = padder.pad(left, right)
    for name in sys.argv[3:] or VARIANTS:
        args = argparse.Namespace(
            encoder="vitl", hidden_dims=[128] * 3, corr_implementation="reg", shared_backbone=False, corr_levels=2,
            corr_radius=4, n_downsample=2, slow_fast_gru=False, n_gru_layers=3, max_disp=192, mixed_precision=False,
        )  # fmt: skip
        torch.load = load_pretrained
        try:
            net = Monster(args)
        finally:
            torch.load = load
        ckpt = torch.load(weights / VARIANTS[name], map_location="cpu", weights_only=True)
        net.load_state_dict({k.removeprefix("module."): v for k, v in ckpt.items()}, strict=True)
        with torch.inference_mode():
            disp = net.eval()(lp, rp, iters=ITERS, test_mode=True)
        disp = padder.unpad(disp)[0, 0].numpy()
        write(
            f"monster-{name}@1",
            {"disparity": disp},
            family="monster",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, {ITERS} iters, torch {torch.__version__}",
        )
        print(f"  median disparity {np.median(disp):.4f}")


if __name__ == "__main__":
    main()
