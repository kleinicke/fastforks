"""DEFOM-Stereo goldens from the reference implementation.

Reference: upstream Insta360-Research-Team/DEFOM-Stereo (`REF_COMMIT`), ViT-L SceneFlow checkpoint, CPU float32,
32 iterations of which 8 scale updates, upstream `InputPadder(divis_by=32)` (demo.py defaults). Upstream imports timm,
opt_einsum, scipy, cv2 and torchvision; run this with a Python that has them.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/defom_stereo.py DEFOM_CHECKOUT WEIGHTS_DIR [vitl-sceneflow ...]

`WEIGHTS_DIR` holds upstream's `defomstereo_<encoder>_<data>.pth` files. The checkout only has to contain `REF_COMMIT`;
the reference runs from an export of that commit, in an empty working directory (upstream loads
`./checkpoints/depth_anything_v2_*.pth` from the working directory when it exists; the checkpoint holds those weights).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
from _common import pair, reference_tree, write

UPSTREAM = "https://github.com/Insta360-Research-Team/DEFOM-Stereo"
REF_COMMIT = "5b27591d83956e5d329ff66704271a9e68b1c75f"
VARIANTS = {"vitl-sceneflow": "vitl"}  # ref name -> DINOv2 encoder of the checkpoint `defomstereo_<enc>_sceneflow.pth`
ITERS, SCALE_ITERS = 32, 8


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    tree = reference_tree(repo, REF_COMMIT)
    sys.path[:0] = [str(tree / "core"), str(tree)]  # demo.py runs with core/ importable
    # upstream utils.py imports tensorboard for training logging only; a stub keeps the reference import-clean
    stub = types.ModuleType("torch.utils.tensorboard")
    stub.SummaryWriter = object  # stub module
    sys.modules["torch.utils.tensorboard"] = stub
    os.chdir(tempfile.mkdtemp(prefix="ff-defom-cwd-"))
    from core.defom_stereo import DEFOMStereo  # noqa: PLC0415  # reference checkout on sys.path
    from utils.utils import InputPadder  # noqa: PLC0415  # reference checkout on sys.path

    left, right = (torch.from_numpy(x).permute(2, 0, 1)[None].float() for x in pair())
    padder = InputPadder(left.shape, divis_by=32)
    lp, rp = padder.pad(left, right)
    for name in sys.argv[3:] or VARIANTS:
        enc = VARIANTS[name]
        args = argparse.Namespace(
            dinov2_encoder=enc, idepth_scale=0.5, hidden_dims=[128] * 3, corr_levels=2, corr_radius=4,
            scale_list=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0], scale_corr_radius=2, n_downsample=2,
            context_norm="batch", n_gru_layers=3, mixed_precision=False, shared_backbone=False,
        )  # fmt: skip
        net = DEFOMStereo(args)
        ckpt = torch.load(weights / f"defomstereo_{enc}_sceneflow.pth", map_location="cpu", weights_only=True)
        net.load_state_dict(ckpt)
        with torch.inference_mode():
            disp = net.eval()(lp, rp, iters=ITERS, scale_iters=SCALE_ITERS, test_mode=True)
        disp = padder.unpad(disp)[0, 0].numpy()
        write(
            f"defom-stereo-{name}@1",
            {"disparity": disp},
            family="defom_stereo",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, {ITERS} iters ({SCALE_ITERS} scale), torch {torch.__version__}",
        )
        print(f"  median disparity {np.median(disp):.4f}")


if __name__ == "__main__":
    main()
