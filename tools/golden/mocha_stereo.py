"""MoCha-Stereo goldens from the reference implementation.

Reference: upstream ZYangChen/MoCha-Stereo (`REF_COMMIT`), CPU float32, 32 iterations, test mode, upstream
`InputPadder(divis_by=32)` (evaluate_stereo.py defaults). Upstream imports timm (and cv2 / scipy for its data code);
run this with a Python that has them.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/mocha_stereo.py MOCHA_CHECKOUT WEIGHTS_DIR [v1 v2]

`WEIGHTS_DIR` holds `mocha-stereo.pth` (V1) and `mocha-V2.pth`. The checkout only has to contain `REF_COMMIT`; the
reference runs from an export of that commit, in an empty working directory. Upstream builds its EfficientNetV2-L with
`timm.create_model(..., pretrained=True)` from a relative file; here with `pretrained=False` (the checkpoints hold all
its weights). Upstream has no V2 code: V2 is the V1 network with the MCCV replaced by the plain left features and the
checkpoint's `CCV.` weights dropped (see the card).
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import timm  # reference environment only
import torch
from _common import pair, reference_tree, write

UPSTREAM = "https://github.com/ZYangChen/MoCha-Stereo"
REF_COMMIT = "78c5889c467fe31414c9425f1280afc0a8f6c1a0"
VARIANTS = {"v1": "mocha-stereo.pth", "v2": "mocha-V2.pth"}  # ref name -> checkpoint file
ITERS = 32


class _LeftFeatures(torch.nn.Module):
    def forward(self, features_left, features_right):  # noqa: ANN001, ANN202, ARG002 - MCCV signature
        return list(features_left)


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    tree = reference_tree(repo, REF_COMMIT) / "MoCha-Stereo"
    sys.path[:0] = [str(tree / "core"), str(tree)]
    os.chdir(tempfile.mkdtemp(prefix="ff-mocha-cwd-"))
    from mocha_stereo import Mocha  # noqa: PLC0415  # reference checkout on sys.path
    from utils.utils import InputPadder  # noqa: PLC0415  # reference checkout on sys.path

    create = timm.create_model
    left, right = (torch.from_numpy(x).permute(2, 0, 1)[None].float() for x in pair())
    padder = InputPadder(left.shape, divis_by=32)
    lp, rp = padder.pad(left, right)
    for name in sys.argv[3:] or VARIANTS:
        args = types.SimpleNamespace(
            hidden_dims=[128] * 3, corr_implementation="reg", shared_backbone=False, corr_levels=2, corr_radius=4,
            n_downsample=2, slow_fast_gru=False, n_gru_layers=3, max_disp=192, mixed_precision=False,
        )  # fmt: skip
        timm.create_model = lambda *a, **k: create(*a, **dict(k, pretrained=False))
        try:
            net = Mocha(args)
        finally:
            timm.create_model = create
        ckpt = torch.load(weights / VARIANTS[name], map_location="cpu", weights_only=True)
        sd = {k.removeprefix("module."): v for k, v in ckpt.items()}
        if name == "v2":
            net.mccv = _LeftFeatures()
            sd = {k: v for k, v in sd.items() if not k.startswith("CCV.")}
        net.load_state_dict(sd, strict=True)
        with torch.inference_mode():
            disp = net.eval()(lp, rp, iters=ITERS, test_mode=True)
        disp = padder.unpad(disp)[0, 0].numpy()
        write(
            f"mocha-{name}-sceneflow@1",
            {"disparity": disp},
            family="mocha_stereo",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, {ITERS} iters, torch {torch.__version__}",
        )
        print(f"  median disparity {np.median(disp):.4f}")


if __name__ == "__main__":
    main()
