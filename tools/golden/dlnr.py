"""DLNR goldens from the reference implementation.

Reference: upstream David-Zhao-1997/High-frequency-Stereo-Matching-Network at `REF_COMMIT` (unpatched), the
evaluate_stereo.py architecture defaults, corr "reg" (upstream's PyTorch lookup; its default reg_cuda needs the CUDA
extension, same values), CPU float32 (mixed_precision off), 10 iterations, upstream `InputPadder(divis_by=32)`.
Upstream imports einops, opt_einsum and scipy; run this with a Python that has them.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/dlnr.py DLNR_CHECKOUT WEIGHTS_DIR [sceneflow middlebury]

`WEIGHTS_DIR` holds upstream's `DLNR_<name>.pth`. The checkout only has to contain `REF_COMMIT`; the reference runs
from an export of that commit (`_common.reference_tree`).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch
from _common import pair, reference_tree, write

UPSTREAM = "https://github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network"
REF_COMMIT = "c10475fe30ff198781ecc7c712c24c781ff05096"
VARIANTS = {"sceneflow": "DLNR_SceneFlow.pth", "middlebury": "DLNR_Middlebury.pth"}  # ref name -> checkpoint file
ITERS = 10


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT)))
    from core.dlnr import DLNR  # noqa: PLC0415  # reference checkout on sys.path
    from core.utils.utils import InputPadder  # noqa: PLC0415  # reference checkout on sys.path

    args = types.SimpleNamespace(
        hidden_dims=[128] * 3, corr_implementation="reg", shared_backbone=False, corr_levels=4, corr_radius=4,
        n_downsample=2, slow_fast_gru=False, n_gru_layers=3, mixed_precision=False,
    )  # fmt: skip
    left, right = (torch.from_numpy(x).permute(2, 0, 1)[None].float() for x in pair())
    padder = InputPadder(left.shape, divis_by=32)
    lp, rp = padder.pad(left, right)
    for name in sys.argv[3:] or VARIANTS:
        net = DLNR(args)
        sd = torch.load(weights / VARIANTS[name], map_location="cpu", weights_only=True)
        net.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()})
        with torch.inference_mode():
            _, flow = net.eval()(lp, rp, iters=ITERS, test_mode=True)
        disp = -padder.unpad(flow)[0, 0].numpy()
        write(
            f"dlnr-{name}@1",
            {"disparity": disp},
            family="dlnr",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, corr reg, {ITERS} iters, torch {torch.__version__}",
        )
        print(f"  median disparity {np.median(disp):.4f}")


if __name__ == "__main__":
    main()
