"""NMRF goldens from the reference implementation.

Reference: upstream aeolusguan/NMRF (`REF_COMMIT`) + `patches/nmrf-topk-ties.patch` (top-k ties to the lower index, as
upstream behaves on CUDA), CPU float32: the default config + the zero-shot evaluation config
(`configs/zero_shot_evaluation.yaml`, max_disp 312), strict load of the checkpoint's "model", upstream's test-mode
forward (pads to a multiple of 8 itself, InputPadder mode "proposal"). Upstream imports einops and yacs; run this with a
Python that has them.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/nmrf.py NMRF_CHECKOUT WEIGHTS_DIR [sceneflow]

`WEIGHTS_DIR` holds upstream's `<name>.pth`. The checkout only has to contain `REF_COMMIT`; the reference runs from an
export of that commit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from _common import pair, reference_tree, write

UPSTREAM = "https://github.com/aeolusguan/NMRF"
REF_COMMIT = "a50bd8f407ae481c38f95cf141e1b5256e962cb7"
PATCH = "nmrf-topk-ties.patch"  # top-k ties to the lower index, as upstream behaves on CUDA
VARIANTS = {"sceneflow": "sceneflow.pth"}  # ref name -> checkpoint file


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    tree = reference_tree(repo, REF_COMMIT, PATCH)
    sys.path.insert(0, str(tree))
    from nmrf.config import get_cfg  # noqa: PLC0415  # reference checkout on sys.path
    from nmrf.models.NMRF import NMRF  # noqa: PLC0415  # reference checkout on sys.path

    cfg = get_cfg()
    cfg.merge_from_file(str(tree / "configs" / "zero_shot_evaluation.yaml"))
    left, right = (torch.from_numpy(x).permute(2, 0, 1)[None].float() for x in pair())
    for name in sys.argv[3:] or VARIANTS:
        net = NMRF(cfg)
        ckpt = torch.load(weights / VARIANTS[name], map_location="cpu", weights_only=True)
        net.load_state_dict(ckpt["model"])
        with torch.inference_mode():
            disp = net.eval()({"img1": left, "img2": right})["disp"]
        disp = disp[0].numpy()
        write(
            f"nmrf-{name}@1",
            {"disparity": disp},
            family="nmrf",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=PATCH,
            settings=f"CPU float32, max_disp {cfg.DPN.MAX_DISP}, torch {torch.__version__}",
        )
        print(f"  median disparity {np.median(disp):.4f}")


if __name__ == "__main__":
    main()
