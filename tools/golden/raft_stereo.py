"""RAFT-Stereo goldens from the reference implementation.

Reference: upstream princeton-vl/RAFT-Stereo at `REF_COMMIT` (unpatched), default architecture, corr "reg", CPU
float32, 32 iterations, upstream `InputPadder(divis_by=32)`.

    FASTFORKS_GOLDEN_PATH=... uv run python tools/golden/raft_stereo.py RAFT_STEREO_CHECKOUT WEIGHTS_DIR [eth3d ...]

`WEIGHTS_DIR` holds the upstream `raftstereo-<variant>.pth` files. The checkout only has to contain `REF_COMMIT`;
the reference runs from an export of that commit (`_common.reference_tree`).
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch
from _common import pair, reference_tree, write

UPSTREAM = "https://github.com/princeton-vl/RAFT-Stereo"
REF_COMMIT = "6e93ed2169bd858dbb43033988563f3b0bb49506"
VARIANTS = ("middlebury", "sceneflow", "eth3d")
ITERS = 32


def _stub_training_imports() -> None:
    """Upstream imports opt_einsum and scipy at module level; inference uses neither."""
    sys.modules["opt_einsum"] = types.SimpleNamespace(contract=None)  # import stub
    scipy = types.ModuleType("scipy")
    scipy.interpolate = types.ModuleType("scipy.interpolate")  # import stub
    sys.modules |= {"scipy": scipy, "scipy.interpolate": scipy.interpolate}


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    _stub_training_imports()
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT)))
    from core.raft_stereo import RAFTStereo  # noqa: PLC0415  # reference checkout on sys.path
    from core.utils.utils import InputPadder  # noqa: PLC0415  # reference checkout on sys.path

    args = argparse.Namespace(
        hidden_dims=[128] * 3, corr_implementation="reg", shared_backbone=False, corr_levels=4, corr_radius=4,
        n_downsample=2, context_norm="batch", slow_fast_gru=False, n_gru_layers=3, mixed_precision=False,
    )  # fmt: skip
    left, right = (torch.from_numpy(x).permute(2, 0, 1)[None].float() for x in pair())
    padder = InputPadder(left.shape, divis_by=32)
    lp, rp = padder.pad(left, right)
    for name in sys.argv[3:] or VARIANTS:
        net = RAFTStereo(args)
        sd = torch.load(weights / f"raftstereo-{name}.pth", map_location="cpu", weights_only=True)
        net.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()})
        with torch.inference_mode():
            _, flow = net.eval()(lp, rp, iters=ITERS, test_mode=True)
        disp = -padder.unpad(flow)[0, 0].numpy()
        settings = f"CPU float32, {ITERS} iters, torch {torch.__version__}"
        write(
            f"raft-stereo-{name}@1",
            {"disparity": disp},
            family="raft_stereo",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=settings,
        )
        print(f"  median disparity {np.median(disp):.4f}")


if __name__ == "__main__":
    main()
