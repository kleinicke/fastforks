"""GREAT-IGEV goldens from the reference implementation.

Reference: upstream JarvisLee0423/GREAT-Stereo (`REF_COMMIT`), CPU float32, 32 iterations, upstream
`InputPadder(divis_by=32)`, upstream's evaluation ranges (max_disp 768 for Middlebury, 192 for the others). Upstream
builds its backbone with timm (`mobilenetv2_100`, `pretrained=True` hard-coded) and uses einops; run this with a
Python that has timm and einops installed (see `_patch_timm` for the two shims). The checkpoints hold the backbone
weights.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/great_igev.py GREAT_STEREO_CHECKOUT WEIGHTS_DIR [sceneflow ...]

`WEIGHTS_DIR` holds the upstream `great-igev-<variant>-submit.pth` files.
The checkout only has to contain `REF_COMMIT`; the reference runs from an export of that commit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from _common import pair, reference_tree, write

if TYPE_CHECKING:
    from collections.abc import Callable

UPSTREAM = "https://github.com/JarvisLee0423/GREAT-Stereo"
REF_COMMIT = "2a6b1efede8c3a9eec9540790c2700ea2f3dee7d"
VARIANTS = {"sceneflow": 192, "kitti": 192, "eth3d": 192, "middlebury": 768}  # variant -> max_disp
ITERS = 32


def _patch_timm() -> None:
    """Shim timm for the reference: no ImageNet download, and a stem `act1` for timm >= 0.6.

    timm >= 0.6 folds the stem ReLU6 into `bn1` (`BatchNormAct2d`), where upstream also reads a separate `act1`: give
    it an identity, the same math.
    """
    import timm  # noqa: PLC0415 - reference-only dependency

    create = cast("Callable[..., torch.nn.Module]", timm.create_model)

    def create_model(*a: object, **kw: object) -> torch.nn.Module:
        model = create(*a, **{**kw, "pretrained": False})
        if not hasattr(model, "act1"):
            model.act1 = torch.nn.Identity()
        return model

    vars(timm)["create_model"] = create_model


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    _patch_timm()
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT)))
    from models.great_stereo.great_igev_stereo import GREATStereo  # noqa: PLC0415  # reference checkout on sys.path
    from utils.utils import InputPadder  # noqa: PLC0415  # reference checkout on sys.path

    left, right = (torch.from_numpy(x).permute(2, 0, 1)[None].float() for x in pair())
    padder = InputPadder(left.shape, divis_by=32)
    lp, rp = padder.pad(left, right)
    for name in sys.argv[3:] or VARIANTS:
        args = argparse.Namespace(
            name="great-igev-stereo", channels=[128] * 3, n_downsample=2, n_gru_layers=3, max_disp=VARIANTS[name],
            cv_levels=2, cv_radius=4, slow_fast_gru=False, mixed_precision=False, precision_dtype="float32",
        )  # fmt: skip
        net = GREATStereo(args)
        sd = torch.load(weights / f"great-igev-{name}-submit.pth", map_location="cpu", weights_only=True)
        net.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()})
        with torch.inference_mode():
            disp = net.eval()(lp, rp, iters=ITERS, test_mode=True)[1]
        disp = padder.unpad(disp)[0, 0].numpy()
        settings = f"CPU float32, {ITERS} iters, max_disp {VARIANTS[name]}, timm shims, torch {torch.__version__}"
        write(
            f"great-igev-{name}@1",
            {"disparity": disp},
            family="great_igev",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=settings,
        )
        print(f"  median disparity {np.median(disp):.4f}")


if __name__ == "__main__":
    main()
