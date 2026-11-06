"""IGEV-Stereo goldens from the reference implementation.

Reference: upstream gangweix/IGEV (`REF_COMMIT`), IGEV-Stereo, CPU float32, 32 iterations, upstream
`InputPadder(divis_by=32)`. Upstream builds its backbone with timm (`mobilenetv2_100`, `pretrained=True` hard-coded);
run this with a Python that has timm installed (see `_patch_timm` for the two shims). The checkpoints hold the
backbone weights.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/igev_stereo.py IGEV_CHECKOUT WEIGHTS_DIR [sceneflow ...]

`WEIGHTS_DIR` holds the upstream `<variant>.pth` files (or upstream's `<group>/<variant>.pth` layout).
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

UPSTREAM = "https://github.com/gangweix/IGEV"
REF_COMMIT = "4a3d937aab788b3ff3ed49b7b5c81340ab46a5eb"
VARIANTS = ("sceneflow", "middlebury", "eth3d", "kitti12", "kitti15")
ITERS = 32


def _patch_timm() -> None:
    """Shim timm for the reference: no ImageNet download, and a stem `act1` for timm >= 0.6.

    timm >= 0.6 folds the stem ReLU6 into `bn1` (`BatchNormAct2d`), where upstream (written for timm 0.5) also reads a
    separate `act1`: give it an identity, the same math.
    """
    import timm  # noqa: PLC0415 - reference-only dependency

    create = cast("Callable[..., torch.nn.Module]", timm.create_model)

    def create_model(*a: object, **kw: object) -> torch.nn.Module:
        model = create(*a, **{**kw, "pretrained": False})
        if not hasattr(model, "act1"):
            model.act1 = torch.nn.Identity()
        return model

    vars(timm)["create_model"] = create_model


def _checkpoint(weights: Path, name: str) -> Path:
    flat = weights / f"{name}.pth"
    if flat.is_file():
        return flat
    (hit,) = weights.glob(f"*/{name}.pth")
    return hit


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    _patch_timm()
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT) / "IGEV-Stereo"))
    from core.igev_stereo import IGEVStereo  # noqa: PLC0415  # reference checkout on sys.path
    from core.utils.utils import InputPadder  # noqa: PLC0415  # reference checkout on sys.path

    args = argparse.Namespace(
        hidden_dims=[128] * 3, corr_levels=2, corr_radius=4, n_downsample=2, n_gru_layers=3, max_disp=192,
        mixed_precision=False, precision_dtype="float32",
    )  # fmt: skip
    left, right = (torch.from_numpy(x).permute(2, 0, 1)[None].float() for x in pair())
    padder = InputPadder(left.shape, divis_by=32)
    lp, rp = padder.pad(left, right)
    for name in sys.argv[3:] or VARIANTS:
        net = IGEVStereo(args)
        sd = torch.load(_checkpoint(weights, name), map_location="cpu", weights_only=True)
        net.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()})
        with torch.inference_mode():
            disp = net.eval()(lp, rp, iters=ITERS, test_mode=True)
        disp = padder.unpad(disp)[0, 0].numpy()
        settings = f"CPU float32, {ITERS} iters, timm shims, torch {torch.__version__}"
        write(
            f"igev-stereo-{name}@1",
            {"disparity": disp},
            family="igev_stereo",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=settings,
        )
        print(f"  median disparity {np.median(disp):.4f}")


if __name__ == "__main__":
    main()
