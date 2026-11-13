"""BoQ goldens from upstream's `hubconf.py` (github.com/amaralibey/Bag-of-Queries @ `COMMIT`, unpatched).

Reference: `get_trained_boq("dinov2", 12288)` without the download (its backbone's `torch.hub.load` of
`facebookresearch/dinov2` served from a local checkout at `DINOV2_COMMIT`, `pretrained=False`; xFormers off), the
GitHub release weights loaded strictly, CPU float32, on scikit-image's `coffee.png` (CC0, Rachel Michetti;
`IMAGE_SHA256`; stored as `input`). Pre-processing as the README: resize to 322 x 322 (bicubic, antialiased:
torchvision `Resize` on tensors), ImageNet normalisation (resizing first and normalising after is the same affine map
per channel).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/boq.py <BoQ clone> <DINOv2 clone> <dinov2_12288.pth> COFFEE_PNG
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from _common import reference_tree, write
from PIL import Image

UPSTREAM = "https://github.com/amaralibey/Bag-of-Queries"
COMMIT = "1a4965ea7dfd9bd0dd846adf7a0e430f68101d12"
DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"  # tools/golden/dinov2.py REF_COMMIT
IMAGE_SHA256 = "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7"  # skimage 0.26 data/coffee.png
SIZE = 322
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def network(clone: Path, weights: Path, dinov2: Path | None = None) -> Any:  # noqa: ANN401 - upstream's model class
    """Upstream `VPRModel` of `clone` (its `hubconf.py`) with `weights`, CPU float32, eval.

    `dinov2`: the facebookresearch/dinov2 checkout (default: `$FF_DINOV2_REPO`).
    """
    os.environ["XFORMERS_DISABLED"] = "1"
    tree = str(reference_tree(dinov2 or Path(os.environ["FF_DINOV2_REPO"]), DINOV2_COMMIT))
    hub_load = torch.hub.load

    def local(_repo: str, name: str, **kw: Any) -> Any:  # noqa: ANN401
        return hub_load(tree, name, source="local", pretrained=False, **kw)

    paths = [str(clone / "src"), str(clone)]  # upstream's `boq` module before this file (also named boq)
    sys.path[:0] = paths
    own = sys.modules.pop("boq", None)
    try:
        mod = importlib.import_module("hubconf")
    finally:
        del sys.path[: len(paths)]
        if own is not None:
            sys.modules["boq"] = own
    with mock.patch.object(torch.hub, "load", local), mock.patch.object(torch.hub, "load_state_dict_from_url") as url:
        url.return_value = torch.load(weights, map_location="cpu", weights_only=True)
        net = mod.get_trained_boq("dinov2", 12288)  # loads the state dict strictly
    return net.eval()


def predict(net: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - upstream's model class
    """`[B, 3, H, W]` RGB in [0, 1] -> descriptors `[B, 12288]` float32 (the network's dtype and device)."""
    p = next(net.parameters())
    y = F.interpolate(x, (SIZE, SIZE), mode="bicubic", align_corners=False, antialias=True)
    mean, std = (torch.tensor(v, device=x.device)[:, None, None] for v in (MEAN, STD))
    return net(((y - mean) / std).to(p.dtype))[0].float()


def main() -> None:
    clone, dinov2, weights, png = (Path(a) for a in sys.argv[1:5])
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's coffee.png"
        raise SystemExit(msg)
    img = np.asarray(Image.open(png).convert("RGB"))
    net = network(clone, weights, dinov2)
    x = torch.from_numpy(img.copy()).permute(2, 0, 1)[None].float() / 255
    with torch.no_grad():
        d = predict(net, x)[0].numpy().astype(np.float32)
    write(
        "boq-dinov2@1",
        {"input": img.astype(np.float32), "descriptor": d},
        family="boq",
        upstream=f"{UPSTREAM}@{COMMIT}",
        patch=None,
        settings=(
            f"CPU float32, hubconf.py get_trained_boq('dinov2', 12288) + release v1.0 weights, dinov2 hub code @"
            f"{DINOV2_COMMIT[:12]}, {SIZE}x{SIZE} bicubic antialiased, ImageNet normalisation, coffee.png, "
            f"torch {torch.__version__}"
        ),
        task="retrieval",
    )
    print(f"  descriptor {d.shape}, norm {np.linalg.norm(d):.6f}")


if __name__ == "__main__":
    main()
