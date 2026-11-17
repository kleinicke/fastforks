"""InSPyReNet goldens from the reference implementation (plemeri/transparent-background `COMMIT`, 1.3.4).

Reference: `transparent_background/InSPyReNet.py` `InSPyReNet_SwinB(depth=64, threshold=None, base_size)` as `Remover`
builds it (unpatched, loaded from the package directory without its `__init__`; needs timm and kornia), CPU float32,
the variant's checkpoint loaded strictly, on scikit-image's `coffee.png` (CC0, Rachel Michetti; `IMAGE_SHA256`; stored
as `input`). Pre- and post-processing as `Remover.process` on float tensors: resize to `size` x `size` (antialiased
bilinear; upstream: PIL bilinear on the uint8 image), ImageNet normalisation, the network (sigmoid, min-max), bilinear
resize back with `align_corners=True`.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/inspyrenet.py <transparent_background dir> <weights dir> COFFEE_PNG
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from _common import write
from PIL import Image

from fastforks._core import registry

FAMILY = "inspyrenet"
UPSTREAM = "https://github.com/plemeri/transparent-background"
COMMIT = "f0fa91701a98cfc8e955c554e84522f365ec6da3"
IMAGE_SHA256 = "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7"  # skimage 0.26 data/coffee.png
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def network(package: Path, weights: Path, size: int) -> Any:  # noqa: ANN401 - the upstream model class
    """`InSPyReNet_SwinB` of the `transparent_background` package directory `package` with `weights`, float32, eval."""
    if "transparent_background" not in sys.modules:  # the package without its __init__ (Remover's GUI / IO imports)
        pkg = types.ModuleType("transparent_background")
        pkg.__path__ = [str(package)]
        sys.modules["transparent_background"] = pkg
    mod = importlib.import_module("transparent_background.InSPyReNet")
    net = mod.InSPyReNet_SwinB(depth=64, pretrained=False, threshold=None, base_size=[size, size])
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return net.eval()


def predict(net: Any, x: torch.Tensor, size: int) -> torch.Tensor:  # noqa: ANN401 - the upstream model
    """`[1, 3, H, W]` RGB in [0, 1] (on the network's device) -> alpha `[1, H, W]` float32."""
    y = F.interpolate(x, (size, size), mode="bilinear", align_corners=False, antialias=True)
    mean, std = (torch.tensor(v, device=x.device)[:, None, None] for v in (MEAN, STD))
    pred = net((y - mean) / std).float()
    return F.interpolate(pred, x.shape[-2:], mode="bilinear", align_corners=True)[:, 0]


def main() -> None:
    package, wdir, png = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's coffee.png"
        raise SystemExit(msg)
    img = np.asarray(Image.open(png).convert("RGB"))
    x = torch.from_numpy(img.copy()).permute(2, 0, 1)[None].float() / 255
    for card, variant in registry.cards("matting").values():
        if card.family != FAMILY:
            continue
        size = {**card.params, **variant.params}["size"]
        net = network(package, wdir / variant.weights[0].name, size)
        with torch.no_grad():
            alpha = predict(net, x, size)[0].numpy().astype(np.float32)
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img.astype(np.float32), "alpha": alpha},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{COMMIT}",
            patch=None,
            settings=(
                f"CPU float32, InSPyReNet_SwinB threshold None, {size}x{size} antialiased bilinear, ImageNet, "
                f"sigmoid min-max, bilinear back (align_corners), coffee.png, torch {torch.__version__}"
            ),
            task="matting",
        )
        print(f"  {variant.id}: alpha {alpha.shape}, mean {alpha.mean():.4f}")


if __name__ == "__main__":
    main()
