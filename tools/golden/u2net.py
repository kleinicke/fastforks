"""U²-Net goldens from the reference implementation (xuebinqin/U-2-Net `COMMIT`).

Reference: `model/u2net.py` `U2NET` / `U2NETP` (unpatched, loaded by path), CPU float32, the variant's `.pth` loaded
strictly, on scikit-image's `coffee.png` (CC0, Rachel Michetti; `IMAGE_SHA256`; stored as `input`). Pre- and
post-processing as `u2net_test.py` on float tensors: resize to 320 x 320 (bilinear, antialiased; upstream: scikit-image
`resize`), divide by the maximum (`ToTensorLab`), ImageNet normalisation, `d0`, min-max (`normPRED`), bilinear resize
back to the image size (upstream: PIL on the 8-bit map; float keeps the golden free of rounding).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/u2net.py <U-2-Net checkout> <weights dir> COFFEE_PNG
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from _common import write
from PIL import Image

from fastforks._core import registry

FAMILY = "u2net"
UPSTREAM = "https://github.com/xuebinqin/U-2-Net"
COMMIT = "ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29"
IMAGE_SHA256 = "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7"  # skimage 0.26 data/coffee.png
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def network(checkout: Path, weights: Path, arch: str) -> Any:  # noqa: ANN401 - the upstream model class
    """`U2NET` / `U2NETP` of `checkout/model/u2net.py` with `weights`, CPU float32, eval."""
    spec = importlib.util.spec_from_file_location("upstream_u2net", checkout / "model" / "u2net.py")
    if spec is None or spec.loader is None:
        msg = f"no model/u2net.py in {checkout}"
        raise SystemExit(msg)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    net = {"u2net": mod.U2NET, "u2netp": mod.U2NETP}[arch](3, 1)
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return net.eval()


def predict(net: Any, x: torch.Tensor, size: int) -> torch.Tensor:  # noqa: ANN401 - the upstream model
    """`[B, 3, H, W]` RGB in [0, 1] -> alpha `[B, H, W]` float32."""
    y = F.interpolate(x, (size, size), mode="bilinear", align_corners=False, antialias=True)
    y = y / y.amax((1, 2, 3), keepdim=True)
    mean, std = (torch.tensor(v, device=x.device)[:, None, None] for v in (MEAN, STD))
    d0 = net((y - mean) / std)[0].float()
    lo, hi = d0.amin((1, 2, 3), keepdim=True), d0.amax((1, 2, 3), keepdim=True)
    return F.interpolate((d0 - lo) / (hi - lo), x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]


def main() -> None:
    checkout, wdir, png = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's coffee.png"
        raise SystemExit(msg)
    img = np.asarray(Image.open(png).convert("RGB"))
    x = torch.from_numpy(img.copy()).permute(2, 0, 1)[None].float() / 255
    for card, variant in registry.cards("matting").values():
        if card.family != FAMILY:
            continue
        p = {**card.params, **variant.params}
        net = network(checkout, wdir / variant.weights[0].name, p["arch"])
        with torch.no_grad():
            alpha = predict(net, x, p["size"])[0].numpy().astype(np.float32)
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img.astype(np.float32), "alpha": alpha},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{COMMIT}",
            patch=None,
            settings=(
                f"CPU float32, model/u2net.py, {p['size']}x{p['size']} bilinear antialiased, divide by the max, "
                f"ImageNet normalisation, d0, min-max, bilinear back, coffee.png, torch {torch.__version__}"
            ),
            task="matting",
        )
        print(f"  {variant.id}: alpha {alpha.shape}, mean {alpha.mean():.4f}")


if __name__ == "__main__":
    main()
