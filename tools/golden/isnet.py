"""IS-Net goldens from the reference implementation (xuebinqin/DIS `COMMIT`).

Reference: `IS-Net/models/isnet.py` `ISNetDIS` (unpatched, loaded by path; needs torchvision), CPU float32, the
variant's `.pth` loaded strictly, on scikit-image's `coffee.png` (CC0, Rachel Michetti; `IMAGE_SHA256`; stored as
`input`). Pre- and post-processing as `IS-Net/Inference.py` on float tensors: bilinear resize to 1024 x 1024 (upstream
resizes the uint8 tensor, which truncates), normalise with mean 0.5 and std 1, `d1`, bilinear resize back, min-max
(upstream then rounds to 8 bits).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/isnet.py <DIS checkout> <weights dir> COFFEE_PNG
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

FAMILY = "isnet"
UPSTREAM = "https://github.com/xuebinqin/DIS"
COMMIT = "b6764e20381f6f42a70f83fa3324181529ed1403"
IMAGE_SHA256 = "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7"  # skimage 0.26 data/coffee.png
MEAN, STD = (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)


def network(checkout: Path, weights: Path) -> Any:  # noqa: ANN401 - the upstream model class
    """`ISNetDIS` of `checkout/IS-Net/models/isnet.py` with `weights`, CPU float32, eval."""
    spec = importlib.util.spec_from_file_location("upstream_isnet", checkout / "IS-Net" / "models" / "isnet.py")
    if spec is None or spec.loader is None:
        msg = f"no IS-Net/models/isnet.py in {checkout}"
        raise SystemExit(msg)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    net = mod.ISNetDIS(3, 1)
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return net.eval()


def predict(net: Any, x: torch.Tensor, size: int) -> torch.Tensor:  # noqa: ANN401 - the upstream model
    """`[B, 3, H, W]` RGB in [0, 1] -> alpha `[B, H, W]` float32."""
    y = F.interpolate(x, (size, size), mode="bilinear", align_corners=False)
    mean, std = (torch.tensor(v, device=x.device)[:, None, None] for v in (MEAN, STD))
    d1 = net((y - mean) / std)[0][0].float()
    y = F.interpolate(d1, x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    lo, hi = y.amin((1, 2), keepdim=True), y.amax((1, 2), keepdim=True)
    return (y - lo) / (hi - lo)


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
        net = network(checkout, wdir / variant.weights[0].name)
        with torch.no_grad():
            alpha = predict(net, x, p["size"])[0].numpy().astype(np.float32)
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img.astype(np.float32), "alpha": alpha},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{COMMIT}",
            patch=None,
            settings=(
                f"CPU float32, IS-Net/models/isnet.py, {p['size']}x{p['size']} bilinear, mean 0.5 std 1, d1, bilinear "
                f"back, min-max, coffee.png, torch {torch.__version__}"
            ),
            task="matting",
        )
        print(f"  {variant.id}: alpha {alpha.shape}, mean {alpha.mean():.4f}")


if __name__ == "__main__":
    main()
