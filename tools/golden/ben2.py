"""BEN2 goldens from the authors' `BEN2.py` (PramaLLC/BEN2 `COMMIT`; needs einops, timm, opencv).

Reference: `BEN_Base` of `BEN2.py` (unpatched), CPU float32 (`predict` without the forward's fp16 autocast
decorator; with `autocast`: the authors' CUDA fp16 run), the safetensors loaded strictly, on scikit-image's
`coffee.png` (CC0, Rachel Michetti; `IMAGE_SHA256`; stored as `input`). Pre- and post-processing as
`BEN_Base.inference` on float tensors: resize to 1024 x 1024 (bicubic, antialiased; upstream: PIL Lanczos on the
8-bit image), ImageNet normalisation, the network's sigmoid mask, bilinear resize back to the image size, min-max
normalised (upstream then rounds to 8 bits; float keeps the golden free of rounding).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/ben2.py <BEN2 checkout> <weights.safetensors> COFFEE_PNG
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from _common import write
from PIL import Image

UPSTREAM = "https://github.com/PramaLLC/BEN2"
COMMIT = "2c99a5da477b5523585bfa5c893888a6e818a8f6"
IMAGE_SHA256 = "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7"  # skimage 0.26 data/coffee.png
SIZE = 1024
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def network(checkout: Path, weights: Path) -> Any:  # noqa: ANN401 - the upstream model class
    """`BEN_Base` of `checkout/BEN2.py` with `weights`, CPU float32, eval."""
    from safetensors.torch import load_file  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("upstream_ben2", checkout / "BEN2.py")
    if spec is None or spec.loader is None:
        msg = f"no BEN2.py in {checkout}"
        raise SystemExit(msg)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    net = mod.BEN_Base()
    net.load_state_dict(load_file(weights), strict=True)
    return net.eval()


def predict(net: Any, x: torch.Tensor, *, autocast: bool = False) -> torch.Tensor:  # noqa: ANN401 - the upstream model
    """`[B, 3, H, W]` RGB in [0, 1] -> alpha `[B, H, W]` float32 (the network's device).

    `autocast`: upstream's decorated forward (CUDA fp16 autocast, inference mode); else the plain function, float32.
    """
    y = F.interpolate(x, (SIZE, SIZE), mode="bicubic", align_corners=False, antialias=True)
    mean, std = (torch.tensor(v, device=x.device)[:, None, None] for v in (MEAN, STD))
    forward = type(net).forward if autocast else inspect.unwrap(type(net).forward)
    y = forward(net, (y - mean) / std).float()
    y = F.interpolate(y, x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    lo, hi = y.amin((1, 2), keepdim=True), y.amax((1, 2), keepdim=True)
    return (y - lo) / (hi - lo)


def main() -> None:
    checkout, weights, png = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's coffee.png"
        raise SystemExit(msg)
    img = np.asarray(Image.open(png).convert("RGB"))
    net = network(checkout, weights)
    x = torch.from_numpy(img.copy()).permute(2, 0, 1)[None].float() / 255
    with torch.no_grad():
        alpha = predict(net, x)[0].numpy().astype(np.float32)
    write(
        "ben2@1",
        {"input": img.astype(np.float32), "alpha": alpha},
        family="ben2",
        upstream=f"{UPSTREAM}@{COMMIT}",
        patch=None,
        settings=(
            f"CPU float32, BEN2.py + weights PramaLLC/BEN2@e48a20765fb421d19dcdb0bf3cc61e802ca5ec8f, {SIZE}x{SIZE} "
            f"bicubic antialiased, ImageNet normalisation, sigmoid, bilinear back, min-max, coffee.png, torch "
            f"{torch.__version__}"
        ),
        task="matting",
    )
    print(f"  alpha {alpha.shape}, mean {alpha.mean():.4f}")


if __name__ == "__main__":
    main()
