"""MegaLoc goldens from upstream's `megaloc_model.py` (github.com/gmberton/MegaLoc @ `COMMIT`, unpatched).

Reference: `MegaLoc()` with the HF weights (gberton/MegaLoc @ `HF_REVISION`) loaded strictly, CPU float32, on
scikit-image's `coffee.png` (CC0, Rachel Michetti; `IMAGE_SHA256`; stored as `input`). Pre-processing as the README:
ImageNet normalisation, resize to 322 x 322 (bilinear, antialiased: torchvision `Resize` on tensors; resizing first
and normalising after is the same affine map per channel).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/megaloc.py <MegaLoc clone> <megaloc.safetensors> COFFEE_PNG
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from _common import write
from PIL import Image

UPSTREAM = "https://github.com/gmberton/MegaLoc"
COMMIT = "5fe0dd697c4a70ba3e23607f6716ab3c606b16db"
HF_REVISION = "a0f34722c4297ff787e022433799250180860af7"
IMAGE_SHA256 = "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7"  # skimage 0.26 data/coffee.png
SIZE = 322
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def network(clone: Path, weights: Path) -> Any:  # noqa: ANN401 - upstream's model class
    """Upstream `MegaLoc` of `clone` (its `megaloc_model.py`) with `weights`, CPU float32, eval."""
    from safetensors.torch import load_file  # noqa: PLC0415

    sys.path.insert(0, str(clone))
    mod = importlib.import_module("megaloc_model")
    sys.path.remove(str(clone))
    net = mod.MegaLoc()
    net.load_state_dict(load_file(weights), strict=True)
    return net.eval()


def predict(net: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - upstream's model class
    """`[B, 3, H, W]` RGB in [0, 1] -> descriptors `[B, 8448]` float32 (the network's dtype and device)."""
    p = next(net.parameters())
    y = F.interpolate(x, (SIZE, SIZE), mode="bilinear", align_corners=False, antialias=True)
    mean, std = (torch.tensor(v, device=x.device)[:, None, None] for v in (MEAN, STD))
    return net(((y - mean) / std).to(p.dtype)).float()


def main() -> None:
    clone, weights, png = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's coffee.png"
        raise SystemExit(msg)
    img = np.asarray(Image.open(png).convert("RGB"))
    net = network(clone, weights)
    x = torch.from_numpy(img.copy()).permute(2, 0, 1)[None].float() / 255
    with torch.no_grad():
        d = predict(net, x)[0].numpy().astype(np.float32)
    write(
        "megaloc@1",
        {"input": img.astype(np.float32), "descriptor": d},
        family="megaloc",
        upstream=f"{UPSTREAM}@{COMMIT}",
        patch=None,
        settings=(
            f"CPU float32, megaloc_model.py + weights gberton/MegaLoc@{HF_REVISION}, {SIZE}x{SIZE} bilinear "
            f"antialiased, ImageNet normalisation, coffee.png, torch {torch.__version__}"
        ),
        task="retrieval",
    )
    print(f"  descriptor {d.shape}, norm {np.linalg.norm(d):.6f}")


if __name__ == "__main__":
    main()
