"""GeoCalib goldens from upstream's `geocalib` package (github.com/cvg/GeoCalib @ `COMMIT`, unpatched).

Reference: `geocalib.GeoCalib(weights=<checkpoint>)` and its `calibrate` (pinhole camera model, kornia resize to a
320 shorter side, crop to multiples of 32, 30 LM steps), CPU float32, on scikit-image's `coffee.png` (CC0, Rachel
Michetti; `IMAGE_SHA256`; stored as `input`). The Hamburger heads draw random NMF bases on every call; the reference
runs after `torch.manual_seed(0)`, whose first two draws are the fixed bases the fastforks port uses.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/geocalib.py <GeoCalib clone> <geocalib-pinhole.tar> COFFEE_PNG
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from _common import write
from PIL import Image

UPSTREAM = "https://github.com/cvg/GeoCalib"
COMMIT = "1dedb29f94fc11b896c5f572005978aeebccd67f"
IMAGE_SHA256 = "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7"  # skimage 0.26 data/coffee.png
OUTPUTS = ("intrinsics", "gravity", "roll", "pitch", "vfov")


def network(clone: Path, weights: Path) -> Any:  # noqa: ANN401 - upstream's model class
    """Upstream `GeoCalib` of `clone` with the checkpoint `weights`, CPU float32, eval."""
    sys.path.insert(0, str(clone))
    mod = importlib.import_module("geocalib")
    sys.path.remove(str(clone))
    return mod.GeoCalib(weights=str(weights)).eval()


def predict(net: Any, x: torch.Tensor, camera_model: str = "pinhole") -> dict[str, torch.Tensor]:  # noqa: ANN401
    """`[B, 3, H, W]` RGB in [0, 1] -> intrinsics `[B, 3, 3]`, gravity `[B, 3]`, roll / pitch / vfov `[B]` (float32).

    The NMF bases are drawn from the global generator after `torch.manual_seed(0)` (as the goldens).
    """
    torch.manual_seed(0)
    out = net.calibrate(x, camera_model=camera_model)
    camera, gravity = out["camera"], out["gravity"]
    maps = {"intrinsics": camera.K, "gravity": gravity.vec3d, "roll": gravity.roll, "pitch": gravity.pitch}
    return {k: v.float() for k, v in (maps | {"vfov": camera.vfov}).items()}


def main() -> None:
    clone, weights, png = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's coffee.png"
        raise SystemExit(msg)
    img = np.asarray(Image.open(png).convert("RGB"))
    net = network(clone, weights)
    x = torch.from_numpy(img.copy()).permute(2, 0, 1)[None].float() / 255
    with torch.no_grad():
        out = {k: v[0].numpy().astype(np.float32) for k, v in predict(net, x).items()}
    write(
        "geocalib-pinhole@1",
        {"input": img.astype(np.float32), **out},
        family="geocalib",
        upstream=f"{UPSTREAM}@{COMMIT}",
        patch=None,
        settings=(
            f"CPU float32, geocalib.GeoCalib(weights=geocalib-pinhole.tar).calibrate(camera_model='pinhole') after "
            f"torch.manual_seed(0), coffee.png, torch {torch.__version__}"
        ),
        task="calibration",
    )
    print("  " + ", ".join(f"{k} {np.array2string(v, precision=5)}" for k, v in out.items()))


if __name__ == "__main__":
    main()
