"""AnyCalib goldens from upstream's `anycalib` package (github.com/javrtg/AnyCalib @ `COMMIT`, unpatched).

Reference: `anycalib.AnyCalib(model_id=None)` with the `anycalib_pinhole.pt` state dict and its `predict(image,
cam_id="pinhole")` (resize to about 102400 pixels, DINOv2 ViT-L ray field, linear fit + 10 Gauss-Newton steps,
intrinsics of the input image), the vertical field of view by `Pinhole().get_vfov` at the input height, CPU float32,
on scikit-image's `coffee.png` (CC0, Rachel Michetti; `IMAGE_SHA256`; stored as `input`).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/anycalib.py <AnyCalib clone> <anycalib_pinhole.pt> COFFEE_PNG
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

UPSTREAM = "https://github.com/javrtg/AnyCalib"
COMMIT = "027a8497d893f4b2596f23d6324c05e4b81064ed"
IMAGE_SHA256 = "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7"  # skimage 0.26 data/coffee.png


def network(clone: Path, weights: Path) -> Any:  # noqa: ANN401 - upstream's model class
    """Upstream `AnyCalib` of `clone` with the checkpoint `weights`, CPU float32, eval; the fit always in float32."""
    sys.path.insert(0, str(clone))
    mod = importlib.import_module("anycalib")
    sys.path.remove(str(clone))
    net = mod.AnyCalib(model_id=None)
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    fit = net.calibrator  # a plain callable

    def fit_fp32(out: dict, data: dict) -> dict:
        """The camera fit in float32 also under `torch.autocast` (its cuSOLVER LU has no float16; fp32: a no-op)."""
        with torch.autocast(data["image"].device.type, enabled=False):
            out = {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in out.items()}
            return fit(out, data)

    net.calibrator = fit_fp32
    return net.eval()


def predict(net: Any, x: torch.Tensor) -> dict[str, torch.Tensor]:  # noqa: ANN401
    """`[B, 3, H, W]` RGB in [0, 1] -> intrinsics `[B, 3, 3]`, vfov `[B]` (float32)."""
    from anycalib.cameras.pinhole import Pinhole  # noqa: PLC0415 - from the clone `network` imported

    params = torch.stack(net.predict(x, cam_id="pinhole")["intrinsics"]).float()  # [B, 4] fx, fy, cx, cy
    k = params.new_zeros(params.shape[0], 3, 3)
    k[:, 0, 0], k[:, 1, 1], k[:, 0, 2], k[:, 1, 2], k[:, 2, 2] = *params.unbind(-1), 1.0
    vfov, _ = Pinhole().get_vfov(params, x.shape[-2])
    return {"intrinsics": k, "vfov": vfov.float()}


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
        "anycalib-pinhole@1",
        {"input": img.astype(np.float32), **out},
        family="anycalib",
        upstream=f"{UPSTREAM}@{COMMIT}",
        patch=None,
        settings=f"CPU float32, anycalib.AnyCalib().predict(cam_id='pinhole'), coffee.png, torch {torch.__version__}",
        task="calibration",
    )
    print("  " + ", ".join(f"{k} {np.array2string(v, precision=5)}" for k, v in out.items()))


if __name__ == "__main__":
    main()
