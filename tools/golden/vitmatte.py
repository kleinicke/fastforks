"""ViTMatte goldens from the transformers port (`VitMatteForImageMatting`, the code the HF weights are converted for).

Reference: transformers `VitMatteImageProcessor` + `VitMatteForImageMatting` (unpatched), CPU float32, loaded from a
snapshot of the HF repo (config, preprocessor config, weights) on the authors' demo pair `retriever_rgb.png` /
`retriever_trimap.png` (hustvl/ViTMatte `COMMIT`, MIT; `IMAGE_SHA256`; RGB and the trimap's first channel stored as
`input` / `trimap`, 0..255). The processor normalises the image to [-1, 1], appends the trimap / 255 and zero-pads to
a multiple of 32; the alpha is cropped back to the image.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/vitmatte.py <ref> <HF snapshot dir> <demo dir>
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from _common import write
from PIL import Image

UPSTREAM = "https://github.com/huggingface/transformers"
COMMIT = "8cd7ef068380977c3962c4cb733cb1fe7f2241a5"  # hustvl/ViTMatte (demo images)
IMAGE_SHA256 = {
    "retriever_rgb.png": "cbb6a257cf6b63ab992d82f266b2cd129aa13994988341bd0b7133071236bb9c",
    "retriever_trimap.png": "6f9dcc280203641a8ce1e6a81ec661c4b6693217d6c95a9ec297c9fe5e2adbd6",
}
REPOS = {
    "vitmatte-s@1": "hustvl/vitmatte-small-composition-1k@6a58ad7646403c1df626fbd746900aec7361ea1d",
    "vitmatte-b@1": "hustvl/vitmatte-base-composition-1k@bf486d01a7d9e3dbcc8400f7942835caf0eaf76e",
}


def network(snap: Path, *, fp16: bool = False) -> transformers.VitMatteForImageMatting:
    """`VitMatteForImageMatting` from an HF snapshot (config + weights), eval, float32 or the authors' fp16 dtype."""
    dtype = torch.float16 if fp16 else torch.float32
    return transformers.VitMatteForImageMatting.from_pretrained(snap, torch_dtype=dtype).eval()


def predict(net: transformers.VitMatteForImageMatting, image: torch.Tensor, trimap: torch.Tensor) -> torch.Tensor:
    """The processor's maths on tensors: `[1, 3, H, W]` RGB and `[1, 1, H, W]` trimap in [0, 1] -> alpha `[1, H, W]`.

    Normalise to [-1, 1], append the trimap, zero-pad bottom-right to a multiple of 32, run, crop.
    """
    h, w = image.shape[-2:]
    x = torch.cat([(image - 0.5) / 0.5, trimap], 1)
    x = F.pad(x, (0, -w % 32, 0, -h % 32)).to(net.dtype)
    with torch.inference_mode():
        return net(pixel_values=x).alphas[:, 0, :h, :w].float()


def main() -> None:
    ref, snap, demo = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    for name, sha in IMAGE_SHA256.items():
        if hashlib.sha256((demo / name).read_bytes()).hexdigest() != sha:
            msg = f"{demo / name} is not the ViTMatte demo image"
            raise SystemExit(msg)
    rgb = Image.open(demo / "retriever_rgb.png").convert("RGB")
    tri = Image.open(demo / "retriever_trimap.png").convert("L")
    proc = transformers.VitMatteImageProcessor.from_pretrained(snap)
    net = transformers.VitMatteForImageMatting.from_pretrained(snap, torch_dtype=torch.float32).eval()
    inputs = proc(images=rgb, trimaps=tri, return_tensors="pt")
    with torch.no_grad():
        alphas = net(**inputs).alphas
    h, w = rgb.height, rgb.width
    alpha = alphas[0, 0, :h, :w].numpy().astype(np.float32)
    write(
        ref,
        {
            "input": np.asarray(rgb).astype(np.float32),
            "trimap": np.asarray(tri).astype(np.float32),
            "alpha": alpha,
        },
        family="vitmatte",
        upstream=f"{UPSTREAM}@v{transformers.__version__}",
        patch=None,
        settings=(
            f"CPU float32, transformers {transformers.__version__} VitMatteImageProcessor + VitMatteForImageMatting, "
            f"weights {REPOS[ref]}, demo retriever_rgb.png + retriever_trimap.png (hustvl/ViTMatte@{COMMIT}), "
            f"pad to 32, crop, torch {torch.__version__}"
        ),
        task="matting",
    )
    print(f"  {ref}: alpha {alpha.shape}, mean {alpha.mean():.4f}")


if __name__ == "__main__":
    main()
