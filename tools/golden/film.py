"""FILM frame-interpolation goldens from the reference implementation.

Reference: the authors' TorchScript release of dajes/frame-interpolation-pytorch (`film_net_fp32.pt`, the card's
weights file) loaded with `torch.jit.load`, called as `inference.py` at `REF_COMMIT` on the CPU in float32: the
frames zero-padded to a multiple of 64 and centred by upstream `util.pad_batch`, `dt = new_full((1, 1), 0.5)`, the
frame clamped to [0, 1] and cropped. Inputs: the RAFT goldens' pair (tools/golden/raft_flow.py: a 384x512
astronaut.png crop and its affine warp). Stored as float32: `input` (image1, RGB), `input2` (image2, RGB) and
`frame` ([3, H, W] RGB in [0, 1]).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/film.py FILM_PYTORCH_CHECKOUT WEIGHTS_DIR [film ...]

Needs opencv-python (upstream `util.py` imports cv2). Run with `CUDA_VISIBLE_DEVICES=`.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from _common import reference_tree, write
from raft_flow import CROP, IMAGE_SHA256, warp

from fastforks._core import registry

FAMILY = "film"
UPSTREAM = "https://github.com/dajes/frame-interpolation-pytorch"
REF_COMMIT = "33bf477ea2c65a2ed1b5f2280ee1f02be06910e8"


def interpolate(model: torch.jit.ScriptModule, tree: Path, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Upstream's middle frame of an RGB `[H, W, 3]` float32 pair in [0, 1] (as inference.py)."""
    sys.path.insert(0, str(tree))
    from util import pad_batch  # noqa: PLC0415  # reference checkout

    b1, crop = pad_batch(img1[None], 64)
    b2, _ = pad_batch(img2[None], 64)
    x0, x1 = (torch.from_numpy(b).permute(0, 3, 1, 2) for b in (b1, b2))
    with torch.no_grad():
        pred = model(x0, x1, x0.new_full((1, 1), 0.5)).clamp(0, 1)
    y1, x1_, y2, x2 = crop
    return pred[0, :, y1:y2, x1_:x2].numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of dajes/frame-interpolation-pytorch")
    ap.add_argument("weights", type=Path, help="directory with the variants' TorchScript files (card weight names)")
    ap.add_argument("variants", nargs="*", help="variant ids (default: all)")
    a = ap.parse_args()
    if torch.cuda.is_available():
        msg = "run with CUDA_VISIBLE_DEVICES= (the reference runs on the CPU)"
        raise SystemExit(msg)
    import skimage  # noqa: PLC0415

    png = Path(skimage.data.__file__).parent / "astronaut.png"
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's astronaut.png"
        raise SystemExit(msg)
    y0, x0, h, w = CROP
    img1 = np.ascontiguousarray(skimage.io.imread(png)[y0 : y0 + h, x0 : x0 + w, :3])
    img2 = warp(img1)
    f1, f2 = (i.astype(np.float32) / np.float32(255) for i in (img1, img2))  # util.load_image
    tree = reference_tree(a.checkout, REF_COMMIT)
    for card, variant in registry.cards("interpolation").values():
        if card.family != FAMILY or (a.variants and variant.id not in a.variants):
            continue
        model = torch.jit.load(a.weights / variant.weights[0].name, map_location="cpu").eval().float()
        frame = interpolate(model, tree, f1, f2)
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img1.astype(np.float32), "input2": img2.astype(np.float32), "frame": frame.astype(np.float32)},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, the TorchScript release as inference.py, dt 0.5, zero pad to 64 centred "
            f"(util.pad_batch), clamp [0, 1], skimage astronaut.png {h}x{w} crop + affine warp as raft_flow.py, "
            f"torch {torch.__version__}",
            task="interpolation",
        )
        print(f"  {variant.id}: frame mean {frame.mean():.4f}, range [{frame.min():.3f}, {frame.max():.3f}]")


if __name__ == "__main__":
    main()
