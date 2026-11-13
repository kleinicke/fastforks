"""RIFE frame-interpolation goldens from the reference implementation.

Reference: upstream hzwer/Practical-RIFE (`REF_COMMIT`, for `model/`) with each release's own `train_log/`
(`IFNet_HDv3.py`, `RIFE_HDv3.py`, `flownet.pkl` from the variant's archive): `Model.load_model(dir, -1)` and
`Model.inference(I0, I1)` (time 0.5, `scale` 1.0) on the CPU in float32, the inputs zero-padded at the right and bottom
to a multiple of 128 as `inference_video.py` does, the output cropped back.
Inputs: the RAFT goldens' pair (tools/golden/raft_flow.py: a 384x512 astronaut.png crop and its affine warp).
Stored as float32: `input` (image1, RGB), `input2` (image2, RGB) and `frame` ([3, H, W] in [0, 1]).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/rife.py PRACTICAL_RIFE_CHECKOUT RELEASES [rife-4.25 ...]

RELEASES holds one unpacked release archive per variant id (`RELEASES/rife-4.25/train_log/...`). Needs torchvision
(upstream's `model/loss.py` imports it). Run with `CUDA_VISIBLE_DEVICES=` (upstream moves the model to CUDA if it can).
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from _common import reference_tree, write
from raft_flow import CROP, IMAGE_SHA256, warp

from fastforks._core import registry

FAMILY = "rife"
UPSTREAM = "https://github.com/hzwer/Practical-RIFE"
REF_COMMIT = "bbfd2ea90910789a860ea3e2b32a240cd577b75e"
PAD = 128  # inference_video.py at scale 1.0: max(128, int(128 / scale))


def build(tree: Path, release: Path) -> object:
    """Upstream's `Model` with the release in `release/train_log` loaded (`load_model(dir, -1)`), in eval mode."""
    shutil.copytree(release / "train_log", tree / "train_log", dirs_exist_ok=True)
    for name in [m for m in sys.modules if m.split(".")[0] in {"train_log", "model"}]:
        del sys.modules[name]  # each release has its own IFNet_HDv3
    sys.path.insert(0, str(tree))
    try:
        from train_log.RIFE_HDv3 import Model  # noqa: PLC0415  # reference checkout

        model = Model()
        model.load_model(str(tree / "train_log"), -1)
        model.eval()
        return model
    finally:
        sys.path.remove(str(tree))


def interpolate(model: object, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
    """Upstream's middle frame of a `[1, 3, H, W]` pair in [0, 1], padded and cropped as `inference_video.py`."""
    h, w = image1.shape[-2:]
    pad = (0, -w % PAD, 0, -h % PAD)
    with torch.inference_mode():
        return model.inference(F.pad(image1, pad), F.pad(image2, pad))[..., :h, :w]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of hzwer/Practical-RIFE")
    ap.add_argument("releases", type=Path, help="directory of unpacked release archives, one per variant id")
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
    t1, t2 = (torch.from_numpy(i).permute(2, 0, 1)[None].float() / 255 for i in (img1, img2))
    for card, variant in registry.cards("interpolation").values():
        if card.family != FAMILY or (a.variants and variant.id not in a.variants):
            continue
        tree = reference_tree(a.checkout, REF_COMMIT)
        frame = interpolate(build(tree, a.releases / variant.id), t1, t2)[0].numpy()
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img1.astype(np.float32), "input2": img2.astype(np.float32), "frame": frame.astype(np.float32)},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, upstream RIFE_HDv3 Model.inference timestep 0.5 scale 1.0, zero pad to {PAD} "
            f"right / bottom, skimage astronaut.png {h}x{w} crop + affine warp as raft_flow.py, "
            f"torch {torch.__version__}",
            task="interpolation",
        )
        print(f"  {variant.id}: frame mean {frame.mean():.4f}, range [{frame.min():.3f}, {frame.max():.3f}]")


if __name__ == "__main__":
    main()
