"""ReCoVEr optical-flow goldens from the reference implementation.

Reference: upstream visinf/ReCoVEr (`REF_COMMIT`), `models/recover.py` `ReCoVEr_{CX,RN,MN}` (default args, as its
demo.py builds them) with the variant's checkpoint loaded strictly, CPU float32, `iters=4` (upstream's default and
the runner's), `test_mode=True`, the `final` prediction (cost volume disabled, upstream's default).
Upstream's encoders load torchvision's ImageNet weights at construction (overwritten by the checkpoint);
`_no_imagenet` builds them with `weights=None` instead, so nothing is downloaded.
Inputs: the RAFT goldens' pair (tools/golden/raft_flow.py: a 384x512 astronaut.png crop and its affine warp), passed
as upstream takes them (RGB in [0, 255]; upstream scales to [-1, 1] and pads itself; 384x512 needs no padding).
Stored as float32: `input` (image1, RGB), `input2` (image2, RGB) and `flow` ([2, H, W]: dx, dy in pixels).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/recover.py RECOVER_CHECKOUT [recover-cx ...]

Needs torchvision (upstream imports it). The checkout only has to contain `REF_COMMIT`; the reference runs from an
export of that commit.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.models
from _common import reference_tree, write
from raft_flow import CROP, IMAGE_SHA256, warp

from fastforks._core import registry
from fastforks._core.checkpoint import load_state_dict

FAMILY = "recover"
UPSTREAM = "https://github.com/visinf/ReCoVEr"
REF_COMMIT = "503a27e38bad1080bd29da91df74c5523e11f8d8"
ITERS = 4  # the card's default `iters` (upstream's default args)
CLASSES = {"cx": "ReCoVEr_CX", "rn": "ReCoVEr_RN", "mn": "ReCoVEr_MN"}


def _no_imagenet() -> None:
    """Make upstream's torchvision builders (`weights=...IMAGENET1K_V1`) build untrained networks (no download)."""
    for name in ("resnet34", "mobilenet_v3_large", "convnext_tiny"):
        build = getattr(torchvision.models, name)
        setattr(torchvision.models, name, lambda weights=None, _build=build: _build(weights=None))  # noqa: ARG005


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of visinf/ReCoVEr")
    ap.add_argument("variants", nargs="*", help="variant ids (default: all)")
    a = ap.parse_args()
    import skimage  # noqa: PLC0415

    png = Path(skimage.data.__file__).parent / "astronaut.png"
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's astronaut.png"
        raise SystemExit(msg)
    y0, x0, h, w = CROP
    img1 = np.ascontiguousarray(skimage.io.imread(png)[y0 : y0 + h, x0 : x0 + w, :3])
    img2 = warp(img1)
    tree = reference_tree(a.checkout, REF_COMMIT)
    sys.path[:0] = [str(tree)]
    _no_imagenet()
    from models import recover  # noqa: PLC0415  # reference checkout on sys.path

    t1, t2 = (torch.from_numpy(i).permute(2, 0, 1)[None].float() for i in (img1, img2))
    for card, variant in registry.cards("flow").values():
        if card.family != FAMILY or (a.variants and variant.id not in a.variants):
            continue
        arch = variant.params.get("arch", card.params["arch"])
        model = getattr(recover, CLASSES[arch])(pretrained=False).eval()
        (ckpt,) = registry.weights(card, variant)
        model.load_state_dict(load_state_dict(ckpt), strict=True)
        with torch.inference_mode():
            flow = model(t1, t2, iters=ITERS, test_mode=True)["final"][0].numpy()
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img1.astype(np.float32), "input2": img2.astype(np.float32), "flow": flow.astype(np.float32)},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, upstream {CLASSES[arch]} iters={ITERS} test_mode, skimage astronaut.png {h}x{w} "
            f"crop + affine warp as raft_flow.py, torch {torch.__version__}",
            task="flow",
        )
        print(f"  {variant.id}: mean flow ({flow[0].mean():.3f}, {flow[1].mean():.3f}), |max| {np.abs(flow).max():.2f}")


if __name__ == "__main__":
    main()
