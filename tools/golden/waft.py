"""WAFT optical-flow goldens from the reference implementation.

Reference: upstream princeton-vl/WAFT (`REF_COMMIT`), `model/waft_a1.py` `ViTWarpV8` built from upstream's
`config/a1/<variant>.json` with the variant's checkpoint loaded strictly (upstream's `load_ckpt` loads with
`strict=False`; here nothing may be missing or unexpected), CPU float32, `iters=5` (the runner's default), the last
iteration's flow, as upstream's evaluate.py runs it (`InferenceWrapper` with `scale=0` and no tiling: the plain call).
Upstream builds Depth Anything V2 with `pretrained=True` (a local depth-anything-ckpts/ file) and the refinement ViT
with `timm.create_model(..., pretrained=True)`; both are overwritten by the checkpoint, so the reference builds them
untrained (nothing is read or downloaded).
Inputs: the RAFT goldens' pair (tools/golden/raft_flow.py: a 384x512 astronaut.png crop and its affine warp), passed
as upstream takes them (RGB in [0, 255]; upstream normalises and pads to multiples of 112 itself).
Stored as float32: `input` (image1, RGB), `input2` (image2, RGB) and `flow` ([2, H, W]: dx, dy in pixels).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/waft.py WAFT_CHECKOUT [waft-a1 ...]

Needs timm, scipy, cv2, torchvision and scikit-image (upstream imports them). The checkout only has to contain
`REF_COMMIT`; the reference runs from an export of that commit.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from _common import reference_tree, write
from raft_flow import CROP, IMAGE_SHA256, warp

from fastforks._core import registry
from fastforks._core.checkpoint import load_state_dict

FAMILY = "waft"
UPSTREAM = "https://github.com/princeton-vl/WAFT"
REF_COMMIT = "b152ff1cad1af8c185ee7b141997c48ff3334c87"
ITERS = 5  # the card's default `iters` (upstream config/a1: 5)
CONFIG = {"waft-a1": "tar-c-t.json", "waft-a1-kitti": "tar-c-t-kitti.json", "waft-a1-sintel": "tar-c-t-sintel.json"}


def build(tree: Path, ckpt: Path, variant: str) -> torch.nn.Module:
    """Upstream `ViTWarpV8` from `config/a1/<variant's config>` with `ckpt` loaded strictly (eval, CPU)."""
    if str(tree) not in sys.path:
        sys.path[:0] = [str(tree)]
        import timm  # noqa: PLC0415 - the reference's own dependency

        vars(timm)["create_model"] = functools.partial(timm.create_model, pretrained=False)  # no ImageNet download
    from model import waft_a1  # noqa: PLC0415 - reference checkout on sys.path

    feature = getattr(waft_a1.DepthAnythingFeature, "func", waft_a1.DepthAnythingFeature)
    waft_a1.DepthAnythingFeature = functools.partial(feature, pretrained=False)  # no depth-anything-ckpts/ read
    cfg = json.loads((tree / "config/a1" / CONFIG[variant]).read_text())
    model = waft_a1.ViTWarpV8(argparse.Namespace(**cfg)).eval()
    model.load_state_dict(load_state_dict(ckpt), strict=True)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of princeton-vl/WAFT")
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
    t1, t2 = (torch.from_numpy(i).permute(2, 0, 1)[None].float() for i in (img1, img2))
    for card, variant in registry.cards("flow").values():
        if card.family != FAMILY or (a.variants and variant.id not in a.variants):
            continue
        (ckpt,) = registry.weights(card, variant)
        model = build(tree, ckpt, variant.id)
        with torch.inference_mode():
            flow = model(t1, t2, iters=ITERS)["flow"][-1][0].numpy()
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img1.astype(np.float32), "input2": img2.astype(np.float32), "flow": flow.astype(np.float32)},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, upstream ViTWarpV8 (config/a1/{CONFIG[variant.id]}) iters={ITERS}, skimage "
            f"astronaut.png {h}x{w} crop + affine warp as raft_flow.py, torch {torch.__version__}",
            task="flow",
        )
        print(f"  {variant.id}: mean flow ({flow[0].mean():.3f}, {flow[1].mean():.3f}), |max| {np.abs(flow).max():.2f}")


if __name__ == "__main__":
    main()
