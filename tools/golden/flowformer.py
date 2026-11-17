"""FlowFormer Small goldens from the authors' pinned implementation.

Inputs are the same textured 384x512 pair as the other flow families. Upstream receives RGB in [0, 255]
and returns its full-resolution flow as tuple element zero. Requires reference dependencies timm==0.4.12,
yacs, einops, loguru and scikit-image, separate from the runtime environment.

    python tools/golden/flowformer.py FLOWFORMER_CHECKOUT [flowformer-small-things ...]
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
from fastforks._core.checkpoint import load_state_dict

FAMILY = "flowformer"
UPSTREAM = "https://github.com/drinkingcoder/FlowFormer-Official"
REF_COMMIT = "6ba7ea82b45394d3a7a25808399695427c9febd8"
ITERS = 32  # upstream Small refinement count


def build(tree: Path, ckpt: Path) -> torch.nn.Module:
    """Pinned upstream Small model and the authors' weights, CPU float32."""
    sys.path[:0] = [str(tree), str(tree / "core")]
    from configs.small_things_eval import get_cfg  # noqa: PLC0415 - pinned upstream
    from core.FlowFormer import build_flowformer  # noqa: PLC0415 - pinned upstream

    cfg = get_cfg()
    cfg.latentcostformer.pretrain = False
    model = build_flowformer(cfg)
    model.load_state_dict(load_state_dict(ckpt), strict=True)
    return model.eval()


def calc_flow(model: torch.nn.Module, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
    """Upstream's full-resolution flow; tuple element one is the low-resolution auxiliary."""
    return model(image1, image2)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of drinkingcoder/FlowFormer-Official")
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
        model = build(tree, ckpt)
        with torch.inference_mode():
            flow = calc_flow(model, t1, t2)[0].numpy()
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img1.astype(np.float32), "input2": img2.astype(np.float32), "flow": flow.astype(np.float32)},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, upstream FlowFormer Small iters={ITERS}, skimage astronaut.png {h}x{w} crop "
            f"+ affine warp as raft_flow.py, torch {torch.__version__}",
            task="flow",
        )
        print(f"  {variant.id}: mean flow ({flow[0].mean():.3f}, {flow[1].mean():.3f}), |max| {np.abs(flow).max():.2f}")


if __name__ == "__main__":
    main()
