"""NeuFlow v2 optical-flow goldens from the reference implementation.

Reference: upstream neufieldrobotics/NeuFlow_v2 (`REF_COMMIT`), `NeuFlow/neuflow.py` `NeuFlow` with the variant's
checkpoint loaded strictly, CPU float32 (`init_bhwd(..., amp=False)`), upstream's default iterations (`iters_s16=1`,
`iters_s8=8` = the card's default `iters`), the last entry of the returned flow list.
Inputs: the RAFT goldens' pair (tools/golden/raft_flow.py: a 384x512 astronaut.png crop and its affine warp), passed
as upstream takes them (RGB in [0, 255]; upstream divides by 255; 384x512 needs no padding).
Stored as float32: `input` (image1, RGB), `input2` (image2, RGB) and `flow` ([2, H, W]: dx, dy in pixels).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/neuflow_v2.py NEUFLOW_V2_CHECKOUT [neuflow-v2 ...]

Needs huggingface_hub (upstream imports it). The checkout only has to contain `REF_COMMIT`; the reference runs from
an export of that commit.
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

FAMILY = "neuflow_v2"
UPSTREAM = "https://github.com/neufieldrobotics/NeuFlow_v2"
REF_COMMIT = "204b5e3744461d90303b9ff82caa7a1bb56a2ca2"
ITERS = 8  # the card's default `iters` (upstream `iters_s8`)


def build(tree: Path, ckpt: Path) -> torch.nn.Module:
    """Upstream `NeuFlow` (from the exported `tree`, on `sys.path`) with the checkpoint, float32, eval."""
    if str(tree) not in sys.path:
        sys.path.insert(0, str(tree))
    from NeuFlow.neuflow import NeuFlow  # noqa: PLC0415  # reference checkout

    model = NeuFlow()
    model.load_state_dict(load_state_dict(ckpt, key="model"), strict=True)
    return model.eval()


def calc_flow(model: torch.nn.Module, image1: torch.Tensor, image2: torch.Tensor, *, amp: bool = False) -> torch.Tensor:
    """Upstream's call on a `[B, 3, H, W]` pair in [0, 255] (H, W multiples of 16): `init_bhwd`, the last flow."""
    b, _, h, w = image1.shape
    key = (b, h, w, image1.device, amp)
    if getattr(model, "bhwd", None) != key:  # as upstream infer.py: once per size
        model.init_bhwd(*key)  # ty: ignore[call-non-callable]
        model.bhwd = key
    return model(image1.clone(), image2.clone(), iters_s8=ITERS)[-1]  # upstream divides its inputs in place


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of neufieldrobotics/NeuFlow_v2")
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
            settings=f"CPU float32, upstream NeuFlow iters_s16=1 iters_s8={ITERS}, skimage astronaut.png {h}x{w} crop "
            f"+ affine warp as raft_flow.py, torch {torch.__version__}",
            task="flow",
        )
        print(f"  {variant.id}: mean flow ({flow[0].mean():.3f}, {flow[1].mean():.3f}), |max| {np.abs(flow).max():.2f}")


if __name__ == "__main__":
    main()
