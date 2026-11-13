"""MegaFlow optical-flow goldens from the reference implementation.

Reference: upstream cvg/megaflow (`REF_COMMIT`), `megaflow.model.MegaFlow` as `from_pretrained` builds it
(fuse_cnn, temporal attention, fix_width) with the checkpoint loaded strictly, CPU float32, eval mode, the pair as a
two-frame clip, `forward(..., num_reg_refine=ITERS)["flow_preds"][-1]`. Upstream's ResNet FPN loads torchvision's
ImageNet ResNet34 at construction (overwritten by the checkpoint); `build` stubs it with an untrained one, so nothing
is downloaded. Inputs: the RAFT goldens' pair (tools/golden/raft_flow.py: a 384x512 astronaut.png crop and its affine
warp), passed as upstream takes them (RGB in [0, 255]). Stored as float32: `input` (image1, RGB), `input2` (image2,
RGB) and `flow` ([2, H, W]: dx, dy in pixels).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/megaflow.py MEGAFLOW_CHECKOUT [megaflow ...]

Needs torchvision and huggingface_hub (upstream imports them). ~12 GB RAM. The checkout only has to contain
`REF_COMMIT`; the reference runs from an export of that commit.
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

FAMILY = "megaflow"
UPSTREAM = "https://github.com/cvg/megaflow"
REF_COMMIT = "2328f8bb26901171c17e7e147413f9ed8c563b1a"
ITERS = 8  # the card's default `iters` (upstream README and demo_flow.py)


def build(tree: Path, ckpt: Path) -> torch.nn.Module:
    """Upstream `MegaFlow` (from the exported `tree`) with the checkpoint, float32, eval, no torchvision download."""
    import torchvision.models  # noqa: PLC0415

    if str(tree) not in sys.path:
        sys.path.insert(0, str(tree))
    from megaflow.model import MegaFlow  # noqa: PLC0415  # ty: ignore[unresolved-import]

    real = torchvision.models.resnet34
    torchvision.models.resnet34 = lambda **_: real(weights=None)  # ty: ignore[invalid-assignment]
    try:
        net = MegaFlow(fuse_cnn=True, use_temporal_attn=True, fix_width=True)
    finally:
        torchvision.models.resnet34 = real
    net.load_state_dict(load_state_dict(ckpt), strict=True)
    return net.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of cvg/megaflow")
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
    frames = torch.stack([torch.from_numpy(i).permute(2, 0, 1).float() for i in (img1, img2)])[None]
    for card, variant in registry.cards("flow").values():
        if card.family != FAMILY or (a.variants and variant.id not in a.variants):
            continue
        model = build(tree, registry.weights(card, variant)[0])
        with torch.inference_mode():
            flow = model(frames, num_reg_refine=ITERS)["flow_preds"][-1][0, 0].numpy()
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img1.astype(np.float32), "input2": img2.astype(np.float32), "flow": flow.astype(np.float32)},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, upstream MegaFlow num_reg_refine={ITERS}, the pair as a two-frame clip, "
            f"skimage astronaut.png {h}x{w} crop + affine warp as raft_flow.py, torch {torch.__version__}",
            task="flow",
        )
        print(f"  {variant.id}: mean flow ({flow[0].mean():.3f}, {flow[1].mean():.3f}), |max| {np.abs(flow).max():.2f}")


if __name__ == "__main__":
    main()
