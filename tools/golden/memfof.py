"""MEMFOF optical-flow goldens from the reference implementation.

Reference: upstream msu-video-group/memfof (`REF_COMMIT`), `memfof/model.py` `MEMFOF` (its defaults: resnet34,
dim 512, 4 levels, radius 4, 2 blocks) with the variant's checkpoint loaded strictly, CPU float32, eval mode,
`iters=ITERS`. The frames are upstream's first frame of a clip, (image1, image1, image2), run as its
`MEMFOFLit.scale_and_forward_flow` runs them (frames resized by 2^scale, the forward flow `flow[-1][:, 1]` resized back
and divided by 2^scale; `scale` from the card). Upstream's encoders load torchvision's ImageNet ResNet weights at
construction (overwritten by the checkpoint); `backbone_weights=None` builds them untrained, so nothing is downloaded.
Inputs: the RAFT goldens' pair (tools/golden/raft_flow.py: a 384x512 astronaut.png crop and its affine warp), passed
as upstream takes them (RGB in [0, 255]). Stored as float32: `input` (image1, RGB), `input2` (image2, RGB) and `flow`
([2, H, W]: dx, dy in pixels).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/memfof.py MEMFOF_CHECKOUT [memfof ...]

Needs torchvision, einops and huggingface_hub (upstream imports them). The checkout only has to contain
`REF_COMMIT`; the reference runs from an export of that commit.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from _common import reference_tree, write
from raft_flow import CROP, IMAGE_SHA256, warp

from fastforks._core import registry
from fastforks._core.checkpoint import load_state_dict

FAMILY = "memfof"
UPSTREAM = "https://github.com/msu-video-group/memfof"
REF_COMMIT = "a51de9fc59c6fe20ba08e079372c7b583d58a712"
ITERS = 8  # the card's default `iters` (upstream demo and config/eval: 8)


def scale_and_forward_flow(model: torch.nn.Module, images: torch.Tensor, scale: int) -> torch.Tensor:
    """Upstream `MEMFOFLit.scale_and_forward_flow` (its flow output) for `[B, 3, 3, H, W]` frames."""
    imgs = torch.stack(
        [F.interpolate(images[:, i], scale_factor=2**scale, mode="bilinear", align_corners=False) for i in range(3)],
        dim=1,
    )
    flow = model(imgs, iters=ITERS)["flow"][-1][:, 1]
    return F.interpolate(flow, scale_factor=0.5**scale, mode="bilinear", align_corners=False) * (0.5**scale)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of msu-video-group/memfof")
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
    from memfof.model import MEMFOF  # noqa: PLC0415  # reference checkout on sys.path

    t1, t2 = (torch.from_numpy(i).permute(2, 0, 1)[None].float() for i in (img1, img2))
    frames = torch.stack([t1, t1, t2], dim=1)
    for card, variant in registry.cards("flow").values():
        if card.family != FAMILY or (a.variants and variant.id not in a.variants):
            continue
        scale = variant.params.get("scale", card.params["scale"])
        model = MEMFOF(backbone_weights=None).eval()
        (ckpt,) = registry.weights(card, variant)
        model.load_state_dict(load_state_dict(ckpt), strict=True)
        with torch.inference_mode():
            flow = scale_and_forward_flow(model, frames, scale)[0].numpy()
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img1.astype(np.float32), "input2": img2.astype(np.float32), "flow": flow.astype(np.float32)},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, upstream MEMFOF iters={ITERS}, frames (image1, image1, image2), forward flow, "
            f"scale_and_forward_flow scale={scale}, skimage astronaut.png {h}x{w} crop + affine warp as raft_flow.py, "
            f"torch {torch.__version__}",
            task="flow",
        )
        print(f"  {variant.id}: mean flow ({flow[0].mean():.3f}, {flow[1].mean():.3f}), |max| {np.abs(flow).max():.2f}")


if __name__ == "__main__":
    main()
