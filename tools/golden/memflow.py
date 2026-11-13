"""MemFlow optical-flow goldens from the reference implementation.

Reference: upstream DQiaole/MemFlow (`REF_COMMIT`): `core.Networks.build_network` on the variant's config
(`configs/<stage>_memflownet.py`) with the checkpoint's `model` loaded strictly, CPU float32, eval mode, and one
`inference.inference_core_skflow.InferenceCore.step(images, end=True)` on the pair (upstream's evaluation of the last
pair of a video; with an empty memory the readout attends over the pair's own keys and values), its upsampled flow.
Upstream's non-flash memory readout (flash_attn is not installed). `core/Networks/encoders.py` imports timm for the
Twins-SVT encoders this network does not build: a stub module stands in for it. Inputs: the RAFT goldens' pair
(tools/golden/raft_flow.py: a 384x512 astronaut.png crop and its affine warp; multiples of 8, so upstream's
`InputPadder` adds nothing), normalised as upstream (`2 * (x / 255) - 1`). Stored as float32: `input` (image1, RGB),
`input2` (image2, RGB) and `flow` ([2, H, W]: dx, dy in pixels).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/memflow.py MEMFLOW_CHECKOUT [memflow ...]

Needs yacs, loguru, einops and scipy (upstream imports them). The checkout only has to contain `REF_COMMIT`; the
reference runs from an export of that commit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import torch
from _common import reference_tree, write
from raft_flow import CROP, IMAGE_SHA256, warp

from fastforks._core import registry
from fastforks._core.checkpoint import load_state_dict

FAMILY = "memflow"
UPSTREAM = "https://github.com/DQiaole/MemFlow"
REF_COMMIT = "5486fb88a55b04c2e2b292b3abaa7641d9988831"
CONFIGS = {"memflow": "things_memflownet", "memflow-sintel": "sintel_memflownet", "memflow-kitti": "kitti_memflownet"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of DQiaole/MemFlow")
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
    sys.path[:0] = [str(tree), str(tree / "core")]
    sys.modules.setdefault("timm", types.ModuleType("timm"))  # only the Twins-SVT encoders use it
    from core.Networks import build_network  # noqa: PLC0415  # reference checkout on sys.path
    from inference.inference_core_skflow import InferenceCore  # noqa: PLC0415

    t1, t2 = (torch.from_numpy(i).permute(2, 0, 1).float() for i in (img1, img2))
    images = 2 * (torch.stack([t1, t2])[None] / 255.0) - 1.0
    for card, variant in registry.cards("flow").values():
        if card.family != FAMILY or (a.variants and variant.id not in a.variants):
            continue
        cfg = importlib.import_module(f"configs.{CONFIGS[variant.id]}").get_cfg()
        model = build_network(cfg).eval()
        (ckpt,) = registry.weights(card, variant)
        model.load_state_dict(load_state_dict(ckpt, key="model", strip=""), strict=True)
        with torch.inference_mode():
            _, flow = InferenceCore(model, config=cfg).step(images, end=True)
        flow = flow[0].numpy()
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img1.astype(np.float32), "input2": img2.astype(np.float32), "flow": flow.astype(np.float32)},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, upstream MemFlowNet configs/{CONFIGS[variant.id]}.py, InferenceCore.step(end=True) "
            f"on the pair (val_decoder_depth {cfg.val_decoder_depth}, empty memory, non-flash readout), skimage "
            f"astronaut.png {h}x{w} crop + affine warp as raft_flow.py, torch {torch.__version__}",
            task="flow",
        )
        print(f"  {variant.id}: mean flow ({flow[0].mean():.3f}, {flow[1].mean():.3f}), |max| {np.abs(flow).max():.2f}")


if __name__ == "__main__":
    main()
