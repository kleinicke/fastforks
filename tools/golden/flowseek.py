"""FlowSeek optical-flow goldens from the reference implementation.

Reference: upstream mattpoggi/flowseek (`REF_COMMIT`), `core/flowseek.py` `FlowSeek` built from upstream's
`config/eval/flowseek-T.json` with the variant's checkpoint loaded strictly, CPU float32, `iters=4` (the config's
and the runner's default), `test_mode=True`, the `final` prediction.
Upstream is written for CUDA and for a prepared directory; `build` runs it unchanged around that:
- its constructor reads `weights/depth_anything_v2_vits.pth` (overwritten by the checkpoint, which holds the same
  network as `dav2.*`): that file is written into the exported tree from the checkpoint's `dav2.*` tensors;
- its encoders load torchvision's ImageNet ResNet weights at construction (overwritten by the checkpoint):
  `_no_imagenet` builds them with `weights=None` instead, so nothing is downloaded;
- it calls `.cuda()` on the Depth Anything network and on constants: on the CPU (`cpu()`), `.cuda()` returns the
  object itself.
Inputs: the RAFT goldens' pair (tools/golden/raft_flow.py: a 384x512 astronaut.png crop and its affine warp), passed
as upstream takes them (RGB in [0, 255]; upstream scales and pads itself; 384x512 needs no padding).
Stored as float32: `input` (image1, RGB), `input2` (image2, RGB) and `flow` ([2, H, W]: dx, dy in pixels).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/flowseek.py FLOWSEEK_CHECKOUT [flowseek-t ...]

Needs torchvision, huggingface_hub and cv2 (upstream imports them). The checkout only has to contain `REF_COMMIT`;
the reference runs from an export of that commit.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from _common import reference_tree, write
from raft_flow import CROP, IMAGE_SHA256, warp

from fastforks._core import registry
from fastforks._core.checkpoint import load_state_dict

if TYPE_CHECKING:
    from collections.abc import Iterator

FAMILY = "flowseek"
UPSTREAM = "https://github.com/mattpoggi/flowseek"
REF_COMMIT = "ba6407370bceeb96244cad4d60d37489f6eb471e"
ITERS = 4  # the card's default `iters` (upstream config/eval/flowseek-T.json: 4)
CONFIG = "flowseek-T.json"


def _no_imagenet() -> None:
    """Make upstream's `resnet18/34(weights=...IMAGENET1K_V1)` build an untrained ResNet (no download)."""
    import torchvision.models  # noqa: PLC0415

    for name in ("resnet18", "resnet34"):
        make = getattr(torchvision.models, name)
        if getattr(make, "ff_no_imagenet", False):
            continue
        new = lambda weights=None, _make=make: _make(weights=None)  # noqa: ARG005, E731
        new.ff_no_imagenet = True
        setattr(torchvision.models, name, new)


@contextlib.contextmanager
def cpu() -> Iterator[None]:
    """Run upstream on the CPU: while active, `.cuda()` of modules and tensors returns the object itself."""
    saved = torch.nn.Module.cuda, torch.Tensor.cuda
    torch.nn.Module.cuda = torch.Tensor.cuda = lambda self, *_a, **_k: self
    try:
        yield
    finally:
        torch.nn.Module.cuda, torch.Tensor.cuda = saved


def build(tree: Path, ckpt: Path) -> torch.nn.Module:
    """Upstream `FlowSeek` (size T, from the exported `tree`) with the checkpoint loaded strictly, float32, eval."""
    for p in (str(tree), str(tree / "core")):
        if p not in sys.path:
            sys.path.insert(0, p)
    _no_imagenet()
    # upstream's core/depth_anything_v2/ has no __init__.py; as a namespace package it would lose against
    # tools/golden/depth_anything_v2.py on sys.path
    (tree / "core/depth_anything_v2/__init__.py").touch()
    # by path: this tool has the name of upstream's module
    spec = importlib.util.spec_from_file_location("flowseek_upstream", tree / "core/flowseek.py")
    assert spec is not None and spec.loader is not None  # noqa: S101, PT018
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)

    sd = load_state_dict(ckpt)
    (tree / "weights").mkdir(exist_ok=True)
    torch.save({k[5:]: v for k, v in sd.items() if k.startswith("dav2.")}, tree / "weights/depth_anything_v2_vits.pth")
    cfg = json.loads((tree / "config/eval" / CONFIG).read_text())
    with contextlib.chdir(tree):  # upstream opens weights/depth_anything_v2_vits.pth relative to the working dir
        model = upstream.FlowSeek(argparse.Namespace(**cfg))
    model.load_state_dict(sd, strict=True)
    return model.eval()


def calc_flow(model: torch.nn.Module, image1: torch.Tensor, image2: torch.Tensor, iters: int = ITERS) -> torch.Tensor:
    """Upstream's flow of a `[B, 3, H, W]` pair in [0, 255] (any size: upstream pads itself)."""
    return model(image1, image2, iters=iters, test_mode=True)["final"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of mattpoggi/flowseek")
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
        with cpu(), torch.inference_mode():
            flow = calc_flow(build(tree, ckpt), t1, t2)[0].numpy()
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img1.astype(np.float32), "input2": img2.astype(np.float32), "flow": flow.astype(np.float32)},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32 (.cuda() calls made no-ops), upstream FlowSeek ({CONFIG}) iters={ITERS}, "
            f"skimage astronaut.png {h}x{w} crop + affine warp as raft_flow.py, torch {torch.__version__}",
            task="flow",
        )
        print(f"  {variant.id}: mean flow ({flow[0].mean():.3f}, {flow[1].mean():.3f}), |max| {np.abs(flow).max():.2f}")


if __name__ == "__main__":
    main()
