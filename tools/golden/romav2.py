"""RoMa v2 dense-matching goldens from the reference implementation.

Reference: upstream Parskatt/RoMaV2 (`REF_COMMIT`), `romav2.RoMaV2.match` with the release checkpoint, CPU float32:
`enable_amp=False` for the descriptor, `torch.autocast` replaced by a no-op (upstream's DPT head, VGG features and
refiner blocks enter it unconditionally) and float32 matmul precision "highest". The descriptor is the pinned DINOv3
hub checkout (`DINOV3_COMMIT`, `torch.hub.load(..., source="local", pretrained=False)`; its weights come from the
RoMa v2 checkpoint). Setting "precise" (800 / 1280, the card defaults) one-directional (`bidirectional=False`: the
first image's outputs are those of the bidirectional call) without threshold.
Inputs: the RAFT goldens' pair (tools/golden/raft_flow.py: a 384x512 astronaut.png crop and its affine warp) in [0, 1].
`warp_AB` and the overlap probability `overlap_AB` are resized (bilinear) to the first image's size and the warp
converted to pixels of the second image, as `fastforks.matching` returns them.
Stored as float32: `input` (image1, RGB), `input2` (image2, RGB), `warp` ([2, H, W]: x, y) and `certainty` ([H, W]).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/romav2.py ROMAV2_CHECKOUT DINOV3_CHECKOUT

Upstream's own dependencies (einops) must be importable. The checkouts only have to contain the commits; the
reference runs from exports of them.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from _common import reference_tree, write
from raft_flow import CROP, IMAGE_SHA256, warp

from fastforks._core import registry
from fastforks._core.checkpoint import load_state_dict

FAMILY = "romav2"
UPSTREAM = "https://github.com/Parskatt/RoMaV2"
REF_COMMIT = "95c9968145c8906b7b59383258e9f73b02853d89"
DINOV3_COMMIT = "adc254450203739c8149213a7a69d8d905b4fcfa"  # facebookresearch/dinov3, the descriptor's hub code


def build(tree: Path, dinov3: Path, ckpt: Path, *, amp: bool = False) -> Any:  # noqa: ANN401 - upstream RoMaV2
    """Upstream `RoMaV2` from the exported trees with the checkpoint, eval; float32 unless `amp` (upstream's bf16)."""
    src = str(tree / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    sd = load_state_dict(ckpt)
    hub_load = torch.hub.load

    def local(*_a: Any, model: str, **_kw: Any) -> Any:  # noqa: ANN401 - the pinned hub code, no download
        return hub_load(str(dinov3), model, source="local", pretrained=False)

    torch.hub.load = local
    torch.hub.load_state_dict_from_url = lambda *_a, **_kw: sd
    if not amp:
        torch.autocast = lambda *_a, **_kw: contextlib.nullcontext()
    from romav2.features import Descriptor  # noqa: PLC0415  # reference checkout
    from romav2.romav2 import RoMaV2  # noqa: PLC0415

    torch.set_float32_matmul_precision("highest")  # upstream's forward requires it
    model = RoMaV2(RoMaV2.Cfg(descriptor=Descriptor.Cfg(enable_amp=amp), setting="precise"))
    torch.hub.load = hub_load
    return model.eval()


def to_api(preds: dict[str, torch.Tensor], size1: tuple[int, int], size2: tuple[int, int]) -> dict[str, torch.Tensor]:
    """Upstream `match` outputs as `fastforks.matching` returns them: image1-sized, warp in image2 pixels."""
    w = preds["warp_AB"].float().permute(0, 3, 1, 2)
    c = preds["overlap_AB"].float().permute(0, 3, 1, 2)
    w, c = (F.interpolate(x, size1, mode="bilinear", align_corners=False) for x in (w, c))
    scale = torch.tensor((size2[1], size2[0]), dtype=w.dtype, device=w.device).view(1, 2, 1, 1)
    return {"warp": (w + 1) * (scale / 2) - 0.5, "certainty": c[:, 0]}


def configure(model: Any, coarse: int, fine: int) -> None:  # noqa: ANN401 - upstream RoMaV2
    """Sizes of the card options, one direction, no threshold."""
    model.H_lr = model.W_lr = coarse
    model.H_hr = model.W_hr = fine or None
    model.bidirectional, model.threshold = False, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of Parskatt/RoMaV2")
    ap.add_argument("dinov3", type=Path, help="git checkout of facebookresearch/dinov3")
    a = ap.parse_args()
    import skimage  # noqa: PLC0415

    png = Path(skimage.data.__file__).parent / "astronaut.png"
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's astronaut.png"
        raise SystemExit(msg)
    y0, x0, h, w = CROP
    img1 = np.ascontiguousarray(skimage.io.imread(png)[y0 : y0 + h, x0 : x0 + w, :3])
    img2 = warp(img1)
    t1, t2 = (torch.from_numpy(i).permute(2, 0, 1)[None].float() / 255 for i in (img1, img2))
    tree, dino = reference_tree(a.checkout, REF_COMMIT), reference_tree(a.dinov3, DINOV3_COMMIT)
    for card, variant in registry.cards("matching").values():
        if card.family != FAMILY:
            continue
        (ckpt,) = registry.weights(card, variant)
        model = build(tree, dino, ckpt)
        params = dict(card.options)
        configure(model, params["coarse_size"], params["fine_size"])
        with torch.inference_mode():
            out = to_api(model.match(t1, t2), (h, w), (h, w))
        maps = {k: v[0].numpy() for k, v in out.items()}
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img1.astype(np.float32), "input2": img2.astype(np.float32)} | maps,
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, upstream RoMaV2.match {params}, one-directional, no threshold, autocast off, "
            f"DINOv3 hub code {DINOV3_COMMIT}, skimage astronaut.png {h}x{w} crop + affine warp as raft_flow.py, "
            f"torch {torch.__version__}",
            task="matching",
        )
        c = maps["certainty"]
        print(
            f"  {variant.id}: certainty mean {c.mean():.3f}, warp mean ({maps['warp'][0].mean():.1f}, "
            f"{maps['warp'][1].mean():.1f})"
        )


if __name__ == "__main__":
    main()
