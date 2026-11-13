"""RoMa v1 dense-matching goldens from the reference implementation.

Reference: upstream Parskatt/RoMa (`REF_COMMIT`), `romatch.roma_outdoor(...).match` with the release checkpoint and
Meta's DINOv2 ViT-L/14 weights, CPU float32 (upstream turns autocast off on the CPU) and float32 matmul precision
"highest" (upstream requires it); the plain-PyTorch local correlation (`use_custom_corr=False`), one direction
(`symmetric=False`: the first image's half of the default symmetric call). Upstream's tensor path: both images are
resized here (bicubic, antialiased, as the runner) to the coarse and fine squares of the card defaults and normalised
with the ImageNet mean / std (upstream resizes PIL images with torchvision's bicubic `Resize`).
Inputs: the RAFT goldens' pair (tools/golden/raft_flow.py: a 384x512 astronaut.png crop and its affine warp) in [0, 1].
The A-to-B warp and the certainty are resized (bilinear) to the first image's size and the warp converted to pixels of
the second image, as `fastforks.matching` returns them.
Stored as float32: `input` (image1, RGB), `input2` (image2, RGB), `warp` ([2, H, W]: x, y) and `certainty` ([H, W]).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/romav1.py ROMA_CHECKOUT

Upstream's own dependency einops must be importable (loguru is replaced by a stub). The checkout only has to contain
the commit; the reference runs from an export of it.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from _common import reference_tree, write
from raft_flow import CROP, IMAGE_SHA256, warp

from fastforks._core import registry
from fastforks._core.checkpoint import load_state_dict

FAMILY = "romav1"
UPSTREAM = "https://github.com/Parskatt/RoMa"
REF_COMMIT = "77f8d68803526dcddfd9b7a46bc76125bdc25f15"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build(  # noqa: PLR0913 - upstream's options
    tree: Path,
    ckpts: list[Path],
    sizes: tuple[int, int],
    device: str = "cpu",
    *,
    custom_corr: bool = False,
    amp: bool = True,
) -> Any:  # noqa: ANN401 - upstream RegressionMatcher
    """Upstream `roma_outdoor` from the exported tree with the two weight files, one-directional, eval.

    `sizes`: the coarse and fine squares (fine 0: no upsampling pass).
    On CUDA upstream runs its fp16 autocast regions and DINOv2 in fp16 (`amp=False`: `amp_dtype=torch.float32`, the
    float32 original); on the CPU float32.
    """
    if str(tree) not in sys.path:
        sys.path.insert(0, str(tree))
    if "loguru" not in sys.modules:  # upstream logs one line with it
        sys.modules["loguru"] = types.SimpleNamespace(logger=logging.getLogger("romatch"))
    from romatch import roma_outdoor  # noqa: PLC0415 - reference checkout

    torch.set_float32_matmul_precision("highest")  # upstream refuses anything else
    coarse, fine = sizes
    weights, dino = (load_state_dict(p) for p in ckpts)
    model = roma_outdoor(
        device,
        weights=weights,
        dinov2_weights=dino,
        coarse_res=coarse,
        upsample_res=fine or None,
        symmetric=False,
        use_custom_corr=custom_corr,
        upsample_preds=bool(fine),
        amp_dtype=torch.float16 if amp else torch.float32,
    )
    return model.eval()


def prepare(img: torch.Tensor, size: int) -> torch.Tensor:
    """`[B, 3, H, W]` in [0, 1] -> the normalised `size` square (the runner's resize)."""
    out = F.interpolate(img, (size, size), mode="bicubic", align_corners=False, antialias=True)
    mean, std = (torch.tensor(v, device=img.device).view(1, 3, 1, 1) for v in (MEAN, STD))
    return (out - mean) / std


def match(model: Any, img1: torch.Tensor, img2: torch.Tensor, coarse: int, fine: int) -> dict[str, torch.Tensor]:  # noqa: ANN401
    """Upstream `match` on the tensor path, as `fastforks.matching` returns it: image1-sized, warp in image2 pixels."""
    hi = {"im_A_high_res": prepare(img1, fine), "im_B_high_res": prepare(img2, fine)} if fine else {}
    w, c = model.match(prepare(img1, coarse), prepare(img2, coarse), **hi)
    w = w[..., 2:].float().permute(0, 3, 1, 2)
    c = c[:, None].float()
    size1, (h2, w2) = img1.shape[-2:], img2.shape[-2:]
    w, c = (F.interpolate(x, size1, mode="bilinear", align_corners=False) for x in (w, c))
    scale = torch.tensor((w2, h2), dtype=w.dtype, device=w.device).view(1, 2, 1, 1)
    return {"warp": (w + 1) * (scale / 2) - 0.5, "certainty": c[:, 0]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of Parskatt/RoMa")
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
    tree = reference_tree(a.checkout, REF_COMMIT)
    for card, variant in registry.cards("matching").values():
        if card.family != FAMILY:
            continue
        params = dict(card.options)
        coarse, fine = params["coarse_size"], params["fine_size"]
        model = build(tree, list(registry.weights(card, variant)), (coarse, fine))
        with torch.inference_mode():
            out = match(model, t1, t2, coarse, fine)
        maps = {k: v[0].numpy() for k, v in out.items()}
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img1.astype(np.float32), "input2": img2.astype(np.float32)} | maps,
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, upstream roma_outdoor match {params}, one-directional, native local correlation, "
            f"tensor path (bicubic antialiased resize + ImageNet normalisation), skimage astronaut.png {h}x{w} crop "
            f"+ affine warp as raft_flow.py, torch {torch.__version__}",
            task="matching",
        )
        c = maps["certainty"]
        print(
            f"  {variant.id}: certainty mean {c.mean():.3f}, warp mean ({maps['warp'][0].mean():.1f}, "
            f"{maps['warp'][1].mean():.1f})"
        )


if __name__ == "__main__":
    main()
