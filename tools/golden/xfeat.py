"""XFeat sparse-matching goldens from the reference implementation.

Reference: upstream verlab/accelerated_features (`REF_COMMIT`) `XFeat.detectAndCompute` per image (`top_k` =
`max_keypoints` from the card options), then `XFeat.match(..., min_cossim=-1)` (as `match_xfeat`; score = the
descriptors' cosine similarity) or upstream's kornia `LighterGlue` on the data dict of `match_lighterglue` (min_conf
0.1), the release weights (the card's files), CPU float32. Inputs: the RAFT goldens' pair (tools/golden/raft_flow.py:
a 384x512 astronaut.png crop and its affine warp) in [0, 1]. Stored as float32: `input`, `input2` (RGB),
`keypoints1` `[N, 2]`, `keypoints2` `[M, 2]` (pixels of each input), `pairs` `[K, 2]` (indices) and `scores` `[K]`.

    CUDA_VISIBLE_DEVICES= FASTFORKS_GOLDEN_PATH=... python tools/golden/xfeat.py XFEAT_CHECKOUT

Upstream's own dependency kornia must be importable (LighterGlue). Upstream picks CUDA when it sees a GPU: hide it.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from _common import reference_tree, write
from raft_flow import CROP, IMAGE_SHA256, warp

from fastforks._core import registry
from fastforks.shared.lightglue import resize_long

if TYPE_CHECKING:
    from collections.abc import Callable

FAMILY = "xfeat"
UPSTREAM = "https://github.com/verlab/accelerated_features"
REF_COMMIT = "e92685f57f8318b18725c5c8c0bd28c7fe188d9a"


def build(
    files: dict[str, Path], matcher: str, max_keypoints: int, long_side: int = 0
) -> Callable[[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
    """Upstream XFeat (+ LighterGlue) on upstream's device: `[3, H, W]` pair in [0, 1] -> keypoints, pairs, scores.

    The reference checkout must be on `sys.path`. `long_side > 0` resizes as the fastforks runner does (upstream XFeat
    does not resize) and maps the keypoints back.
    """
    from modules.xfeat import XFeat  # noqa: PLC0415  # reference checkout

    xf = XFeat(weights=str(files["xfeat.pt"]), top_k=max_keypoints)
    if matcher == "lighterglue":
        from modules.lighterglue import LighterGlue  # noqa: PLC0415

        xf.lighterglue = LighterGlue(weights=str(files["xfeat-lighterglue.pt"]))

    def features(image: torch.Tensor) -> dict:
        x = resize_long(image[None], long_side)[0] if long_side > 0 else image
        return xf.detectAndCompute(x[None])[0] | {"image_size": x.shape[-1:-3:-1]}  # notebook usage: (width, height)

    def to_input(d: dict, image: torch.Tensor) -> torch.Tensor:  # back to input pixels after matching
        (w, h), k = image.shape[-1:-3:-1], d["keypoints"]
        scale = torch.tensor([d["image_size"][0] / w, d["image_size"][1] / h], device=k.device)
        return k if long_side <= 0 else (k + 0.5) / scale - 0.5

    @torch.inference_mode()
    def run(image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        d0, d1 = features(image1), features(image2)
        if matcher == "mnn":
            idx0, idx1 = xf.match(d0["descriptors"], d1["descriptors"], min_cossim=-1)
            pairs = torch.stack([idx0, idx1], dim=-1)
            scores = (d0["descriptors"][idx0] * d1["descriptors"][idx1]).sum(-1)
        else:  # upstream match_lighterglue's data dict, with the scores it drops
            data = {
                "keypoints0": d0["keypoints"][None, ...],
                "keypoints1": d1["keypoints"][None, ...],
                "descriptors0": d0["descriptors"][None, ...],
                "descriptors1": d1["descriptors"][None, ...],
                "image_size0": torch.tensor(d0["image_size"]).to(xf.dev)[None, ...],
                "image_size1": torch.tensor(d1["image_size"]).to(xf.dev)[None, ...],
            }
            out = xf.lighterglue(data, min_conf=0.1)
            pairs, scores = out["matches"][0], out["scores"][0]
        k1, k2 = to_input(d0, image1), to_input(d1, image2)
        return {"keypoints1": k1, "keypoints2": k2, "pairs": pairs, "scores": scores}

    return run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of verlab/accelerated_features")
    a = ap.parse_args()
    import skimage  # noqa: PLC0415

    png = Path(skimage.data.__file__).parent / "astronaut.png"
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's astronaut.png"
        raise SystemExit(msg)
    y0, x0, h, w = CROP
    img1 = np.ascontiguousarray(skimage.io.imread(png)[y0 : y0 + h, x0 : x0 + w, :3])
    img2 = warp(img1)
    t1, t2 = (torch.from_numpy(i).permute(2, 0, 1).float() / 255 for i in (img1, img2))
    sys.path.insert(0, str(reference_tree(a.checkout, REF_COMMIT)))
    for card, variant in registry.cards("matching").values():
        if card.family != FAMILY:
            continue
        files = {f.name: f for f in registry.weights(card, variant)}
        params, opts = dict(variant.params), dict(card.options)
        run = build(files, params["matcher"], opts["max_keypoints"], opts["long_side"])
        maps = {k: v.cpu().numpy() for k, v in run(t1, t2).items()}
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img1.astype(np.float32), "input2": img2.astype(np.float32)} | maps,
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, upstream XFeat.detectAndCompute top_k {opts['max_keypoints']} + "
            f"{'match(min_cossim=-1)' if params['matcher'] == 'mnn' else 'kornia LighterGlue min_conf 0.1'}, skimage "
            f"astronaut.png {h}x{w} crop + affine warp as raft_flow.py, torch {torch.__version__}",
            task="matching",
        )
        n1, n2, k = (len(maps[x]) for x in ("keypoints1", "keypoints2", "pairs"))
        print(f"  {variant.id}: {n1} / {n2} keypoints, {k} matches")


if __name__ == "__main__":
    main()
