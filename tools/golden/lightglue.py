"""LightGlue sparse-matching goldens from the reference implementation.

Reference: upstream cvg/LightGlue (`REF_COMMIT`) `match_pair` with `ALIKED` / `DISK` (kornia) and `LightGlue` at their
defaults, `max_num_keypoints` and `resize` from the card options, the release weights (the card's files; upstream's
hub downloads are answered from them), CPU float32. Inputs: the RAFT goldens' pair (tools/golden/raft_flow.py: a
384x512 astronaut.png crop and its affine warp) in [0, 1]. Stored as float32: `input`, `input2` (RGB), `keypoints1`
`[N, 2]`, `keypoints2` `[M, 2]` (pixels of each input), `pairs` `[K, 2]` (indices, as float) and `scores` `[K]`.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/lightglue.py LIGHTGLUE_CHECKOUT

Upstream's own dependencies (kornia, opencv) must be importable. The checkout only has to contain the commit.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from _common import reference_tree, write
from raft_flow import CROP, IMAGE_SHA256, warp

from fastforks._core import registry

FAMILY = "lightglue"
UPSTREAM = "https://github.com/cvg/LightGlue"
REF_COMMIT = "5a650c444718caffc2669dc0094cde908f42dd73"


def from_files(files: dict[str, Path], url: str, **_kw: object) -> dict:
    """Upstream's hub download answered from the card's (hash-checked) file of the same name."""
    return torch.load(files[url.rsplit("/", 1)[1]], map_location="cpu", weights_only=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of cvg/LightGlue")
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
    from lightglue import ALIKED, DISK, LightGlue  # noqa: PLC0415  # reference checkout
    from lightglue.utils import match_pair  # noqa: PLC0415

    for card, variant in registry.cards("matching").values():
        if card.family != FAMILY:
            continue
        files = {f.name: f for f in registry.weights(card, variant)}
        # upstream downloads by URL: answer from the card's (hash-checked) files, matched by file name
        torch.hub.load_state_dict_from_url = functools.partial(from_files, files)
        params, opts = dict(variant.params), dict(card.options)
        extractor = {"aliked": ALIKED, "disk": DISK}[params["extractor"]]
        ex = extractor(max_num_keypoints=opts["max_keypoints"]).eval()
        lg = LightGlue(features=params["extractor"]).eval()
        with torch.inference_mode():
            f1, f2, m = match_pair(ex, lg, t1, t2, resize=opts["long_side"])
        maps = {
            "keypoints1": f1["keypoints"].numpy(),
            "keypoints2": f2["keypoints"].numpy(),
            "pairs": m["matches"].numpy(),
            "scores": m["scores"].numpy(),
        }
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img1.astype(np.float32), "input2": img2.astype(np.float32)} | maps,
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, upstream match_pair {params['extractor']} + LightGlue, max_num_keypoints "
            f"{opts['max_keypoints']}, resize {opts['long_side']}, skimage astronaut.png {h}x{w} crop + affine warp as "
            f"raft_flow.py, torch {torch.__version__}",
            task="matching",
        )
        n1, n2, k = (len(maps[x]) for x in ("keypoints1", "keypoints2", "pairs"))
        print(f"  {variant.id}: {n1} / {n2} keypoints, {k} matches")


if __name__ == "__main__":
    main()
