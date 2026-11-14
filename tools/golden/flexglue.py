r"""FlexGlue sparse-matching goldens from the reference implementation.

Reference: upstream cvg/FlexGlue (`REF_COMMIT`) `FlexGlue(mp=None, weights=<the card's flexglue.pth>).match_pair(
image0, image1, detector=ALIKED(origin="corner", max_num_keypoints=...), sparse=True)` (its README usage, the
transformers in float32), CPU float32. ALIKED with `origin` is LightGlue after 5a650c4 (FlexGlue's README: "requires
the latest LightGlue commit"); its hub download is answered from the card's aliked-n16.pth. Inputs: the RAFT goldens'
pair (tools/golden/raft_flow.py: a 384x512 astronaut.png crop and its affine warp) in [0, 1]. Stored as float32:
`input`, `input2` (RGB), `keypoints1` / `keypoints2` `[N, 2]` (upstream's corner keypoints - 0.5: pixel centres at
integers), `pairs` `[K, 2]` and `scores` `[K]`.

    CUDA_VISIBLE_DEVICES= FASTFORKS_GOLDEN_PATH=... python tools/golden/flexglue.py FLEXGLUE_CHECKOUT LIGHTGLUE_DIR

LIGHTGLUE_DIR: the directory holding a `lightglue` package with the `origin` option (kornia must be importable).
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

FAMILY = "flexglue"
UPSTREAM = "https://github.com/cvg/FlexGlue"
REF_COMMIT = "99920296d082c9405ebfd612d674efe288e74881"


def from_files(files: dict[str, Path], url: str, **_kw: object) -> dict:
    """Upstream's hub download answered from the card's (hash-checked) file of the same name."""
    return torch.load(files[url.rsplit("/", 1)[1]], map_location="cpu", weights_only=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of cvg/FlexGlue")
    ap.add_argument("lightglue", type=Path, help="directory with a lightglue package that has `origin`")
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
    sys.path[:0] = [str(reference_tree(a.checkout, REF_COMMIT)), str(a.lightglue)]
    from flexglue import FlexGlue  # noqa: PLC0415  # reference checkout
    from lightglue import ALIKED  # noqa: PLC0415

    for card, variant in registry.cards("matching").values():
        if card.family != FAMILY:
            continue
        files = {f.name: f for f in registry.weights(card, variant)}
        torch.hub.load_state_dict_from_url = functools.partial(from_files, files)
        opts, params = dict(card.options), dict(card.params)
        detector = ALIKED(origin="corner", max_num_keypoints=opts["max_keypoints"]).eval()
        matcher = FlexGlue(mp=None, weights=str(files["flexglue.pth"])).eval()
        with torch.inference_mode():
            pred = matcher.match_pair(
                t1, t2, detector=detector, sparse=True,
                filter_threshold=params["threshold"], max_distance_px=params["max_distance_px"],
            )  # fmt: skip
        m0 = pred["matches0"][0]
        idx = torch.nonzero(m0 > -1)[:, 0]
        maps = {
            "keypoints1": (pred["keypoints0"][0] - 0.5).numpy(),
            "keypoints2": (pred["keypoints1"][0] - 0.5).numpy(),
            "pairs": torch.stack([idx, m0[idx]], -1).numpy().astype(np.float32),
            "scores": pred["matching_scores0"][0][idx].numpy(),
        }
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img1.astype(np.float32), "input2": img2.astype(np.float32)} | maps,
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, upstream FlexGlue(mp=None).match_pair(sparse=True) with ALIKED(origin='corner', "
            f"max_num_keypoints={opts['max_keypoints']}) (LightGlue resize 1024), describe 512x512, threshold "
            f"{params['threshold']}, max_distance_px {params['max_distance_px']}, skimage astronaut.png {h}x{w} crop + "
            f"affine warp as raft_flow.py, torch {torch.__version__}",
            task="matching",
        )
        n1, n2, k = (len(maps[x]) for x in ("keypoints1", "keypoints2", "pairs"))
        print(f"  {variant.id}: {n1} / {n2} keypoints, {k} matches")


if __name__ == "__main__":
    main()
