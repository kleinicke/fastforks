"""LoFTR matching goldens from the reference implementation.

Reference: upstream zju3dv/LoFTR (`REF_COMMIT`) `LoFTR` with its `default_cfg` and the variant's evaluation settings
(`coarse.temp_bug_fix`, `match_coarse.border_rm`, `match_coarse.thr` from the card), the release checkpoints (the
card's files), CPU float32. Input as upstream's loaders: grayscale (OpenCV's RGB weights, in float, not rounded to
uint8), resized bilinearly to the card's `long_side` (rounded) and cut down to multiples of 8, with upstream's
`scale0` / `scale1` mapping the matches back to input pixels. Inputs: the RAFT goldens' pair
(tools/golden/raft_flow.py: a 384x512 astronaut.png crop and its affine warp) in [0, 1]. Stored as float32: `input`,
`input2` (RGB), `keypoints1` / `keypoints2` `[N, 2]` (`mkpts0_f` / `mkpts1_f`), `pairs` `[N, 2]` (`[i, i]`) and
`scores` `[N]` (`mconf`).

    FASTFORKS_GOLDEN_PATH=... PYTHONPATH=<einops, kornia < 0.8, yacs> python tools/golden/loftr.py LOFTR_CHECKOUT
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from _common import reference_tree, write
from raft_flow import CROP, IMAGE_SHA256, warp

from fastforks._core import registry

FAMILY = "loftr"
UPSTREAM = "https://github.com/zju3dv/LoFTR"
REF_COMMIT = "df7ca80f917334b94cfbe32cc2901e09a80e70a8"
GRAY = (0.299, 0.587, 0.114)  # cv2.COLOR_RGB2GRAY


def prepare(image: torch.Tensor, long_side: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`[3, H, W]` -> `[1, 1, h, w]` gray at the matching size and upstream's `scale` `[1, 2]` (w / h factors)."""
    h, w = image.shape[-2:]
    s = long_side / max(h, w)
    rh, rw = round(h * s) // 8 * 8, round(w * s) // 8 * 8
    g = (image * torch.tensor(GRAY, device=image.device).view(3, 1, 1)).sum(0)[None, None]
    g = F.interpolate(g, size=(rh, rw), mode="bilinear", align_corners=False)
    return g, torch.tensor([[w / rw, h / rh]], device=image.device)


def build(ckpt: Path, params: dict, threshold: float) -> torch.nn.Module:
    """Upstream LoFTR with the variant's settings (the reference checkout on `sys.path`)."""
    from src.loftr import LoFTR, default_cfg  # noqa: PLC0415  # reference checkout

    cfg = copy.deepcopy(default_cfg)
    cfg["coarse"]["temp_bug_fix"] = params["temp_bug_fix"]
    cfg["match_coarse"]["border_rm"] = params["border_rm"]
    cfg["match_coarse"]["thr"] = threshold
    net = LoFTR(config=cfg)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
    net.load_state_dict({k.removeprefix("matcher."): v for k, v in sd.items()})
    return net.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of zju3dv/LoFTR")
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
        (ckpt,) = registry.weights(card, variant)
        params = dict(card.params) | dict(variant.params)
        long_side = int(card.options["long_side"])
        net = build(ckpt, params, params["threshold"])
        (g0, s0), (g1, s1) = prepare(t1, long_side), prepare(t2, long_side)
        batch = {"image0": g0, "image1": g1, "scale0": s0, "scale1": s1}
        with torch.inference_mode():
            net(batch)
        n = len(batch["mconf"])
        idx = np.arange(n)
        maps = {
            "keypoints1": batch["mkpts0_f"].numpy(),
            "keypoints2": batch["mkpts1_f"].numpy(),
            "pairs": np.stack([idx, idx], -1),
            "scores": batch["mconf"].numpy(),
        }
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img1.astype(np.float32), "input2": img2.astype(np.float32)}
            | {k: v.astype(np.float32) for k, v in maps.items()},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, upstream LoFTR default_cfg + temp_bug_fix {params['temp_bug_fix']}, border_rm "
            f"{params['border_rm']}, thr {params['threshold']}, gray at long side {long_side}, skimage astronaut.png "
            f"{h}x{w} crop + affine warp as raft_flow.py, torch {torch.__version__}",
            task="matching",
        )
        print(f"  {variant.id}: {n} matches")


if __name__ == "__main__":
    main()
