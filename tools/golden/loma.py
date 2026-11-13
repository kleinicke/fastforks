r"""LoMa matching goldens from the reference implementation.

Reference: upstream davnords/loma (`REF_COMMIT`) `LoMa(LoMaB(mp=False, compile=False, weights_url=None))` with the
card's checkpoint loaded as upstream does (`load_state_dict(strict=False)`: the extra layers' keys), its `detect`,
`describe_keypoints`, `forward` and `filter_matches`; CPU float32 (every `amp` flag off, the DINOv2 cast back to
float32; upstream's module-level `device` must be the CPU: run with `CUDA_VISIBLE_DEVICES=`). Input as upstream's
`match` on paths, in float: the detector's image resized to a long side of `long_side` (aspect kept, sides floored
to a multiple of 8) and ImageNet-normalised by DaD, the descriptor's to the `size` square, not normalised (its
`read_image`); both torch bicubic antialiased (= PIL's filter), clamped, not rounded to uint8. Inputs: the RAFT
goldens' pair (tools/golden/raft_flow.py: a 384x512 astronaut.png crop and its affine warp) in [0, 1]. Stored as
float32: `input`, `input2` (RGB), `keypoints1` / `keypoints2` `[N, 2]` (upstream `to_pixel_coords` - 0.5: pixel
centres at integers), `pairs` `[K, 2]` and `scores` `[K]` (the product of the two softmaxes).

    CUDA_VISIBLE_DEVICES= FASTFORKS_GOLDEN_PATH=... TORCH_HOME=<cache with dad.pth, dinov2_vitl14_pretrain.pth> \\
        python tools/golden/loma.py LOMA_CHECKOUT
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from _common import reference_tree, write
from raft_flow import CROP, IMAGE_SHA256, warp

from fastforks._core import registry

if TYPE_CHECKING:
    from collections.abc import Callable

FAMILY = "loma"
UPSTREAM = "https://github.com/davnords/loma"
REF_COMMIT = "415caedde23a9ed479172ed174fe4ff7b4514d4b"


def resize(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """`[3, H, W]` in [0, 1] -> `[1, 3, *size]`."""
    return F.interpolate(image[None], size=size, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)


def build(files: list[Path], params: dict) -> Callable[[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
    """Upstream LoMa on the CPU in float32: `[3, H, W]` pair in [0, 1] -> keypoints (input pixels), pairs, scores.

    `params`: the card's params, variant params and options merged. The reference checkout's `src` must be on
    `sys.path`; DaD and the DINOv2 construction weights come from the torch hub cache.
    """
    import loma  # noqa: PLC0415  # reference checkout
    from loma.loma import filter_matches  # noqa: PLC0415

    cfg = loma.LoMaB(mp=False, compile=False, weights_url=None)
    net = loma.LoMa(cfg)
    sd = torch.load(files[0], map_location="cpu", weights_only=True)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    extra = [k for k in unexpected if not net._is_unexpected_extra_layer_key(k, cfg.n_layers)]  # noqa: SLF001
    if missing or extra:
        msg = f"checkpoint keys: missing {missing}, unexpected {extra}"
        raise SystemExit(msg)
    for m in net.modules():
        if hasattr(m, "amp_dtype"):
            m.amp = False
            m.amp_dtype = torch.float32
    dino = net._descriptor.encoder.frozen_dinov2  # noqa: SLF001
    dino.dinov2_vitl14 = dino.dinov2_vitl14.float()
    net.eval()
    thr, size, long_side = params["threshold"], int(params["size"]), int(params["long_side"])
    num = int(params["max_keypoints"])

    @torch.inference_mode()
    def run(image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        out = []
        for t in (image1, image2):
            h, w = t.shape[-2:]
            s = long_side / max(h, w)
            det = resize(t, (int(s * h) // 8 * 8, int(s * w) // 8 * 8))
            kpts = net._detector.detect({"image": det}, num_keypoints=num)["keypoints"]  # noqa: SLF001
            desc = net._descriptor.describe_keypoints(resize(t, (size, size)), kpts)["descriptions"]  # noqa: SLF001
            out.append((kpts, desc))
        scores = net(out[0][0], out[1][0], out[0][1], out[1][1])["scores"]
        m0, _, mscores0, _ = filter_matches(scores, thr)
        idx = torch.nonzero(m0[0] > -1)[:, 0]
        px = [(o[0][0] + 1) / 2 * torch.tensor([t.shape[-1], t.shape[-2]]) - 0.5
              for o, t in zip(out, (image1, image2), strict=True)]  # fmt: skip
        return {
            "keypoints1": px[0],
            "keypoints2": px[1],
            "pairs": torch.stack([idx, m0[0][idx]], dim=-1),
            "scores": mscores0[0][idx],
        }

    return run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of davnords/loma")
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
    sys.path.insert(0, str(reference_tree(a.checkout, REF_COMMIT) / "src"))
    for card, variant in registry.cards("matching").values():
        if card.family != FAMILY:
            continue
        params = dict(card.params) | dict(variant.params) | dict(card.options)
        run = build(registry.weights(card, variant), params)
        maps = run(t1, t2)
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img1.astype(np.float32), "input2": img2.astype(np.float32)}
            | {k: v.numpy().astype(np.float32) for k, v in maps.items()},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, upstream LoMa(LoMaB(mp=False)) with {variant.weights[0].name}, DaD detect on a "
            f"long side of {params['long_side']} + DeDoDe-G describe on {params['size']}x{params['size']} (torch "
            f"bicubic antialias), {params['max_keypoints']} keypoints, filter_matches threshold {params['threshold']}, "
            f"skimage astronaut.png {h}x{w} crop + affine warp as raft_flow.py, torch {torch.__version__}",
            task="matching",
        )
        print(f"  {variant.id}: {len(maps['pairs'])} matches")


if __name__ == "__main__":
    main()
