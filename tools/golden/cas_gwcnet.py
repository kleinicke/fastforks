"""Cas-GwcNet goldens from the reference implementation.

Reference: upstream alibaba/cascade-stereo (`REF_COMMIT`, `CasStereoNet/`) + `patches/cas-gwcnet-align-corners.patch`
(the warp's `grid_sample` gets `align_corners=True`, the PyTorch 1.2 default the code was trained with; current
PyTorch defaults to False), CPU float32, README evaluation settings (ndisps 48,24, disparity intervals 4,1, neighbour
search 3, cost-aggregation channels 32,32,16), output `outputs["stage2"]["pred"]`; preprocessing as upstream's KITTI
test loader: ImageNet normalization, then zeros on top and on the right up to a multiple of 32. The pair is raised to
each variant's `min_size` (Cas-GwcNet-c: 225; its 64 x 64 pyramid pooling at 1/4 resolution needs >= 256 px).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/cas_gwcnet.py CASCADE_STEREO_CHECKOUT WEIGHTS_DIR

`WEIGHTS_DIR` holds upstream's `casgwcnet.ckpt` and `casgwcnet-c-2.ckpt`. The checkout only has to contain
`REF_COMMIT`; the reference runs from an export of that commit.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from _common import pair, reference_tree, size, write

UPSTREAM = "https://github.com/alibaba/cascade-stereo"
REF_COMMIT = "2ae7da65515c82025383a0fbee8d6fe063dc7101"
PATCH = "cas-gwcnet-align-corners.patch"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
VARIANTS = {  # ref: (upstream model name, checkpoint, card io.min_size)
    "cas-gwcnet-gc-sceneflow@1": ("gwcnet", "casgwcnet.ckpt", 1),
    "cas-gwcnet-c-sceneflow@1": ("gwcnet-c", "casgwcnet-c-2.ckpt", 225),
}


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT, PATCH) / "CasStereoNet"))
    from models import __models__  # noqa: PLC0415  # reference checkout on sys.path

    mean, std = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)
    for ref, (name, ckpt, min_size) in VARIANTS.items():
        images = pair(*size(min_size))
        left, right = ((torch.from_numpy(x).permute(2, 0, 1)[None].float() / 255 - mean) / std for x in images)
        h, w = left.shape[2:]
        pads = (0, (-w) % 32, (-h) % 32, 0)
        lp, rp = F.pad(left, pads), F.pad(right, pads)
        with contextlib.redirect_stdout(io.StringIO()):  # the constructor prints its settings
            net = __models__[name](
                maxdisp=192,
                ndisps=[48, 24],
                disp_interval_pixel=[4.0, 1.0],
                cr_base_chs=[32, 32, 16],
                grad_method="detach",
                using_ns=True,
                ns_size=3,
            )
        sd = torch.load(weights / ckpt, map_location="cpu", weights_only=True)["model"]
        net.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()})
        with torch.inference_mode():
            disp = net.eval()(lp, rp)["stage2"]["pred"]
        disp = disp[0, pads[2] :, :w].numpy()
        write(
            ref,
            {"disparity": disp},
            family="cas_gwcnet",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=PATCH,
            settings=f"CPU float32, model {name}, ndisps 48,24, intervals 4,1, ns 3, torch {torch.__version__}",
        )
        print(f"{ref}: median disparity {np.median(disp):.4f}")


if __name__ == "__main__":
    main()
