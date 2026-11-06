"""CFNet goldens from the reference implementation.

Reference: upstream gallenszl/CFNet (`REF_COMMIT`) + `patches/cfnet-cpu-device.patch` (the disparity sampler and the
warp use `tensor.device`: upstream's `get_device()` is -1 on the CPU), CPU float32, max disparity 192, upstream test
preprocessing: ImageNet normalization, then zeros on top and on the right up to a multiple of 64
(`datasets/eth3dLoader.py`).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/cfnet.py CFNET_CHECKOUT WEIGHTS_DIR

`WEIGHTS_DIR` holds upstream's `sceneflow_pretraining.ckpt`. The checkout only has to contain `REF_COMMIT`; the
reference runs from an export of that commit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from _common import pair, reference_tree, write

UPSTREAM = "https://github.com/gallenszl/CFNet"
REF_COMMIT = "982230dd360583fc35743d0b7ed528982fe5fd38"
PATCH = "cfnet-cpu-device.patch"
MAX_DISP = 192
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT, PATCH)))
    from models import __models__  # noqa: PLC0415  # reference checkout on sys.path

    mean, std = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)
    left, right = ((torch.from_numpy(x).permute(2, 0, 1)[None].float() / 255 - mean) / std for x in pair())
    h, w = left.shape[2:]
    pads = (0, (-w) % 64, (-h) % 64, 0)
    lp, rp = F.pad(left, pads), F.pad(right, pads)
    net = __models__["cfnet"](MAX_DISP)
    sd = torch.load(weights / "sceneflow_pretraining.ckpt", map_location="cpu", weights_only=True)["model"]
    net.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()})
    with torch.inference_mode():
        disp = net.eval()(lp, rp)[0][0]
    disp = disp[0, pads[2] :, :w].numpy()
    write(
        "cfnet-sceneflow@1",
        {"disparity": disp},
        family="cfnet",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=PATCH,
        settings=f"CPU float32, max disparity {MAX_DISP}, torch {torch.__version__}",
    )
    print(f"  median disparity {np.median(disp):.4f}")


if __name__ == "__main__":
    main()
