"""GwcNet goldens from the reference implementation.

Reference: upstream xy-guo/GwcNet (`REF_COMMIT`), CPU float32, max disparity 192, upstream test preprocessing:
ImageNet normalization, then zeros on top and on the right up to a multiple of 32 (`datasets/kitti_dataset.py`).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/gwcnet.py GWCNET_CHECKOUT WEIGHTS_DIR [gc g]

`WEIGHTS_DIR` holds upstream's archive layout (`sceneflow/gwcnet-<v>/checkpoint_000015.ckpt`).
The checkout only has to contain `REF_COMMIT`; the reference runs from an export of that commit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from _common import pair, reference_tree, write

UPSTREAM = "https://github.com/xy-guo/GwcNet"
REF_COMMIT = "0987074f48b098c6e975a58ca5aacf233b0fb0eb"
VARIANTS = ("gc", "g")
MAX_DISP = 192
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT)))
    from models import __models__  # noqa: PLC0415  # reference checkout on sys.path

    mean, std = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)
    left, right = ((torch.from_numpy(x).permute(2, 0, 1)[None].float() / 255 - mean) / std for x in pair())
    h, w = left.shape[2:]
    pads = (0, (-w) % 32, (-h) % 32, 0)
    lp, rp = F.pad(left, pads), F.pad(right, pads)
    for name in sys.argv[3:] or VARIANTS:
        net = __models__[f"gwcnet-{name}"](MAX_DISP)
        ckpt = weights / "sceneflow" / f"gwcnet-{name}" / "checkpoint_000015.ckpt"
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)["model"]
        net.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()})
        with torch.inference_mode():
            disp = net.eval()(lp, rp)[-1]
        disp = disp[0, pads[2] :, :w].numpy()
        write(
            f"gwcnet-{name}-sceneflow@1",
            {"disparity": disp},
            family="gwcnet",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, max disparity {MAX_DISP}, torch {torch.__version__}",
        )
        print(f"  median disparity {np.median(disp):.4f}")


if __name__ == "__main__":
    main()
