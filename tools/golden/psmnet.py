"""PSMNet goldens from the reference implementation.

Reference: upstream JiaRenChang/PSMNet (`REF_COMMIT`) + `patches/psmnet-cpu-device.patch` (the cost volume and the
regression's disparity values stay on the input's device: upstream hard-codes `.cuda()`), stacked hourglass, CPU
float32, max disparity 192, upstream test preprocessing (`Test_img.py`): ImageNet normalization, then zeros on top and
on the right up to a multiple of 16. The pair is raised to the card's `min_size` (241: the 64 x 64 pyramid pooling at
1/4 resolution needs >= 256 px after padding).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/psmnet.py PSMNET_CHECKOUT WEIGHTS_DIR

`WEIGHTS_DIR` holds upstream's `pretrained_sceneflow_new.tar`. The checkout only has to contain `REF_COMMIT`; the
reference runs from an export of that commit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from _common import pair, reference_tree, size, write

UPSTREAM = "https://github.com/JiaRenChang/PSMNet"
REF_COMMIT = "87ac9093afbf6545c093bd9d26c5ffd66e49a7b8"
PATCH = "psmnet-cpu-device.patch"
MAX_DISP = 192
MIN_SIZE = 241  # card io.min_size
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT, PATCH)))
    from models import stackhourglass  # noqa: PLC0415  # reference checkout on sys.path

    mean, std = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)
    images = pair(*size(MIN_SIZE))
    left, right = ((torch.from_numpy(x).permute(2, 0, 1)[None].float() / 255 - mean) / std for x in images)
    h, w = left.shape[2:]
    pads = (0, (-w) % 16, (-h) % 16, 0)
    lp, rp = F.pad(left, pads), F.pad(right, pads)
    net = stackhourglass(MAX_DISP)
    sd = torch.load(weights / "pretrained_sceneflow_new.tar", map_location="cpu", weights_only=True)["state_dict"]
    net.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()})
    with torch.inference_mode():
        disp = net.eval()(lp, rp)
    disp = disp[0, 0, pads[2] :, :w].numpy()  # upstream returns [B, 1, H, W]
    write(
        "psmnet-sceneflow@1",
        {"disparity": disp},
        family="psmnet",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=PATCH,
        settings=f"CPU float32, max disparity {MAX_DISP}, torch {torch.__version__}",
    )
    print(f"  median disparity {np.median(disp):.4f}")


if __name__ == "__main__":
    main()
