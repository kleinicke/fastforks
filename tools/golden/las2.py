"""LiteAnyStereo V2 goldens from the reference implementation.

Reference: upstream TomTomTommi/LiteAnyStereo (`REF_COMMIT`), LAS2 S / M / L / H, CPU float32, max disparity 192, 4
update iterations for H, upstream `demo.py` preprocessing (0-255 RGB, `InputPadder(divis_by=32)`). Upstream builds its
FasterNet-T0 backbone with timm (`features_only=True`, no download with `fnet_pretrained=False`): run this with a
Python that has timm >= 1.0.20 installed. The checkpoints hold the backbone weights.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/las2.py LAS_CHECKOUT WEIGHTS_DIR [s m l h]

`WEIGHTS_DIR` holds the upstream `LAS2_<S|M|L|H>.pth` files. The checkout only has to contain `REF_COMMIT`; the
reference runs from an export of that commit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from _common import pair, reference_tree, write

UPSTREAM = "https://github.com/TomTomTommi/LiteAnyStereo"
REF_COMMIT = "8c97bd4c4da3712c2ac60003a23201dfdb5935f4"
SIZES = ("s", "m", "l", "h")
MAX_DISP = 192


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT)))
    from core.liteanystereov2 import build_liteanystereo  # noqa: PLC0415  # reference checkout on sys.path
    from core.utils.utils import InputPadder  # noqa: PLC0415  # reference checkout on sys.path

    left, right = (torch.from_numpy(x).permute(2, 0, 1)[None].float() for x in pair())
    padder = InputPadder(left.shape, divis_by=32)
    lp, rp = padder.pad(left, right)
    for size in sys.argv[3:] or SIZES:
        net = build_liteanystereo(model_size=size, fnet_pretrained=False, max_disp=MAX_DISP)
        sd = torch.load(weights / f"LAS2_{size.upper()}.pth", map_location="cpu", weights_only=True)
        net.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()})
        with torch.inference_mode():
            disp = net.eval()(lp, rp, max_disp=MAX_DISP, test_mode=True)
        disp = padder.unpad(disp)[0, 0].numpy()
        write(
            f"las2-{size}@1",
            {"disparity": disp},
            family="las2",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, max disparity {MAX_DISP}, torch {torch.__version__}",
        )
        print(f"  median disparity {np.median(disp):.4f}")


if __name__ == "__main__":
    main()
