"""CoEx goldens from the reference implementation.

Reference: upstream antabangun/coex (`REF_COMMIT`), SceneFlow checkpoint, CPU float32, configuration
`configs/stereo/cfg_coex.yaml` + `configs/backbone.yaml`. Upstream's test preprocessing: ImageNet normalization, then
zeros below and to the right up to a multiple of 32. Upstream builds its backbone with timm (`mobilenetv2_100`); run
this with a Python that has timm, PyYAML, scikit-image and OpenCV installed (the checkpoint holds the backbone weights,
`from_scratch` stops timm from downloading ImageNet ones). A Lightning pickle: only run it on the authors' file.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/coex.py COEX_CHECKOUT WEIGHTS_DIR

`WEIGHTS_DIR` holds `sceneflow.ckpt`. The checkout only has to contain `REF_COMMIT`; the reference runs from an export.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from _common import pair, reference_tree, write

UPSTREAM = "https://github.com/antabangun/coex"
REF_COMMIT = "a98b37d33f37754582b1e71f38216af8c69f2347"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    root = reference_tree(repo, REF_COMMIT)
    sys.path.insert(0, str(root))
    from models.stereo.CoEx import CoEx  # noqa: PLC0415  # reference checkout on sys.path

    cfg = yaml.safe_load((root / "configs/stereo/cfg_coex.yaml").read_text())["model"]["stereo"]
    cfg["backbone"].update(yaml.safe_load((root / "configs/backbone.yaml").read_text()))
    cfg["backbone"]["from_scratch"] = True
    net = CoEx(cfg)
    ckpt = torch.load(weights / "sceneflow.ckpt", map_location="cpu", weights_only=False)["state_dict"]
    net.load_state_dict({k.removeprefix("stereo."): v for k, v in ckpt.items()}, strict=True)
    mean, std = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)
    left, right = ((torch.from_numpy(x).permute(2, 0, 1)[None].float() / 255 - mean) / std for x in pair())
    h, w = left.shape[2:]
    pads = (0, 32 - w % 32, 0, 32 - h % 32)  # upstream (stereo.py): a full extra 32 px for sizes that are multiples
    with torch.inference_mode():
        disp = net.eval()(F.pad(left, pads), F.pad(right, pads), training=False)[0][0, :h, :w].numpy()
    write(
        "coex-sceneflow@1",
        {"disparity": disp},
        family="coex",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=None,
        settings=f"CPU float32, max disparity 192, torch {torch.__version__}",
    )
    print(f"  median disparity {np.median(disp):.4f}")


if __name__ == "__main__":
    main()
