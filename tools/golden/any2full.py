"""Any2Full goldens from the reference implementation.

Reference: upstream zhiyuandaily/Any2Full (`REF_COMMIT`), CPU float32, `Any2Full.forward(image, prompt_depth)` with
the run script's defaults (`init_scailing`, depth in [1e-6, 1e3]) on a 98 x 140 image (resized to 518 x 742) and the
12 x 18 synthetic depth map with holes of `tests/synth.py` (`sparse_prompt`; its NaN holes enter upstream as 0).
Upstream needs the prompt at the image size; it gets it nearest-resized to the network size (as `model.prepare`
does), so its own resize is a no-op, and its output goes back to the input size as its `unresize` does it.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/any2full.py ANY2FULL_CHECKOUT WEIGHTS_DIR [vits vitb vitl]

`WEIGHTS_DIR` holds `Any2Full_vit{s,b,l}.pth.tar` (directly or in a subdirectory).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from _common import image, reference_tree, write

from fastforks._core.checkpoint import restricted_load
from fastforks.depth.models.any2full.model import EXTRA
from synth import sparse_prompt

UPSTREAM = "https://github.com/zhiyuandaily/Any2Full"
REF_COMMIT = "780edf6be2bf9260f1cce8e4ddbcb52d7bcdae9e"
VARIANTS = {"vits": "any2full-s", "vitb": "any2full-b", "vitl": "any2full-l"}  # arch: ref
HEIGHT, WIDTH = 98, 140
NET = (518, 742)  # upstream resize_to_multiple: short side 518, both sides rounded up to multiples of 14
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT)))
    from model.ours.any2full import Any2Full  # noqa: PLC0415 - reference checkout

    img = image(HEIGHT, WIDTH)
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255
    x = (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
    d = torch.from_numpy(np.nan_to_num(sparse_prompt(), nan=0.0))[None, None]
    d = F.interpolate(d, NET, mode="nearest")
    args = types.SimpleNamespace(init_scailing=True, max_depth=1e3, min_depth=1e-6)
    for arch in sys.argv[3:] or VARIANTS:
        net = Any2Full(encoder=arch, da_ckpt_path=None, args=args)
        state = restricted_load(next(weights.rglob(f"Any2Full_{arch}.pth.tar")), EXTRA)["state_dict"]
        net.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=True)
        with torch.inference_mode():
            pred = net.eval()(x, d)["pred"]
            depth = F.interpolate(pred, (HEIGHT, WIDTH), mode="bilinear", align_corners=True)[0, 0].numpy()
        write(
            f"{VARIANTS[arch]}@1",
            {"depth": depth, "input": img.astype(np.float32)},
            family="any2full",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, Any2Full.forward on {HEIGHT}x{WIDTH} (init_scailing), torch {torch.__version__}",
            task="depth",
        )
        print(f"  median depth {np.median(depth):.4f} m")


if __name__ == "__main__":
    main()
