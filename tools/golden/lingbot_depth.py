"""LingBot-Depth goldens from the reference implementation.

Reference: upstream robbyant/lingbot-depth (`REF_COMMIT`), CPU float32, `MDMModel.infer(image, depth_in,
resolution_level=9, use_fp16=False, apply_mask=False)` on a 98 x 140 image and the 12 x 18 synthetic depth map with
holes of `tests/synth.py` (`sparse_prompt`).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/lingbot_depth.py LINGBOT_CHECKOUT WEIGHTS_DIR [v05 dc]

`WEIGHTS_DIR` holds `v05.pt` and `dc.pt` (the `model.pt` files of the two Hugging Face repos). Without xFormers
upstream cannot run its per-sample token lists; the blocks here run each list entry on its own (the same maths).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from _common import image, reference_tree, write

from synth import sparse_prompt

UPSTREAM = "https://github.com/robbyant/lingbot-depth"
REF_COMMIT = "f3a237e434ae987bc38281476d6cfb5df3e4d739"
VARIANTS = {"v05": "lingbot-depth", "dc": "lingbot-depth-dc"}  # weight file stem: ref
HEIGHT, WIDTH = 98, 140


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    os.environ["XFORMERS_DISABLED"] = "1"
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT)))
    from mdm.model.dinov2_rgbd.layers.block import Block, NestedTensorBlock  # noqa: PLC0415 - reference checkout
    from mdm.model.v2 import MDMModel  # noqa: PLC0415 - reference checkout

    def nested(self: Block, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        return [Block.forward(self, t) for t in x] if isinstance(x, list) else Block.forward(self, x)

    NestedTensorBlock.forward = nested
    img = image(HEIGHT, WIDTH)
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255
    d = torch.from_numpy(sparse_prompt())[None]
    for stem in sys.argv[3:] or VARIANTS:
        net = MDMModel.from_pretrained(str(next(weights.rglob(f"{stem}.pt")))).eval()
        out = net.infer(x, d, resolution_level=9, use_fp16=False, apply_mask=False)
        depth, mask = out["depth"][0].numpy(), out["mask"][0].numpy()
        write(
            f"{VARIANTS[stem]}@1",
            {"depth": depth, "mask": mask.astype(np.float32), "input": img.astype(np.float32)},
            family="lingbot_depth",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, MDMModel.infer on {HEIGHT}x{WIDTH} (level 9, no mask), torch {torch.__version__}",
            task="depth",
        )
        print(f"  median depth {np.median(depth):.4f} m, mask {mask.mean():.3f}")


if __name__ == "__main__":
    main()
