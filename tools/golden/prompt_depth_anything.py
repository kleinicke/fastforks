"""Prompt Depth Anything goldens from the reference implementation.

Reference: upstream DepthAnything/PromptDA (`REF_COMMIT`), CPU float32, `PromptDA.forward(image, prompt_depth)` on a
98 x 140 image (a multiple of 14, so no resize is involved) and the 12 x 18 synthetic prompt of `tests/synth.py`.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/prompt_depth_anything.py PROMPTDA_CHECKOUT WEIGHTS_DIR [vits vitl]

`WEIGHTS_DIR` holds `prompt_depth_anything_vit{s,l}.ckpt` (directly or in a subdirectory). The checkout needs the
packages `huggingface_hub` (imported, not used) and the `torchhub` DINOv2 it ships.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from _common import image, reference_tree, write

from synth import prompt_depth

UPSTREAM = "https://github.com/DepthAnything/PromptDA"
REF_COMMIT = "9161547ca7151e14fcc8f490ad3cc02fa9437483"
VARIANTS = {"vits": "prompt-depth-anything-s", "vitl": "prompt-depth-anything-l"}  # arch: ref
HEIGHT, WIDTH = 98, 140


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT)))
    from promptda.promptda import PromptDA  # noqa: PLC0415 - reference checkout

    img = image(HEIGHT, WIDTH)
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255
    p = torch.from_numpy(prompt_depth())[None, None]
    for arch in sys.argv[3:] or VARIANTS:
        path = next(weights.rglob(f"prompt_depth_anything_{arch}.ckpt"))
        net = PromptDA(encoder=arch, ckpt_path=str(path)).eval()
        with torch.inference_mode():
            depth = net(x, p)[0, 0].numpy()
        write(
            f"{VARIANTS[arch]}@1",
            {"depth": depth, "input": img.astype(np.float32)},
            family="prompt_depth_anything",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, PromptDA.forward on {HEIGHT}x{WIDTH}, torch {torch.__version__}",
            task="depth",
        )
        print(f"  median depth {np.median(depth):.4f} m")


if __name__ == "__main__":
    main()
