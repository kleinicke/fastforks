"""Marigold goldens from the reference implementation.

Reference: upstream prs-eth/Marigold (`REF_COMMIT`, unpatched) with diffusers / transformers, CPU float32,
`MarigoldDepthPipeline.from_pretrained(WEIGHTS)` (the prs-eth/marigold-depth-v1-1 snapshot `HF_REVISION`) called on
the PIL image with the card defaults: `denoising_steps=4`, `ensemble_size=1`, `processing_res=768`, `match_input_res`,
bilinear resampling, and a CPU `torch.Generator` seeded with 2024 (the runners draw the same CPU noise). Run this with
a Python that has the upstream dependencies (diffusers, transformers, torchvision).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/marigold.py MARIGOLD_CHECKOUT marigold-depth-v1-1/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from _common import image, reference_tree, write
from PIL import Image

UPSTREAM = "https://github.com/prs-eth/Marigold"
REF_COMMIT = "2bfbdeae5d10a50b71f1ba20c865d46e480f6010"
HF_REVISION = "9571e7123e258cf052b4e54241f17971c290e9a8"
STEPS, RES, SEED = 4, 768, 2024


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(reference_tree(repo, REF_COMMIT)))
    sys.path.remove(str(Path(__file__).resolve().parent))  # this script (same name) would shadow upstream's marigold/
    from marigold import MarigoldDepthPipeline  # noqa: PLC0415  # reference checkout

    pipe = MarigoldDepthPipeline.from_pretrained(str(weights), torch_dtype=torch.float32)
    ids = pipe.tokenizer("", padding="do_not_pad", return_tensors="pt").input_ids.tolist()
    out = pipe(
        Image.fromarray(image()),
        denoising_steps=STEPS,
        ensemble_size=1,
        processing_res=RES,
        match_input_res=True,
        batch_size=1,
        generator=torch.Generator().manual_seed(SEED),
        color_map=None,
        show_progress_bar=False,
    )
    depth = np.asarray(out.depth_np, dtype=np.float32)
    write(
        "marigold-depth-v1-1@1",
        {"depth": depth},
        family="marigold",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=None,
        settings=(
            f"CPU float32, weights prs-eth/marigold-depth-v1-1@{HF_REVISION}, {STEPS} DDIM steps, processing_res {RES},"
            f" ensemble 1, CPU generator seed {SEED}, torch {torch.__version__}"
        ),
        task="depth",
    )
    print(f"  empty-prompt ids {ids}, depth {depth.shape}, median {np.median(depth):.4f}")


if __name__ == "__main__":
    main()
