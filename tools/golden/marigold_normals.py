"""Marigold normals LCM goldens from the reference implementation.

Reference: diffusers `MarigoldNormalsPipeline` (`DIFFUSERS` tag, unpatched), CPU float32, `from_pretrained(WEIGHTS)`
(the prs-eth/marigold-normals-lcm-v0-1 snapshot `HF_REVISION`) called on the float image in [0, 1] with the card
defaults: `num_inference_steps=1`, `ensemble_size=1`, `processing_resolution=768`, `match_input_resolution`, bilinear
resampling, and a CPU `torch.Generator` seeded with 2024 (the runners draw the same CPU noise). The normals are stored
in the camera frame (x right, y down, z forward): Marigold's (x, y, z) -> (x, -y, -z). Run this with a Python that has
diffusers and transformers.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/marigold_normals.py marigold-normals-lcm-v0-1/
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from _common import image, write

UPSTREAM = "https://github.com/huggingface/diffusers"
DIFFUSERS = "v0.41.0"
HF_REVISION = "28214f06633a6a73088b569d1d068c3e48ea7902"
STEPS, RES, SEED = 1, 768, 2024


def pipeline(snapshot: str, dtype: torch.dtype = torch.float32) -> Any:  # noqa: ANN401 - diffusers pipeline
    """The reference pipeline from the weights snapshot directory (`dtype`: float16 = the diffusers fp16 option)."""
    import diffusers  # noqa: PLC0415 - reference environment only

    assert diffusers.__version__ == DIFFUSERS.lstrip("v"), diffusers.__version__  # noqa: S101 - tool
    return diffusers.MarigoldNormalsPipeline.from_pretrained(snapshot, torch_dtype=dtype)


def predict(pipe: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - diffusers pipeline
    """`[1, 3, H, W]` float in [0, 1] -> unit normals `[H, W, 3]` in the camera frame (float32), the card defaults."""
    out = pipe(
        x,
        num_inference_steps=STEPS,
        ensemble_size=1,
        processing_resolution=RES,
        match_input_resolution=True,
        batch_size=1,
        generator=torch.Generator().manual_seed(SEED),
        output_type="pt",
    )
    return out.prediction[0].permute(1, 2, 0).float() * torch.tensor([1.0, -1.0, -1.0], device=out.prediction.device)


def main() -> None:
    pipe = pipeline(str(Path(sys.argv[1])))
    x = torch.from_numpy(image()).permute(2, 0, 1)[None].float() / 255
    normals = predict(pipe, x).numpy().astype(np.float32)
    write(
        "marigold-normals-lcm-v0-1@1",
        {"normals": normals},
        family="marigold_normals",
        upstream=f"{UPSTREAM}@{DIFFUSERS}",
        patch=None,
        settings=(
            f"CPU float32, weights prs-eth/marigold-normals-lcm-v0-1@{HF_REVISION}, {STEPS} LCM step(s), processing "
            f"resolution {RES}, ensemble 1, CPU generator seed {SEED}, camera frame, torch {torch.__version__}"
        ),
        task="depth",
    )
    print(f"  normals {normals.shape}, mean z {normals[..., 2].mean():.4f}")


if __name__ == "__main__":
    main()
