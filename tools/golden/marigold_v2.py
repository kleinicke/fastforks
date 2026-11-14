"""Marigold V2 goldens from the lean checkpoint in `exact_mask` mode.

The authors' code (huawei-bayerlab/marigold-v2 with diffusers, bitsandbytes NF4 and peft) needs CUDA and more than
16 GB, so the usual CPU float32 reference run is not possible here. The golden is the family's own network
(`net/`) on the GPU with the option `exact_mask = True`, which passes upstream's all-true attention mask to SDPA: on an
RTX 3090 that output equals the authors' bit for bit at 512, 768 and 1024 px (results/2026-11-14_marigold_v2_3090);
the lean checkpoint equals the converter's output tensor by tensor. Run with the family's own venv:

    FASTFORKS_GOLDEN_PATH=... python tools/golden/marigold_v2.py
"""

from __future__ import annotations

import numpy as np
import torch
from _common import image, write

import fastforks.depth

UPSTREAM = "https://github.com/huawei-bayerlab/marigold-v2"
REF_COMMIT = "main"
REF = "marigold-v2-depth@1"


def main() -> None:
    model = fastforks.depth.load(REF, mode="portable", device="cuda", options={"exact_mask": True})
    depth = model(image()).depth.float().cpu().numpy()
    seed = model.card.options["seed"]
    write(
        REF,
        {"depth": depth},
        family="marigold_v2",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=None,
        settings=(
            f"the family's own net on {torch.cuda.get_device_name()} (bf16 GEMMs, float32 glue) with"
            f" exact_mask = True (bit-equal to the authors' code on an RTX 3090), lean checkpoint"
            f" depth/Log-stage2, 1 step, CPU generator seed {seed}, torch {torch.__version__}"
        ),
        task="depth",
    )
    print(f"  depth {depth.shape}, median {np.median(depth):.4f}")


if __name__ == "__main__":
    main()
