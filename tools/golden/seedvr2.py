"""SeedVR2 goldens from the converted checkpoints, run by the family's own network on the GPU.

The authors' code (ByteDance-Seed/SeedVR with flash-attn and apex) runs only on CUDA in bf16, and the 3 B / 7 B
transformers in float32 do not fit this machine's CPU budget for a reference run, so the usual CPU float32 golden is not
possible. The golden is the family's network (`net/`) on the GPU in its portable mode; its parity with the authors'
code (`tools/seedvr2_reference.py`, same converted weights, same seeded noise) was checked on 2026-11-14.
A ref's golden.toml `options` are its mode (the 7 B: `quant = "nf4"`).
Input: `tests/synth.py image(101, 151)` stored as `input`; output `image` `[3, 2H, 2W]`. Run with the family's venv:

    FASTFORKS_GOLDEN_PATH=... python tools/golden/seedvr2.py [variant ...]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from _common import image, write

import fastforks.restoration
import goldens
from fastforks._core import registry

FAMILY = "seedvr2"
UPSTREAM = "https://github.com/ByteDance-Seed/SeedVR"
REF_COMMIT = "e4de8c24441a67e1b7df56abea10645059bb1185"
SIZE = (101, 151)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("variants", nargs="*", help="variant ids (default: all)")
    a = ap.parse_args()
    img = image(*SIZE)
    for card, variant in registry.cards("restoration").values():
        if card.family != FAMILY or (a.variants and variant.id not in a.variants):
            continue
        ref = f"{variant.id}@{variant.revision}"
        opts = dict(goldens.load(FAMILY, "restoration")["refs"].get(ref, {}).get("options", []))
        model = fastforks.restoration.load(variant.id, mode="portable", device="cuda", options=opts)
        out = model(img).image.float().cpu().numpy()
        write(
            ref,
            {"input": img.astype(np.float32), "image": out},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=(
                f"the family's own net on {torch.cuda.get_device_name()} (bf16, as upstream under autocast), converted"
                f" checkpoint, options {opts}, 1 step, CPU generator seed {model.card.options['seed']}, bottom/right"
                f" replicate padding to 2 x a multiple of 16; torch {torch.__version__}"
            ),
            task="restoration",
        )
        print(f"  {ref}: image {out.shape}, mean {out.mean():.4f}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
