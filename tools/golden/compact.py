"""Real-ESRGAN Compact super-resolution goldens from the reference implementation.

Reference: upstream xinntao/Real-ESRGAN (`REF_COMMIT`): `RealESRGANer` (`half=False`, `pre_pad=0`, no tile, as
`inference_realesrgan.py -n realesr-general-x4v3 -dn 1 --fp32`) with its `SRVGGNetCompact`, on the CPU in float32. As
`tools/golden/realesrgan.py`: float RGB in [0, 1] in, `pre_process`, `process`, `post_process`, clamped to [0, 1], no
uint8 rounding. Input: `tests/synth.py image(101, 151)`, stored as `input`; output `image` `[3, sH, sW]`.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/compact.py REAL_ESRGAN_CHECKOUT WEIGHTS_DIR [ids]

WEIGHTS_DIR holds the release `.pth` files named as the card. Run with `CUDA_VISIBLE_DEVICES=`. `srvgg_arch.py` and
`realesrgan/utils.py` are loaded by path, unmodified, with realesrgan.py's stub modules.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch
from _common import image, reference_tree, write
from realesrgan import REF_COMMIT, STUBS, UPSTREAM, load, run

from fastforks._core import registry

FAMILY = "compact"
SIZE = (101, 151)


def build(checkout: Path, weights: Path, params: dict, *, half: bool, device: torch.device) -> object:
    """Upstream's `RealESRGANer` (`pre_pad` 0, no tile) on `SRVGGNetCompact` with `params`."""
    if "srvgg_arch" not in sys.modules:
        for name, attrs in STUBS.items():
            sys.modules.setdefault(name, types.ModuleType(name)).__dict__.update(attrs)
        tree = reference_tree(checkout, REF_COMMIT) / "realesrgan"
        load("srvgg_arch", tree / "archs" / "srvgg_arch.py")
        if "realesrgan_utils" not in sys.modules:
            load("realesrgan_utils", tree / "utils.py")
    net = sys.modules["srvgg_arch"].SRVGGNetCompact(
        3, 3, num_feat=params["num_feat"], num_conv=params["num_conv"], upscale=params["scale"], act_type="prelu"
    )
    return sys.modules["realesrgan_utils"].RealESRGANer(
        scale=params["scale"], model_path=str(weights), model=net, pre_pad=0, half=half, device=device
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of xinntao/Real-ESRGAN")
    ap.add_argument("weights", type=Path, help="directory with the release .pth files")
    ap.add_argument("variants", nargs="*", help="variant ids (default: all)")
    a = ap.parse_args()
    if torch.cuda.is_available():
        msg = "run with CUDA_VISIBLE_DEVICES= (the reference runs on the CPU)"
        raise SystemExit(msg)
    img = image(*SIZE)
    x = img.astype(np.float32) / 255
    for card, variant in registry.cards("restoration").values():
        if card.family != FAMILY or (a.variants and variant.id not in a.variants):
            continue
        p = {**card.params, **variant.params}
        up = build(a.checkout, a.weights / variant.weights[0].name, p, half=False, device=torch.device("cpu"))
        out = run(up, x).cpu().numpy()
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img.astype(np.float32), "image": out},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, upstream RealESRGANer pre_pad 0 no tile (SRVGGNetCompact) on float RGB, clamped, "
            f"no uint8 rounding; tests/synth.py image{SIZE}; torch {torch.__version__}",
            task="restoration",
        )
        print(f"  {variant.id}: image {out.shape}, mean {out.mean():.4f}")


if __name__ == "__main__":
    main()
