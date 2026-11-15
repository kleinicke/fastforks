"""Restormer restoration goldens from the reference implementation.

Reference: upstream swz30/Restormer (`REF_COMMIT`) `basicsr/models/archs/restormer_arch.py` `Restormer` as the
variant's options build it (WithBias norms; BiasFree for real denoising) and the test scripts' call
(`Motion_Deblurring/test.py`): reflect padding bottom / right to a multiple of 8, `model(input_)`, crop, clamped to
[0, 1] (without the uint8 rounding), on the CPU in float32. Input: `tests/synth.py image(101, 151)` stored as `input`;
output `image` `[3, H, W]`.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/restormer.py RESTORMER_CHECKOUT WEIGHTS_DIR [ids]

WEIGHTS_DIR holds the release `.pth` files (`{"params": ...}`). Run with `CUDA_VISIBLE_DEVICES=`. `restormer_arch.py`
is loaded by path, unmodified; it needs `einops` (e.g. on `PYTHONPATH` from a `uv pip install --target`).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from _common import image, reference_tree, write

from fastforks._core import registry

if TYPE_CHECKING:
    import types

FAMILY = "restormer"
UPSTREAM = "https://github.com/swz30/Restormer"
REF_COMMIT = "68dc6ac472db26f16361150cb7a96a1bc87da93f"
SIZE = (101, 151)
FACTOR = 8


def _arch(checkout: Path) -> types.ModuleType:
    """Upstream `restormer_arch`, loaded by path."""
    name = "restormer_arch"
    if name in sys.modules:
        return sys.modules[name]
    path = reference_tree(checkout, REF_COMMIT) / "basicsr" / "models" / "archs" / "restormer_arch.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    sys.modules[name] = module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(checkout: Path, weights: Path, params: dict, *, device: torch.device) -> torch.nn.Module:
    """Upstream `Restormer` with the checkpoint's `params`, in eval mode."""
    net = _arch(checkout).Restormer(
        dim=params["dim"], num_blocks=list(params["blocks"]), num_refinement_blocks=params["refinement"],
        heads=list(params["heads"]), ffn_expansion_factor=params["ffn"], bias=False, LayerNorm_type=params["norm"],
    )  # fmt: skip
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["params"], strict=True)
    return net.eval().to(device)


def run(net: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """`[1, 3, H, W]` float in [0, 1] -> `[3, H, W]` float32 in [0, 1] (`test.py`: reflect pad, model, crop, clamp)."""
    h, w = x.shape[-2:]
    with torch.inference_mode():
        y = net(F.pad(x, (0, -w % FACTOR, 0, -h % FACTOR), "reflect"))
        return y[0, :, :h, :w].float().clamp(0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of swz30/Restormer")
    ap.add_argument("weights", type=Path, help="directory with the release .pth files")
    ap.add_argument("variants", nargs="*", help="variant ids (default: all)")
    a = ap.parse_args()
    if torch.cuda.is_available():
        msg = "run with CUDA_VISIBLE_DEVICES= (the reference runs on the CPU)"
        raise SystemExit(msg)
    img = image(*SIZE)
    x = torch.from_numpy(img.astype(np.float32) / 255).permute(2, 0, 1)[None]
    for card, variant in registry.cards("restoration").values():
        if card.family != FAMILY or (a.variants and variant.id not in a.variants):
            continue
        p = {**card.params, **variant.params}
        net = build(a.checkout, a.weights / variant.weights[0].name, p, device=torch.device("cpu"))
        out = run(net, x).numpy()
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img.astype(np.float32), "image": out},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, {p['norm']} LayerNorm, reflect padding to a multiple of {FACTOR}, model(input_), "
            f"clamped, no uint8 rounding; tests/synth.py image{SIZE}; torch {torch.__version__}",
            task="restoration",
        )
        print(f"  {variant.id}: image {out.shape}, mean {out.mean():.4f}")


if __name__ == "__main__":
    main()
