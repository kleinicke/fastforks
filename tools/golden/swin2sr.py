"""Swin2SR super-resolution goldens from the reference implementation.

Reference: upstream mv-lab/swin2sr (`REF_COMMIT`) `models/network_swin2sr.py` `Swin2SR` as `main_test_swin2sr.py`
`define_model` builds it for the variant's task, and its test call: mirror padding (`cat(x, flip(x))`) to
`(size // 8 + 1) * 8`, `model(lq)` (no tile), crop, clamp to [0, 1] (without the uint8 rounding), on the CPU in float32.
Input: `tests/synth.py image(101, 151)` stored as `input`; output `image` `[3, sH, sW]`.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/swin2sr.py SWIN2SR_CHECKOUT WEIGHTS_DIR [ids]

WEIGHTS_DIR holds the release v0.0.1 `.pth` files. Run with `CUDA_VISIBLE_DEVICES=`. `network_swin2sr.py` is loaded by
path, unmodified (it imports timm).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from _common import image, reference_tree, write

from fastforks._core import registry

FAMILY = "swin2sr"
UPSTREAM = "https://github.com/mv-lab/swin2sr"
REF_COMMIT = "6f25d7689813f7dd12c5a212b638066fa71bc9e6"
SIZE = (101, 151)
WINDOW = 8


def build(checkout: Path, weights: Path, params: dict, *, device: torch.device) -> torch.nn.Module:
    """Upstream `Swin2SR` with `params` (the card's) and the checkpoint's weights (a plain state dict), in eval mode."""
    if "network_swin2sr" not in sys.modules:
        path = reference_tree(checkout, REF_COMMIT) / "models" / "network_swin2sr.py"
        spec = importlib.util.spec_from_file_location("network_swin2sr", path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        sys.modules["network_swin2sr"] = module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    net = sys.modules["network_swin2sr"].Swin2SR(
        upscale=params["scale"], in_chans=3, img_size=params["img_size"], window_size=WINDOW, img_range=1.0,
        depths=params["depths"], embed_dim=params["embed_dim"], num_heads=params["heads"], mlp_ratio=2,
        upsampler=params["upsampler"], resi_connection=params["resi_connection"],
    )  # fmt: skip
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return net.eval().to(device)


def run(net: torch.nn.Module, x: torch.Tensor, scale: int) -> torch.Tensor:
    """`[1, 3, H, W]` float in [0, 1] -> `[3, sH, sW]` float32 in [0, 1] (`main_test_swin2sr.py`, clamped)."""
    h, w = x.shape[-2:]
    x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, : (h // WINDOW + 1) * WINDOW, :]
    x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, : (w // WINDOW + 1) * WINDOW]
    with torch.inference_mode():
        return net(x)[0, :, : h * scale, : w * scale].float().clamp(0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of mv-lab/swin2sr")
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
        out = run(net, x, p["scale"]).numpy()
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img.astype(np.float32), "image": out},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, main_test_swin2sr.py: mirror padding to (size // 8 + 1) * 8, model(lq), crop, "
            f"clamped, no uint8 rounding; tests/synth.py image{SIZE}; torch {torch.__version__}",
            task="restoration",
        )
        print(f"  {variant.id}: image {out.shape}, mean {out.mean():.4f}")


if __name__ == "__main__":
    main()
