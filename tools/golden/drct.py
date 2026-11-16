"""DRCT super-resolution goldens from the reference implementation.

Reference: upstream ming053l/DRCT (`REF_COMMIT`) `drct/archs/DRCT_arch.py` `DRCT` as the test options build it for the
variant, and `drct/models/drct_model.py`'s test call: reflect padding bottom / right to a multiple of 16
(`pre_process`), `net_g_ema(img)` (no tile), crop (`post_process`), clamped to [0, 1] (without the uint8 rounding), on
the CPU in float32.
Input: `tests/synth.py image(101, 151)` stored as `input`; output `image` `[3, sH, sW]`.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/drct.py DRCT_CHECKOUT WEIGHTS_DIR [ids]

WEIGHTS_DIR holds the Google Drive `.pth` files named as the card. Run with `CUDA_VISIBLE_DEVICES=`. `DRCT_arch.py` is
loaded by path, unmodified; its two BasicSR imports (the registry, timm's `to_2tuple` / `trunc_normal_`) and its
unused `einops` import are stub modules (`STUBS`: BasicSR's package does not import on Python 3.12).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from _common import image, reference_tree, write
from timm.layers import to_2tuple, trunc_normal_

from fastforks._core import registry

FAMILY = "drct"
UPSTREAM = "https://github.com/ming053l/DRCT"
REF_COMMIT = "c0a8b51e69241cf7f8ae598053235ff498517935"
SIZE = (101, 151)
WINDOW = 16


class _Registry:
    def register(self) -> object:
        return lambda cls: cls


STUBS = {
    "basicsr": {},
    "basicsr.utils": {},
    "basicsr.utils.registry": {"ARCH_REGISTRY": _Registry()},
    "basicsr.archs": {},
    "basicsr.archs.arch_util": {"to_2tuple": to_2tuple, "trunc_normal_": trunc_normal_},
    "einops": {"rearrange": None},
}


def build(checkout: Path, weights: Path, params: dict, *, device: torch.device) -> torch.nn.Module:
    """Upstream `DRCT` with `params` (the card's) and the checkpoint's `params_ema` weights, in eval mode."""
    if "drct_arch" not in sys.modules:
        for name, attrs in STUBS.items():
            sys.modules[name] = mod = types.ModuleType(name)
            mod.__dict__.update(attrs)
        path = reference_tree(checkout, REF_COMMIT) / "drct" / "archs" / "DRCT_arch.py"
        spec = importlib.util.spec_from_file_location("drct_arch", path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        sys.modules["drct_arch"] = module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    net = sys.modules["drct_arch"].DRCT(
        upscale=params["scale"], in_chans=3, img_size=params["img_size"], window_size=WINDOW, compress_ratio=3,
        squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5, img_range=1.0, depths=[6] * len(params["heads"]),
        embed_dim=params["embed_dim"], num_heads=params["heads"], mlp_ratio=2, upsampler="pixelshuffle",
        resi_connection="1conv",
    )  # fmt: skip
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)[params["key"]], strict=True)
    return net.eval().to(device)


def run(net: torch.nn.Module, x: torch.Tensor, scale: int) -> torch.Tensor:
    """`[1, 3, H, W]` float in [0, 1] -> `[3, sH, sW]` float32 in [0, 1] (`DRCTModel` test, clamped)."""
    h, w = x.shape[-2:]
    x = F.pad(x, (0, -w % WINDOW, 0, -h % WINDOW), "reflect")
    with torch.inference_mode():
        return net(x)[0, :, : h * scale, : w * scale].float().clamp(0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of ming053l/DRCT")
    ap.add_argument("weights", type=Path, help="directory with the .pth files")
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
            settings=f"CPU float32, drct_model.py test: reflect padding to a multiple of 16, net_g_ema(img), crop, "
            f"clamped, no uint8 rounding; tests/synth.py image{SIZE}; torch {torch.__version__}",
            task="restoration",
        )
        print(f"  {variant.id}: image {out.shape}, mean {out.mean():.4f}")


if __name__ == "__main__":
    main()
