"""DAT super-resolution goldens from the reference implementation.

Reference: upstream zhengchen1999/DAT (`REF_COMMIT`) `basicsr/archs/dat_arch.py` `DAT` as the test options build it for
the variant, and `basicsr/models/dat_model.py`'s test call (`use_chop` False): `net_g(img)`, clamped to [0, 1] (without
the uint8 rounding), on the CPU in float32. Input: `tests/synth.py image(101, 151)` stored as `input`; output `image`
`[3, sH, sW]`.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/dat.py DAT_CHECKOUT WEIGHTS_DIR [ids]

WEIGHTS_DIR holds the Google Drive `.pth` files named as the card. Run with `CUDA_VISIBLE_DEVICES=`. `dat_arch.py` is
loaded by path, unmodified; it needs `einops` and timm, and its BasicSR import (the registry) is a stub module
(`STUBS`: BasicSR's package does not import on Python 3.12).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
from _common import image, reference_tree, write

from fastforks._core import registry

FAMILY = "dat"
UPSTREAM = "https://github.com/zhengchen1999/DAT"
REF_COMMIT = "1645c64fad7c94d94abd090bd1319eec1f26f7cc"
SIZE = (101, 151)


class _Registry:
    def register(self) -> object:
        return lambda cls: cls


STUBS = {
    "basicsr": {},
    "basicsr.utils": {},
    "basicsr.utils.registry": {"ARCH_REGISTRY": _Registry()},
}


def build(checkout: Path, weights: Path, params: dict, *, device: torch.device) -> torch.nn.Module:
    """Upstream `DAT` with `params` (the card's) and the checkpoint's `params` weights, in eval mode."""
    if "dat_arch" not in sys.modules:
        for name, attrs in STUBS.items():
            sys.modules[name] = mod = types.ModuleType(name)
            mod.__dict__.update(attrs)
        path = reference_tree(checkout, REF_COMMIT) / "basicsr" / "archs" / "dat_arch.py"
        spec = importlib.util.spec_from_file_location("dat_arch", path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        sys.modules["dat_arch"] = module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    net = sys.modules["dat_arch"].DAT(
        upscale=params["scale"], in_chans=3, img_size=params["img_size"], img_range=1.0, split_size=params["split"],
        depth=params["depths"], embed_dim=params["embed_dim"], num_heads=params["heads"],
        expansion_factor=params["expansion"], resi_connection=params["resi"], upsampler=params["upsampler"],
    )  # fmt: skip
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)[params["key"]], strict=True)
    return net.eval().to(device)


def run(net: torch.nn.Module, x: torch.Tensor, scale: int) -> torch.Tensor:
    """`[1, 3, H, W]` float in [0, 1] -> `[3, sH, sW]` float32 in [0, 1] (`DATModel` test, clamped)."""
    with torch.inference_mode():
        out = net(x)[0].float().clamp(0, 1)
    if out.shape[-2:] != (x.shape[-2] * scale, x.shape[-1] * scale):
        raise RuntimeError(out.shape)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of zhengchen1999/DAT")
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
            settings=f"CPU float32, dat_model.py test: net_g(img) (no chop), clamped, no uint8 rounding; "
            f"tests/synth.py image{SIZE}; torch {torch.__version__}",
            task="restoration",
        )
        print(f"  {variant.id}: image {out.shape}, mean {out.mean():.4f}")


if __name__ == "__main__":
    main()
