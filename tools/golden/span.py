"""SPAN super-resolution goldens from the reference implementation.

Reference: upstream hongyuanyu/SPAN (`REF_COMMIT`) `basicsr/archs/span_arch.py` `SPAN` in eval mode (its `Conv3XC`
re-fuses the training branches on every call), BasicSR's `SRModel.test` call: `net(lq)` on float RGB in [0, 1], here
clamped to [0, 1] without the uint8 rounding, on the CPU in float32. Input: `tests/synth.py image(101, 151)`, stored as
`input`; output `image` `[3, sH, sW]`.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/span.py SPAN_CHECKOUT WEIGHTS_DIR [ids]

WEIGHTS_DIR holds the `.pth` files of the README's checkpoint archive. Run with `CUDA_VISIBLE_DEVICES=`. The package
`basicsr` of the checkout imports compiled ops and `distutils`, so `span_arch.py` is loaded by path, unmodified, with a
stub for its one BasicSR import (`STUBS`).
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

FAMILY = "span"
UPSTREAM = "https://github.com/hongyuanyu/SPAN"
REF_COMMIT = "c77a5917759f09e66fbc7124220c5afc5ee221e5"
SIZE = (101, 151)
STUBS = {  # module -> attributes span_arch.py imports
    "basicsr": {},
    "basicsr.utils": {},
    "basicsr.utils.registry": {"ARCH_REGISTRY": types.SimpleNamespace(register=lambda: lambda cls: cls)},
}


def build(checkout: Path, weights: Path, params: dict, *, device: torch.device) -> torch.nn.Module:
    """Upstream `SPAN` with `params` (`scale`, `channels`) and the `params_ema` weights, in eval mode."""
    if "span_arch" not in sys.modules:
        for name, attrs in STUBS.items():
            sys.modules.setdefault(name, types.ModuleType(name)).__dict__.update(attrs)
        path = reference_tree(checkout, REF_COMMIT) / "basicsr" / "archs" / "span_arch.py"
        spec = importlib.util.spec_from_file_location("span_arch", path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        sys.modules["span_arch"] = module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    net = sys.modules["span_arch"].SPAN(3, 3, feature_channels=params["channels"], upscale=params["scale"])
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["params_ema"], strict=True)
    return net.eval().to(device)


def run(net: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """`[1, 3, H, W]` float in [0, 1] -> `[3, sH, sW]` float32 in [0, 1] (`SRModel.test`, clamped)."""
    with torch.inference_mode():
        return net(x)[0].float().clamp(0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of hongyuanyu/SPAN")
    ap.add_argument("weights", type=Path, help="directory with the checkpoint archive's .pth files")
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
        out = run(build(a.checkout, a.weights / variant.weights[0].name, p, device=torch.device("cpu")), x).numpy()
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img.astype(np.float32), "image": out},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, upstream SPAN eval mode, net(lq) on float RGB, clamped, no uint8 rounding; "
            f"tests/synth.py image{SIZE}; torch {torch.__version__}",
            task="restoration",
        )
        print(f"  {variant.id}: image {out.shape}, mean {out.mean():.4f}")


if __name__ == "__main__":
    main()
