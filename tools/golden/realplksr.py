"""RealPLKSR super-resolution goldens from the reference implementation.

Reference: upstream neosr-project/neosr (`REF_COMMIT`) `neosr/archs/realplksr_arch.py` `realplksr` in eval mode, the
test call `net(lq)` on float RGB in [0, 1], clamped to [0, 1] without the uint8 rounding, on the CPU in float32. Input:
`tests/synth.py image(101, 151)`, stored as `input`; output `image` `[3, sH, sW]`.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/realplksr.py NEOSR_CHECKOUT WEIGHTS_DIR [ids]

WEIGHTS_DIR holds the weight files named as the card. Run with `CUDA_VISIBLE_DEVICES=`. neosr's package parses the
training options at import, so `realplksr_arch.py` is loaded by path, unmodified, with stubs for its two neosr imports
(`STUBS`; DySample is not used by the card's models).
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
from safetensors.torch import load_file

from fastforks._core import registry

FAMILY = "realplksr"
UPSTREAM = "https://github.com/neosr-project/neosr"
REF_COMMIT = "31c7022620c682cf0961c8634d60787179145c5b"
SIZE = (101, 151)
STUBS = {  # module -> attributes realplksr_arch.py imports
    "neosr": {},
    "neosr.archs": {},
    "neosr.archs.arch_util": {"DySample": None, "net_opt": lambda: (4, False)},
    "neosr.utils": {},
    "neosr.utils.registry": {"ARCH_REGISTRY": types.SimpleNamespace(register=lambda: lambda f: f)},
}


def build(checkout: Path, weights: Path, params: dict, *, device: torch.device) -> torch.nn.Module:
    """Upstream `realplksr` with `params` and the weights, in eval mode."""
    if "realplksr_arch" not in sys.modules:
        for name, attrs in STUBS.items():
            sys.modules.setdefault(name, types.ModuleType(name)).__dict__.update(attrs)
        path = reference_tree(checkout, REF_COMMIT) / "neosr" / "archs" / "realplksr_arch.py"
        spec = importlib.util.spec_from_file_location("realplksr_arch", path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        sys.modules["realplksr_arch"] = module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    net = sys.modules["realplksr_arch"].realplksr(
        dim=params["dim"],
        n_blocks=params["n_blocks"],
        upscaling_factor=params["scale"],
        kernel_size=params["kernel_size"],
        split_ratio=params["split_ratio"],
        use_ea=params["use_ea"],
    )
    net.load_state_dict(load_file(weights), strict=True)
    return net.eval().to(device)


def run(net: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """`[1, 3, H, W]` float in [0, 1] -> `[3, sH, sW]` float32 in [0, 1] (`net(lq)`, clamped)."""
    with torch.inference_mode():
        return net(x)[0].float().clamp(0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of neosr-project/neosr")
    ap.add_argument("weights", type=Path, help="directory with the weight files")
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
            settings=f"CPU float32, upstream realplksr eval mode, net(lq) on float RGB, clamped, no uint8 rounding; "
            f"tests/synth.py image{SIZE}; torch {torch.__version__}",
            task="restoration",
        )
        print(f"  {variant.id}: image {out.shape}, mean {out.mean():.4f}")


if __name__ == "__main__":
    main()
