"""CodeFormer face restoration goldens from the reference implementation.

Reference: upstream sczhou/CodeFormer (`REF_COMMIT`) `basicsr/archs/codeformer_arch.py` `CodeFormer` as
`inference_codeformer.py` builds it, and its call on an aligned face: `normalize(0.5, 0.5)` to [-1, 1],
`net(x, w=0.5, adain=True)[0]`, mapped back from [-1, 1] and clamped (`tensor2img(min_max=(-1, 1))` without the uint8
rounding), on the CPU in float32. Input: the repository's aligned face `inputs/cropped_faces/0143.png` (512 x 512)
stored as `input`; output `image` `[3, 512, 512]`.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/codeformer.py CODEFORMER_CHECKOUT WEIGHTS_DIR

WEIGHTS_DIR holds `codeformer.pth` (the v0.1.0 release). Run with `CUDA_VISIBLE_DEVICES=`. The two arch files are
loaded by path, unmodified; their BasicSR imports (the registry, the logger) are stub modules (`STUBS`).
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import types
from pathlib import Path

import numpy as np
import torch
from _common import reference_tree, write
from PIL import Image

from fastforks._core import registry

FAMILY = "codeformer"
UPSTREAM = "https://github.com/sczhou/CodeFormer"
REF_COMMIT = "b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b"
FACE = "inputs/cropped_faces/0143.png"


class _Registry:
    def register(self) -> object:
        return lambda cls: cls


STUBS = {
    "basicsr": {},
    "basicsr.utils": {"get_root_logger": logging.getLogger},
    "basicsr.utils.registry": {"ARCH_REGISTRY": _Registry()},
    "basicsr.archs": {},
}


def build(tree: Path, weights: Path, params: dict) -> torch.nn.Module:
    """Upstream `CodeFormer` (`inference_codeformer.py`'s arguments) with the checkpoint's `params_ema`."""
    for name, attrs in STUBS.items():
        sys.modules[name] = mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
    for name in ("vqgan_arch", "codeformer_arch"):
        path = tree / "basicsr" / "archs" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"basicsr.archs.{name}", path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        sys.modules[f"basicsr.archs.{name}"] = module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    net = sys.modules["basicsr.archs.codeformer_arch"].CodeFormer(
        dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=["32", "64", "128", "256"]
    )
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)[params["key"]], strict=True)
    return net.eval()


def run(net: torch.nn.Module, x: torch.Tensor, w: float) -> torch.Tensor:
    """`[1, 3, 512, 512]` float in [0, 1] -> `[3, 512, 512]` float32 in [0, 1] (`inference_codeformer.py`'s call)."""
    with torch.inference_mode():
        return (net(x * 2 - 1, w=w, adain=True)[0][0].float().clamp(-1, 1) + 1) / 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of sczhou/CodeFormer")
    ap.add_argument("weights", type=Path, help="directory with codeformer.pth")
    a = ap.parse_args()
    if torch.cuda.is_available():
        msg = "run with CUDA_VISIBLE_DEVICES= (the reference runs on the CPU)"
        raise SystemExit(msg)
    tree = reference_tree(a.checkout, REF_COMMIT)
    img = np.asarray(Image.open(tree / FACE).convert("RGB"))
    for card, variant in registry.cards("restoration").values():
        if card.family != FAMILY:
            continue
        p = {**card.params, **variant.params}
        net = build(tree, a.weights / variant.weights[0].name, p)
        x = torch.from_numpy(img.astype(np.float32) / 255).permute(2, 0, 1)[None]
        out = run(net, x, p["fidelity"]).numpy()
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img.astype(np.float32), "image": out},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, inference_codeformer.py on an aligned face: [-1, 1], net(x, w={p['fidelity']}, "
            f"adain=True)[0], back to [0, 1], clamped, no uint8 rounding; {FACE}; torch {torch.__version__}",
            task="restoration",
        )
        print(f"  {variant.id}: image {out.shape}, mean {out.mean():.4f}")


if __name__ == "__main__":
    main()
