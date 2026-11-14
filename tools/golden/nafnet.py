"""NAFNet restoration goldens from the reference implementation.

Reference: upstream megvii-research/NAFNet (`REF_COMMIT`) `basicsr/models/archs/NAFNet_arch.py` `NAFNet` /
`NAFNetLocal` as the variant's test options build them (GoPro: `NAFNetLocal`, train size 256; SIDD: `NAFNet`), and
`ImageRestorationModel.test`'s call `net_g(lq)` (the network pads to a multiple of 16 and crops itself), clamped to
[0, 1] (without the uint8 rounding), on the CPU in float32. Input: `tests/synth.py image(101, 151)` stored as `input`;
output `image` `[3, H, W]`.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/nafnet.py NAFNET_CHECKOUT WEIGHTS_DIR [ids]

WEIGHTS_DIR holds the released `.pth` files (`{"params": ...}`). Run with `CUDA_VISIBLE_DEVICES=`. The three arch
files are loaded by path, unmodified; their `basicsr` package imports are stubbed (only `get_root_logger` is used).
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
from _common import image, reference_tree, write

from fastforks._core import registry

FAMILY = "nafnet"
UPSTREAM = "https://github.com/megvii-research/NAFNet"
REF_COMMIT = "2b4af71ebe098a92a75910c233a3965a3e93ede4"
SIZE = (101, 151)


def _arch(checkout: Path) -> types.ModuleType:
    """Upstream `NAFNet_arch` with its `basicsr` imports resolved to the exported files (and a logger stub)."""
    name = "basicsr.models.archs.NAFNet_arch"
    if name in sys.modules:
        return sys.modules[name]
    archs = reference_tree(checkout, REF_COMMIT) / "basicsr" / "models" / "archs"
    for pkg in ("basicsr", "basicsr.models", "basicsr.models.archs"):
        sys.modules[pkg] = types.ModuleType(pkg)
    utils = sys.modules["basicsr.utils"] = types.ModuleType("basicsr.utils")
    utils.get_root_logger = logging.getLogger
    for mod in ("arch_util", "local_arch", "NAFNet_arch"):
        spec = importlib.util.spec_from_file_location(f"basicsr.models.archs.{mod}", archs / f"{mod}.py")
        if spec is None or spec.loader is None:
            raise ImportError(archs / f"{mod}.py")
        sys.modules[spec.name] = module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return sys.modules[name]


def build(checkout: Path, weights: Path, params: dict, *, device: torch.device) -> torch.nn.Module:
    """Upstream `NAFNetLocal` (`params["tlc"]` > 0) or `NAFNet` with the checkpoint's `params`, in eval mode."""
    arch = _arch(checkout)
    kw = {
        "width": params["width"], "enc_blk_nums": list(params["enc_blocks"]),
        "middle_blk_num": params["middle_blocks"], "dec_blk_nums": list(params["dec_blocks"]),
    }  # fmt: skip
    t = params["tlc"]
    net = arch.NAFNetLocal(**kw, train_size=(1, 3, t, t)) if t else arch.NAFNet(**kw)
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["params"], strict=True)
    return net.eval().to(device)


def run(net: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """`[1, 3, H, W]` float in [0, 1] -> `[3, H, W]` float32 in [0, 1] (`net_g(lq)`, clamped)."""
    with torch.inference_mode():
        return net(x)[0].float().clamp(0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of megvii-research/NAFNet")
    ap.add_argument("weights", type=Path, help="directory with the released .pth files")
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
        kind = f"NAFNetLocal (train size {p['tlc']})" if p["tlc"] else "NAFNet"
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img.astype(np.float32), "image": out},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, {kind}, net_g(lq), clamped, no uint8 rounding; tests/synth.py image{SIZE}; "
            f"torch {torch.__version__}",
            task="restoration",
        )
        print(f"  {variant.id}: image {out.shape}, mean {out.mean():.4f}")


if __name__ == "__main__":
    main()
