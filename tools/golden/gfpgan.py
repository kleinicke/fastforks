"""GFPGAN v1.4 face restoration goldens from the reference implementation.

Reference: upstream TencentARC/GFPGAN (`REF_COMMIT`) `gfpgan/archs/gfpganv1_clean_arch.py` `GFPGANv1Clean` as
`GFPGANer` builds it (arch "clean"), and `GFPGANer.enhance`'s call on an aligned face: `normalize(0.5, 0.5)` to
[-1, 1], `net(x, return_rgb=False)[0]` with the checkpoint's stored noise (`randomize_noise=False`; upstream's default
draws new noise per call), mapped back from [-1, 1] and clamped (`tensor2img(min_max=(-1, 1))` without the uint8
rounding), on the CPU in float32. Input: the repository's aligned face `inputs/cropped_faces/Adele_crop.png`
(512 x 512) stored as `input`; output `image` `[3, 512, 512]`.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/gfpgan.py GFPGAN_CHECKOUT WEIGHTS_DIR

WEIGHTS_DIR holds `GFPGANv1.4.pth` (the v1.3.0 release). Run with `CUDA_VISIBLE_DEVICES=`. The two arch files are
loaded by path, unmodified; their BasicSR imports (the registry, `default_init_weights`) are stub modules (`STUBS`).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
from _common import reference_tree, write
from PIL import Image

from fastforks._core import registry

FAMILY = "gfpgan"
UPSTREAM = "https://github.com/TencentARC/GFPGAN"
REF_COMMIT = "7552a7791caad982045a7bbe5634bbf1cd5c8679"
FACE = "inputs/cropped_faces/Adele_crop.png"


class _Registry:
    def register(self) -> object:
        return lambda cls: cls


STUBS = {
    "basicsr": {},
    "basicsr.utils": {},
    "basicsr.utils.registry": {"ARCH_REGISTRY": _Registry()},
    "basicsr.archs": {},
    "basicsr.archs.arch_util": {"default_init_weights": lambda *_a, **_k: None},
    "gfpgan": {},
    "gfpgan.archs": {},
}


def build(tree: Path, weights: Path, params: dict) -> torch.nn.Module:
    """Upstream `GFPGANv1Clean` (`GFPGANer`'s clean arguments) with the checkpoint's `params_ema`."""
    for name, attrs in STUBS.items():
        sys.modules[name] = mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
    for name in ("stylegan2_clean_arch", "gfpganv1_clean_arch"):
        path = tree / "gfpgan" / "archs" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"gfpgan.archs.{name}", path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        sys.modules[f"gfpgan.archs.{name}"] = module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    net = sys.modules["gfpgan.archs.gfpganv1_clean_arch"].GFPGANv1Clean(
        out_size=params["size"], num_style_feat=512, channel_multiplier=2, decoder_load_path=None, fix_decoder=False,
        num_mlp=8, input_is_latent=True, different_w=True, narrow=1, sft_half=True,
    )  # fmt: skip
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)[params["key"]], strict=True)
    return net.eval()


def run(net: torch.nn.Module, x: torch.Tensor, *, randomize_noise: bool = False) -> torch.Tensor:
    """`[1, 3, 512, 512]` float in [0, 1] -> `[3, 512, 512]` float32 in [0, 1] (`GFPGANer.enhance`'s call)."""
    with torch.inference_mode():
        out = net(x * 2 - 1, return_rgb=False, randomize_noise=randomize_noise)[0]
        return (out[0].float().clamp(-1, 1) + 1) / 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of TencentARC/GFPGAN")
    ap.add_argument("weights", type=Path, help="directory with GFPGANv1.4.pth")
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
        out = run(net, x).numpy()
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img.astype(np.float32), "image": out},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings="CPU float32, GFPGANer.enhance on an aligned face: [-1, 1], net(x, return_rgb=False, "
            f"randomize_noise=False)[0], back to [0, 1], clamped, no uint8 rounding; {FACE}; torch {torch.__version__}",
            task="restoration",
        )
        print(f"  {variant.id}: image {out.shape}, mean {out.mean():.4f}")


if __name__ == "__main__":
    main()
