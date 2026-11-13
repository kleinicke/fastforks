"""Real-ESRGAN super-resolution goldens from the reference implementation.

Reference: upstream xinntao/Real-ESRGAN (`REF_COMMIT`): `RealESRGANer` (`half=False`, `pre_pad=0`, no tile, as
`inference_realesrgan.py --fp32`) with BasicSR's `RRDBNet` (`BASICSR_COMMIT`), on the CPU in float32. The image goes in
as float RGB in [0, 1] (upstream's `enhance` divides the uint8 image by 255 and rounds the output to uint8; float keeps
the golden free of rounding): `pre_process` (x2: reflect padding to even sizes), `process`, `post_process`, clamped to
[0, 1]. Input: `tests/synth.py image(101, 151)` (odd sizes: the x2 padding runs), stored as `input`; output `image`
`[3, sH, sW]`.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/realesrgan.py REAL_ESRGAN_CHECKOUT BASICSR_CHECKOUT WEIGHTS_DIR [ids]

WEIGHTS_DIR holds the release `.pth` files named as the card. Run with `CUDA_VISIBLE_DEVICES=`. Upstream's package
imports (BasicSR's compiled deformable-conv ops, `distutils`, the training code) do not load on Python 3.12, so the
three files used (`rrdbnet_arch.py`, `arch_util.py`, `realesrgan/utils.py`) are loaded by path, unmodified, with stub
modules for those imports (`STUBS`).
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

FAMILY = "realesrgan"
UPSTREAM = "https://github.com/xinntao/Real-ESRGAN"
REF_COMMIT = "a4abfb2979a7bbff3f69f58f58ae324608821e27"
BASICSR_COMMIT = "8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a"
SIZE = (101, 151)
STUBS = {  # module -> attributes the three files import but the inference path never calls
    "basicsr": {},
    "basicsr.archs": {},
    "basicsr.ops": {},
    "basicsr.ops.dcn": {"ModulatedDeformConvPack": object, "modulated_deform_conv": None},
    "basicsr.utils": {"get_root_logger": None},
    "basicsr.utils.registry": {"ARCH_REGISTRY": types.SimpleNamespace(register=lambda: lambda cls: cls)},
    "basicsr.utils.download_util": {"load_file_from_url": None},
    "distutils": {},
    "distutils.version": {"LooseVersion": str},
}


def load(name: str, path: Path) -> types.ModuleType:
    """Execute the upstream file `path` as module `name`."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build(  # noqa: PLR0913 - the reference's settings
    checkout: Path, basicsr: Path, weights: Path, params: dict, *, half: bool, device: torch.device
) -> object:
    """Upstream's `RealESRGANer` (`pre_pad` 0, no tile) on BasicSR's `RRDBNet` with `params` (`scale`, `num_block`)."""
    if "realesrgan_utils" not in sys.modules:
        for name, attrs in STUBS.items():
            sys.modules[name] = types.ModuleType(name)
            sys.modules[name].__dict__.update(attrs)
        archs = reference_tree(basicsr, BASICSR_COMMIT) / "basicsr" / "archs"
        load("basicsr.archs.arch_util", archs / "arch_util.py")
        load("basicsr.archs.rrdbnet_arch", archs / "rrdbnet_arch.py")
        load("realesrgan_utils", reference_tree(checkout, REF_COMMIT) / "realesrgan" / "utils.py")
    net = sys.modules["basicsr.archs.rrdbnet_arch"].RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=params["num_block"], num_grow_ch=32, scale=params["scale"]
    )
    return sys.modules["realesrgan_utils"].RealESRGANer(
        scale=params["scale"], model_path=str(weights), model=net, pre_pad=0, half=half, device=device
    )


def upscale(upsampler: object, img: np.ndarray) -> np.ndarray:
    """`RealESRGANer.enhance` without the uint8 conversions: `[H, W, 3]` float RGB in [0, 1] -> `[3, sH, sW]`."""
    return run(upsampler, img).cpu().numpy()


def run(upsampler: object, img: np.ndarray) -> torch.Tensor:
    """`pre_process`, `process`, `post_process` of `[H, W, 3]` float RGB in [0, 1]: `[3, sH, sW]` float in [0, 1]."""
    with torch.inference_mode():
        upsampler.pre_process(img)
        upsampler.process()
        return upsampler.post_process()[0].float().clamp(0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of xinntao/Real-ESRGAN")
    ap.add_argument("basicsr", type=Path, help="git checkout of XPixelGroup/BasicSR")
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
        weights = a.weights / variant.weights[0].name
        out = upscale(build(a.checkout, a.basicsr, weights, p, half=False, device=torch.device("cpu")), x)
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img.astype(np.float32), "image": out},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, upstream RealESRGANer pre_pad 0 no tile (BasicSR RRDBNet @{BASICSR_COMMIT[:8]}) "
            f"on float RGB, clamped, no uint8 rounding; tests/synth.py image{SIZE}; torch {torch.__version__}",
            task="restoration",
        )
        print(f"  {variant.id}: image {out.shape}, mean {out.mean():.4f}")


if __name__ == "__main__":
    main()
