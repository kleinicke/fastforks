"""EMA-VFI frame-interpolation goldens from the reference implementation.

Reference: upstream MCG-NJU/EMA-VFI (`REF_COMMIT`) with `patches/ema_vfi-cpu.patch` (the time tensor on the input's
device instead of `.cuda()`): `Trainer.Model` with `config.init_model_config` of the variant (F 32, depths
[2, 2, 2, 4, 4]; small: F 16, depths [2, 2, 2, 2, 2]) and the checkpoint loaded by `Model.load_model` (strict,
`module.` prefixes and the cached shift masks dropped), called as `benchmark/MiddleBury.py` (InputPadder 32,
replicate, centred; `Model.inference` without test-time augmentation, timestep 0.5) on the CPU in float32. The frames
go in as BGR, as the benchmarks' `cv2.imread`. Inputs: the RAFT goldens' pair (tools/golden/raft_flow.py: a 384x512
astronaut.png crop and its affine warp). Stored as float32: `input` (image1, RGB), `input2` (image2, RGB) and
`frame` ([3, H, W] RGB in [0, 1]).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/ema_vfi.py EMA_VFI_CHECKOUT WEIGHTS_DIR [ema-vfi ...]

Needs timm (upstream's imports). Run with `CUDA_VISIBLE_DEVICES=`.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from _common import reference_tree, write
from raft_flow import CROP, IMAGE_SHA256, warp

from fastforks._core import registry

FAMILY = "ema_vfi"
UPSTREAM = "https://github.com/MCG-NJU/EMA-VFI"
REF_COMMIT = "75b6f6a889e695df875e103374040d47a4cfac7c"
PATCH = "ema_vfi-cpu.patch"


def build(tree: Path, weights: Path, width: int, depths: list[int], device: str = "cpu") -> Any:  # noqa: ANN401
    """Upstream `MultiScaleFlow` of `Trainer.Model` (config F `width`, `depths`) with `weights` as `load_model`."""
    sys.path.insert(0, str(tree))
    import config as cfg  # noqa: PLC0415  # reference checkout

    backbone, multiscale = cfg.MODEL_CONFIG["MODEL_TYPE"]
    bcfg, mcfg = cfg.init_model_config(F=width, depth=depths)
    net = multiscale(backbone(**bcfg), **mcfg)
    sd = torch.load(weights, map_location="cpu")
    keep = {k: v for k, v in sd.items() if "module." in k and "attn_mask" not in k and "HW" not in k}
    net.load_state_dict({k.replace("module.", ""): v for k, v in keep.items()})  # Trainer.Model.load_model's convert
    return net.eval().to(device)


def interpolate(net: Any, image1: torch.Tensor, image2: torch.Tensor, timestep: float = 0.5) -> torch.Tensor:  # noqa: ANN401
    """Upstream's frame of an RGB `[1, 3, H, W]` pair in [0, 1] (BGR in and out, padded and cropped as benchmarks)."""
    h, w = image1.shape[-2:]  # benchmark/utils/padder.py InputPadder(shape, divisor=32)
    ph, pw = -h % 32, -w % 32
    pad = [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2]
    i0, i2 = (F.pad(x.flip(1), pad, mode="replicate") for x in (image1, image2))
    with torch.no_grad():  # Trainer.Model.inference, TTA False
        pred = net(torch.cat((i0, i2), 1), timestep=timestep)[3]
    return pred.flip(1)[..., pad[2] : pad[2] + h, pad[0] : pad[0] + w]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of MCG-NJU/EMA-VFI")
    ap.add_argument("weights", type=Path, help="directory with the variants' checkpoints (card weight names)")
    ap.add_argument("variants", nargs="*", help="variant ids (default: all)")
    a = ap.parse_args()
    if torch.cuda.is_available():
        msg = "run with CUDA_VISIBLE_DEVICES= (the reference runs on the CPU)"
        raise SystemExit(msg)
    import skimage  # noqa: PLC0415

    png = Path(skimage.data.__file__).parent / "astronaut.png"
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's astronaut.png"
        raise SystemExit(msg)
    y0, x0, h, w = CROP
    img1 = np.ascontiguousarray(skimage.io.imread(png)[y0 : y0 + h, x0 : x0 + w, :3])
    img2 = warp(img1)
    t1, t2 = (torch.from_numpy(i).permute(2, 0, 1)[None].float() / 255 for i in (img1, img2))
    tree = reference_tree(a.checkout, REF_COMMIT, PATCH)
    for card, variant in registry.cards("interpolation").values():
        if card.family != FAMILY or (a.variants and variant.id not in a.variants):
            continue
        p = {**card.params, **variant.params}
        net = build(tree, a.weights / variant.weights[0].name, p["width"], list(p["depths"]))
        frame = interpolate(net, t1, t2)[0].numpy()
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img1.astype(np.float32), "input2": img2.astype(np.float32), "frame": frame.astype(np.float32)},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=PATCH,
            settings=f"CPU float32, upstream MultiScaleFlow as Model.inference (no TTA), timestep 0.5, BGR, replicate "
            f"pad to 32 centred, skimage astronaut.png {h}x{w} crop + affine warp as raft_flow.py, "
            f"torch {torch.__version__}",
            task="interpolation",
        )
        print(f"  {variant.id}: frame mean {frame.mean():.4f}, range [{frame.min():.3f}, {frame.max():.3f}]")


if __name__ == "__main__":
    main()
