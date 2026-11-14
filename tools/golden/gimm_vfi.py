"""GIMM-VFI frame-interpolation goldens from the reference implementation.

Reference: upstream GSeanCDAT/GIMM-VFI (`REF_COMMIT`) with `patches/gimm_vfi-cpu-softsplat.patch` (the cupy splatting
kernel's CPU branch in numpy, float64 sums; no raft-things.pth load, the RAFT weights are in the checkpoint):
`GIMMVFI_R(GIMMVFIConfig.create(configs/gimmvfi/gimmvfi_r_arb.yaml arch))` with the checkpoint's `state_dict`
(strict), called as `src/video_Nx.py` does (InputPadder 32, one time 0.5, `ds_factor` 1.0) on the CPU in float32.
Inputs: the RAFT goldens' pair (tools/golden/raft_flow.py: a 384x512 astronaut.png crop and its affine warp).
Stored as float32: `input` (image1, RGB), `input2` (image2, RGB) and `frame` ([3, H, W] in [0, 1]).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/gimm_vfi.py GIMM_VFI_CHECKOUT WEIGHTS_DIR [gimm-vfi-r ...]

Needs omegaconf, einops and easydict (upstream's imports). Run with `CUDA_VISIBLE_DEVICES=`.
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

FAMILY = "gimm_vfi"
UPSTREAM = "https://github.com/GSeanCDAT/GIMM-VFI"
REF_COMMIT = "dbc56449994a3c2e045d46fd46bed570239913f8"
PATCH = "gimm_vfi-cpu-softsplat.patch"


def build(tree: Path, weights: Path) -> Any:  # noqa: ANN401  # upstream's GIMMVFI_R, imported from the checkout
    """Upstream `GIMMVFI_R` from its config with `weights` loaded as video_Nx.py does (strict), in eval mode."""
    sys.path.insert(0, str(tree / "src"))
    from models.generalizable_INR.configs import GIMMVFIConfig  # noqa: PLC0415  # reference checkout
    from models.generalizable_INR.gimmvfi_r import GIMMVFI_R  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415

    config = GIMMVFIConfig.create(OmegaConf.load(tree / "configs/gimmvfi/gimmvfi_r_arb.yaml").arch)
    model = GIMMVFI_R(config)
    model.load_state_dict(torch.load(weights, map_location="cpu")["state_dict"], strict=True)
    return model.eval()


def interpolate(
    model: Any,  # noqa: ANN401  # upstream's GIMMVFI_R
    image1: torch.Tensor,
    image2: torch.Tensor,
) -> torch.Tensor:
    """Upstream's middle frame of a `[1, 3, H, W]` pair in [0, 1], padded, called and cropped as video_Nx.py."""
    h, w = image1.shape[-2:]  # src/utils/utils.py InputPadder(shape, 32) (its module imports tensorboard)
    ph, pw = -h % 32, -w % 32
    pad = [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2]
    i0, i2 = (F.pad(x, pad, mode="replicate") for x in (image1, image2))
    xs = torch.cat((i0.unsqueeze(2), i2.unsqueeze(2)), dim=2)
    with torch.no_grad():
        coord = [(model.sample_coord_input(1, xs.shape[-2:], [0.5], device=xs.device, upsample_ratio=1.0), None)]
        out = model(xs, coord, t=[0.5 * torch.ones(1, device=xs.device)], ds_factor=1.0)
    return out["imgt_pred"][0][..., pad[2] : pad[2] + h, pad[0] : pad[0] + w]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help="git checkout of GSeanCDAT/GIMM-VFI")
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
        frame = interpolate(build(tree, a.weights / variant.weights[0].name), t1, t2)[0].numpy()
        write(
            f"{variant.id}@{variant.revision}",
            {"input": img1.astype(np.float32), "input2": img2.astype(np.float32), "frame": frame.astype(np.float32)},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=PATCH,
            settings=f"CPU float32, upstream GIMMVFI_R forward as video_Nx.py, time 0.5, ds_factor 1.0, replicate "
            f"pad to 32 centred, skimage astronaut.png {h}x{w} crop + affine warp as raft_flow.py, "
            f"torch {torch.__version__}",
            task="interpolation",
        )
        print(f"  {variant.id}: frame mean {frame.mean():.4f}, range [{frame.min():.3f}, {frame.max():.3f}]")


if __name__ == "__main__":
    main()
