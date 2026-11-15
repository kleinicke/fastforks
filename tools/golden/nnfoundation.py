"""nnFoundation goldens from the networks nnssl builds (`COMMIT`): dynamic_network_architectures 0.4.4 + timm.

Reference: nnssl's `get_res_enc_l` (ResidualEncoderUNet, its `.encoder`) and `get_network_by_name("PrimusX")` (dna
`Primus` with the checkpoint's adaptation-plan kwargs; `.eva` on the patch tokens), CPU float32, the encoder weights of
the checkpoint (`network_weights`, `encoder.*` / `down_projection.*` + `eva.*`). Input: a 40 x 56 x 72 voxel crop of
TotalSegmentator dataset case s0013 (CC-BY-4.0; `CT_SHA256`), RAS-canonical `(z, y, x)`, nnU-Net's
`ZScoreNormalization(use_mask_for_norm=False)`, zeros at the far end of every axis (CNN: to a multiple of 32; ViT: to
192^3), features cropped to `ceil(n / stride)`. Stored: `input` (Hounsfield units), `features` (the deepest map) and
`features.<i>` (CNN stages 1..4; stage 0, 32 x 40 x 56 x 72, is left out for size and checked through stage 1).

    FASTFORKS_GOLDEN_PATH=... <reference venv>/bin/python tools/golden/nnfoundation.py CT WEIGHTS_DIR
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from dynamic_network_architectures.architectures.primus import Primus
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from einops import rearrange
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import write

UPSTREAM = "https://github.com/MIC-DKFZ/nnssl"
COMMIT = "3d774e942256bd70e5e264c6b94fa38e25f7ed74"
CT_SHA256 = "33a235934cfee4e8534121ca9bfc2e81ca8d4ee62039ed34349d5b753160cba9"  # dataset s0013/ct.nii.gz
CROP = (40, 56, 72)
VARIANTS = {"nnfoundation-cnn@1": "nnFoundationCNN", "nnfoundation-vit@1": "nnFoundationViT"}


def crop(path: Path) -> np.ndarray:
    """The centre crop `[D, H, W]` (float32, RAS canonical)."""
    img = nib.as_closest_canonical(nib.load(path))
    vol = np.asarray(img.dataobj, dtype=np.float32).transpose(2, 1, 0)
    start = [(n - c) // 2 for n, c in zip(vol.shape, CROP, strict=True)]
    return np.ascontiguousarray(vol[tuple(slice(s, s + c) for s, c in zip(start, CROP, strict=True))])


def res_enc_l() -> ResidualEncoderUNet:
    """The ResEnc-L of nnssl's `get_res_enc_l(1, 1)` (architectures/architecture_registry.py)."""
    n = 6
    return ResidualEncoderUNet(
        input_channels=1, n_stages=n, features_per_stage=[32, 64, 128, 256, 320, 320], conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * n, strides=[[1, 1, 1]] + [[2, 2, 2]] * 5, n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
        num_classes=1, n_conv_per_stage_decoder=[1, 1, 1, 1, 1], conv_bias=True, norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True}, nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True},
    )  # fmt: skip


def primus_x() -> Primus:
    """`get_network_by_name("PrimusX", ...)` with the nnFoundationViT adaptation plan's kwargs."""
    return Primus(
        input_channels=1, embed_dim=1056, num_classes=1, eva_depth=40, eva_numheads=16, input_shape=(192, 192, 192),
        patch_embed_size=(8, 8, 8), init_values=0.1, scale_attn_inner=True,
    )  # fmt: skip


def encoder_state(path: Path, prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    sd = torch.load(path, map_location="cpu", weights_only=True)["network_weights"]
    return {k: v for k, v in sd.items() if k.startswith(prefixes)}


def features(arch: str, weights: Path, vol: np.ndarray) -> list[np.ndarray]:
    x = ZScoreNormalization(use_mask_for_norm=False, intensityproperties={}).run(vol.copy(), None)[None]
    target = [-(-n // 32) * 32 for n in CROP] if arch == "cnn" else [192] * 3
    t = F.pad(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), [
        v for n, m in zip(reversed(CROP), reversed(target), strict=True) for v in (0, m - n)
    ])[None]  # fmt: skip
    with torch.no_grad():
        if arch == "cnn":
            net = res_enc_l().eval()
            sd = {k.removeprefix("encoder."): v for k, v in encoder_state(weights, ("encoder.",)).items()}
            net.encoder.load_state_dict(sd, strict=True)
            maps, strides = net.encoder(t), (1, 2, 4, 8, 16, 32)
        else:
            net = primus_x().eval()
            missing, unexpected = net.load_state_dict(
                encoder_state(weights, ("down_projection.", "eva.")), strict=False
            )
            if unexpected or not all(k.startswith(("up_projection.", "mask_token")) for k in missing):
                msg = f"unexpected checkpoint keys: {unexpected or missing}"
                raise SystemExit(msg)
            tok, _ = net.eva(rearrange(net.down_projection(t), "b c w h d -> b (w h d) c"))
            maps, strides = [rearrange(tok, "b (w h d) c -> b c w h d", w=24, h=24, d=24)], (8,)
    return [
        m[0][(slice(None), *(slice(0, -(-n // s)) for n in CROP))].numpy() for m, s in zip(maps, strides, strict=True)
    ]


def main() -> None:
    ct, root = Path(sys.argv[1]), Path(sys.argv[2])
    if hashlib.sha256(ct.read_bytes()).hexdigest() != CT_SHA256:
        msg = f"{ct} is not the TotalSegmentator dataset's s0013/ct.nii.gz"
        raise SystemExit(msg)
    vol = crop(ct)
    torch.set_num_threads(16)
    for ref, folder in VARIANTS.items():
        path = root / folder / "checkpoint_final.pth"
        if not path.is_file():
            print(f"  {ref}: {path} missing, skipped")
            continue
        *rest, last = features("cnn" if "cnn" in ref else "vit", path, vol)
        maps = {"input": vol, "features": last} | {f"features.{i}": m for i, m in enumerate(rest) if i > 0}
        write(
            ref,
            maps,
            family="nnfoundation",
            upstream=f"{UPSTREAM}@{COMMIT}",
            patch=None,
            settings=(
                f"CPU float32, nnssl's network ({folder}) encoder on s0013 ct.nii.gz, centre crop {CROP} (z, y, x) "
                f"RAS, nnU-Net ZScoreNormalization, zero pad at the far end, torch {torch.__version__}"
            ),
            task="volume",
        )
        print(f"  {ref}: " + ", ".join(f"{k} {v.shape}" for k, v in maps.items()))


if __name__ == "__main__":
    main()
