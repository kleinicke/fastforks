"""PointDiT goldens from the reference implementation.

Reference: upstream google-research/pointdit (`REF_COMMIT`): `Denoiser.generate` (3 Euler steps from zeros) on the
upstream network, with its DINOv3 encoder from the facebookresearch/dinov3 hub code (`DINOV3_COMMIT`) loaded from the
Hugging Face DINOv3 checkpoint (keys renamed to the hub layout). CPU float32, no autocast (upstream evaluates under
bf16 autocast on CUDA). The input goes through upstream's wild-image path (`engine.py`): aspect kept, resized
(bilinear, `align_corners=False`) to about 1024 16-px tokens, the point map resized back with nearest.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/pointdit.py POINTDIT_CHECKOUT DINOV3_CHECKOUT WEIGHTS_DIR [ref ...]

`WEIGHTS_DIR` holds the PointDiT `.pth` files and `dinov3-vit{b,l}16-pretrain-lvd1689m/model.safetensors`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from _common import image, reference_tree, write

UPSTREAM = "https://github.com/google-research/pointdit"
REF_COMMIT = "11f53a30aeb4d34d49d2beda655ee6f19b2ad7f8"
DINOV3_COMMIT = "6876159a"
TOKENS, STEPS, PATCH = 1024, 3, 16
VARIANTS = {  # ref: (upstream model, DINOv3, checkpoint)
    "pointdit-b-512": ("PointDiT-B/16", "vitb16", "pointditb-512-mixdata-nodinov3-1d42aacf.pth"),
    "pointdit-l-512": ("PointDiT-L/16", "vitl16", "pointditl-512-mixdata-nodinov3-240c1a4f.pth"),
}


def hub_state(hf: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Hugging Face DINOv3 ViT keys -> the hub `DinoVisionTransformer` keys."""
    out = {
        "cls_token": hf["embeddings.cls_token"],
        "storage_tokens": hf["embeddings.register_tokens"],
        "mask_token": hf["embeddings.mask_token"].reshape(1, -1),
        "patch_embed.proj.weight": hf["embeddings.patch_embeddings.weight"],
        "patch_embed.proj.bias": hf["embeddings.patch_embeddings.bias"],
        "norm.weight": hf["norm.weight"],
        "norm.bias": hf["norm.bias"],
    }
    n = 1 + max(int(k.split(".")[1]) for k in hf if k.startswith("layer."))
    for i in range(n):
        s, d = f"layer.{i}.", f"blocks.{i}."
        a = s + "attention."
        out[d + "attn.qkv.weight"] = torch.cat([hf[a + f"{x}_proj.weight"] for x in "qkv"])
        q_bias, v_bias = hf[a + "q_proj.bias"], hf[a + "v_proj.bias"]
        out[d + "attn.qkv.bias"] = torch.cat([q_bias, torch.zeros_like(q_bias), v_bias])
        out[d + "attn.proj.weight"], out[d + "attn.proj.bias"] = hf[a + "o_proj.weight"], hf[a + "o_proj.bias"]
        for x in ("norm1", "norm2"):
            out[d + x + ".weight"], out[d + x + ".bias"] = hf[s + x + ".weight"], hf[s + x + ".bias"]
        out[d + "ls1.gamma"], out[d + "ls2.gamma"] = hf[s + "layer_scale1.lambda1"], hf[s + "layer_scale2.lambda1"]
        for src, dst in (("up_proj", "fc1"), ("down_proj", "fc2")):
            out[d + f"mlp.{dst}.weight"], out[d + f"mlp.{dst}.bias"] = (
                hf[s + f"mlp.{src}.weight"],
                hf[s + f"mlp.{src}.bias"],
            )
    return out


def wild_size(h: int, w: int) -> tuple[int, int]:
    """Upstream `engine.py` wild-image size at `TOKENS`."""
    f = (TOKENS * PATCH**2 / (h * w)) ** 0.5
    return max(PATCH, round(h * f / PATCH) * PATCH), max(PATCH, round(w * f / PATCH) * PATCH)


def reference(repo: Path, dinov3: Path, ref: str, dino_file: Path, ckpt_file: Path, device: str = "cpu") -> Any:  # noqa: ANN401, PLR0913, PLR0917
    """Upstream `Denoiser` of `ref` (eval mode, on `device`) from the pinned checkouts and the two weight files."""
    if "denoiser" not in sys.modules:
        tree, dtree = reference_tree(repo, REF_COMMIT), reference_tree(dinov3, DINOV3_COMMIT)
        sys.path[:0] = [str(tree), str(dtree)]
        os.environ["DINOV3_REPO"] = str(dtree)  # upstream checks the directory exists
        from dinov3.hub import backbones  # noqa: PLC0415

        # upstream builds the encoder with torch.hub.load(<dinov3 checkout>, ...): the hub entry points directly
        torch.hub.load = lambda _repo, name, **_kw: getattr(backbones, name)(pretrained=False)  # type: ignore[assignment]
    import denoiser  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415

    model, vit = VARIANTS[ref][:2]
    args = SimpleNamespace(
        model=model, img_size=512, attn_dropout=0.0, proj_dropout=0.0, attention_type="torch",
        feature_embedding_type=f"dinov3_{vit}", dinov3_use_intermediate_layers=True,
        dinov3_num_intermediate_layers=4, feature_embedding_lr_scale=0.0, P_mean=-0.8, P_std=0.8, t_eps=5e-2,
        noise_scale=1.0, ema_decay1=0.9999, ema_decay2=0.9996, num_sampling_steps=STEPS,
        generate_noise_scale=0.0, sample_t_eps=0.0,
    )  # fmt: skip
    den = denoiser.Denoiser(args)
    missing, unexpected = den.net.y_embedder.load_state_dict(hub_state(load_file(dino_file)), strict=False)
    missing = [k for k in missing if not k.endswith(("rope_embed.periods", "bias_mask"))]
    if missing or unexpected:
        raise RuntimeError((missing[:5], unexpected[:5]))
    sd = torch.load(ckpt_file, map_location="cpu", weights_only=True)["model_ema1"]
    missing, unexpected = den.net.load_state_dict({k.removeprefix("net."): v for k, v in sd.items()}, strict=False)
    missing = [k for k in missing if not k.startswith("y_embedder.")]
    if missing or unexpected:
        raise RuntimeError((missing[:5], unexpected[:5]))
    return den.to(device).eval()


def predict(den: Any, rgb: torch.Tensor) -> torch.Tensor:  # noqa: ANN401
    """Upstream's wild-image path: `[B, 3, H, W]` float in [0, 1] -> points `[B, 3, H, W]` float32."""
    h, w = rgb.shape[-2:]
    nh, nw = wild_size(h, w)
    x = rgb * 2 - 1  # upstream loaders: [-1, 1]
    if (nh, nw) != (h, w):
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    pts = den.generate(x).float()
    return pts if (nh, nw) == (h, w) else F.interpolate(pts, size=(h, w), mode="nearest")


def main() -> None:
    repo, dinov3, weights = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rgb = image()
    h, w = rgb.shape[:2]
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255
    nh, nw = wild_size(h, w)
    for ref in sys.argv[4:] or VARIANTS:
        _, vit, ckpt = VARIANTS[ref]
        den = reference(
            repo, dinov3, ref, weights / f"dinov3-{vit}-pretrain-lvd1689m" / "model.safetensors", weights / ckpt
        )
        with torch.inference_mode():
            pts = predict(den, x)[0].permute(1, 2, 0).numpy()
        write(
            f"{ref}@1",
            {"points": pts, "depth": pts[..., 2]},
            family="pointdit",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=(
                f"CPU float32, wild-image resize to ~{TOKENS} tokens ({nh}x{nw}), {STEPS} Euler steps, DINOv3 hub"
                f" @{DINOV3_COMMIT} from the HF checkpoint, torch {torch.__version__}"
            ),
            task="depth",
        )
        print(f"  median |z| {np.median(np.abs(pts[..., 2])):.4f} net {nh}x{nw}")
        del den


if __name__ == "__main__":
    main()
