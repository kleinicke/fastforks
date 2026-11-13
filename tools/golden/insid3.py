"""INSID3 goldens from the reference implementation.

Reference: upstream visinf/INSID3 (`REF_COMMIT`), CPU float32, `INSID3.predict_mask` (bilinear refiner, no CRF) with
the DINOv3 ViT-L/16 encoder as transformers `DINOv3ViTModel` (the checkpoint fastforks loads; upstream's torch.hub
`get_intermediate_layers(x, n=1, reshape=True)` is its last-layer patch tokens, shimmed here). Inputs: the repo's cat
example (`assets/ref_cat_image.jpg` + `ref_cat_mask.png` -> `target_cat_image.jpg`), shrunk by 4 (PIL bilinear /
nearest for the mask) and stored in the golden. Outputs: `masks` at the target size and `similarity` (the debiased
patch similarity to the reference prototype, upstream `fg_sim`).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/insid3.py INSID3_CHECKOUT WEIGHTS_DIR

`WEIGHTS_DIR` holds `dinov3-vitl16-pretrain-lvd1689m/` (config.json, model.safetensors). Needs transformers 5.19,
einops, scikit-learn and pycocotools (upstream imports).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from _common import reference_tree, write
from PIL import Image
from torch import nn

UPSTREAM = "https://github.com/visinf/INSID3"
REF_COMMIT = "0c165a10cf52ab91f335883d06260de86854adbe"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
SHRINK = 4


class HubShim(nn.Module):
    """`get_intermediate_layers(x, n=1, reshape=True)` of the torch.hub DINOv3 on transformers' model."""

    def __init__(self, net: nn.Module) -> None:
        """Wrap `net`."""
        super().__init__()
        self.net = net

    def get_intermediate_layers(self, x: torch.Tensor, n: int = 1, reshape: bool = True) -> list[torch.Tensor]:  # noqa: FBT001, FBT002
        """Last-layer patch tokens `[B, C, h, w]` (the only call upstream makes)."""
        assert n == 1  # noqa: S101
        assert reshape  # noqa: S101
        t = self.net(pixel_values=x).last_hidden_state[:, 5:]  # cls + 4 registers first
        return [t.transpose(1, 2).reshape(x.shape[0], -1, x.shape[-2] // 16, x.shape[-1] // 16)]


def _load(path: Path, mode: str) -> np.ndarray:
    im = Image.open(path).convert(mode)
    im = im.resize((im.width // SHRINK, im.height // SHRINK), Image.BILINEAR if mode == "RGB" else Image.NEAREST)
    return np.asarray(im)


def _tensor(img: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255
    return (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    tree = reference_tree(repo, REF_COMMIT)
    sys.path.insert(0, str(tree))
    os.environ["HF_HUB_OFFLINE"] = "1"
    from models.insid3 import INSID3  # noqa: PLC0415 - reference checkout
    from transformers import DINOv3ViTModel  # noqa: PLC0415 - reference environment

    net = DINOv3ViTModel.from_pretrained(
        weights / "dinov3-vitl16-pretrain-lvd1689m", attn_implementation="eager", dtype=torch.float32
    ).eval()
    model = INSID3(HubShim(net), device="cpu").eval()
    ref, tgt = _load(tree / "assets/ref_cat_image.jpg", "RGB"), _load(tree / "assets/target_cat_image.jpg", "RGB")
    mask = _load(tree / "assets/ref_cat_mask.png", "L") > 0
    fg_sim = {}
    original = model._locate_candidates  # noqa: SLF001

    def locate(sim_maps, ref_masks, feat_tgt_deb, ref_prototype, h, w):  # noqa: ANN001, ANN202, PLR0913, PLR0917
        fg_sim["v"] = torch.einsum("bchw,cd->bhw", feat_tgt_deb, ref_prototype)[0]
        return original(sim_maps, ref_masks, feat_tgt_deb, ref_prototype, h, w)

    model._locate_candidates = locate  # noqa: SLF001
    model.set_reference(_tensor(ref), torch.from_numpy(mask))
    model.set_target(_tensor(tgt))
    with torch.inference_mode():
        out = model.segment()
    write(
        "insid3-vitl16@1",
        {
            "masks": out.numpy().astype(np.float32),
            "similarity": fg_sim["v"].numpy(),
            "input": tgt.astype(np.float32),
            "reference": ref.astype(np.float32),
            "reference_mask": mask.astype(np.float32),
        },
        family="insid3",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=None,
        settings=(
            f"CPU float32, INSID3.predict_mask (bilinear) on the repo's cat example shrunk by {SHRINK}, transformers "
            f"DINOv3ViTModel eager, torch {torch.__version__}"
        ),
        task="promptable_segmentation",
    )
    print(f"  mask {out.float().mean():.4f} of the target, similarity max {fg_sim['v'].max():.4f}")


if __name__ == "__main__":
    main()
