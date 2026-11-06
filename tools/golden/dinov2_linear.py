"""DINOv2 linear (ADE20K) goldens from the reference implementation.

Reference: upstream facebookresearch/dinov2 at `REF_COMMIT` (unpatched): the backbone from `dinov2.hub.backbones`
(`dinov2_vit{s,b,l}14(pretrained=False)` + the variant's `_pretrain.pth`), CPU float32. The head and the test-time
protocol of `dinov2_vit*14_ade20k_linear_config.py` (mmseg 0.x, as the repository's segmentation notebook builds the
segmenter) are written out here independently of `fastforks.shared`:

- mmcv keep-ratio `Resize(img_scale=(99999999, 512))` (bilinear, half-pixel centres), ImageNet normalisation.
- mmseg `slide_inference`, crop 512, stride 341; per crop the notebook's `CenterPadding(14)` forward pre-hook of the
  backbone, `get_intermediate_layers(n=out_indices, reshape=True)` (norm on), `BNHead` (`in_index=[3]`,
  `resize_concat`: the last block only; `BatchNorm` eps 1e-5, then `conv_seg` 1x1), logits resized to the crop.
- the averaged logits resized to the image size (`rescale=True`).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/dinov2_linear.py DINOV2_CHECKOUT [vits vitb vitl]

Weight files are the card's (hash-checked, `FASTFORKS_WEIGHTS_PATH` or the authors' URLs).
"""

from __future__ import annotations

import functools
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from _common import image, reference_tree, write

from fastforks._core import registry
from fastforks._core.preprocess import IMAGENET_MEAN, IMAGENET_STD

UPSTREAM = "https://github.com/facebookresearch/dinov2"
REF_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
FAMILY = "dinov2_linear"
TASK = "segmentation"
SHORT, CROP, STRIDE = 512, 512, 341
OUT_INDICES = {"vits": [8, 9, 10, 11], "vitb": [8, 9, 10, 11], "vitl": [20, 21, 22, 23]}


def _center_pad(x: torch.Tensor, multiple: int) -> torch.Tensor:
    pads = []
    for size in x.shape[:1:-1]:  # last dim first, as F.pad wants
        new = math.ceil(size / multiple) * multiple
        pads += [(new - size) // 2, new - size - (new - size) // 2]
    return F.pad(x, pads)


def _encode_decode(
    vit: Any,  # noqa: ANN401 - the upstream DinoVisionTransformer
    head: dict,
    crop: torch.Tensor,
    arch: str,
) -> torch.Tensor:
    feats = vit.get_intermediate_layers(_center_pad(crop, 14), n=OUT_INDICES[arch], reshape=True)
    f = F.batch_norm(feats[3], head["running_mean"], head["running_var"], head["weight"], head["bias"], eps=1e-5)
    out = F.conv2d(f, head["conv_w"], head["conv_b"])
    return F.interpolate(out, crop.shape[2:], mode="bilinear", align_corners=False)


def _slide(
    vit: Any,  # noqa: ANN401 - the upstream DinoVisionTransformer
    head: dict,
    img: torch.Tensor,
    arch: str,
) -> torch.Tensor:
    _, _, h_img, w_img = img.shape
    h_grids = max(h_img - CROP + STRIDE - 1, 0) // STRIDE + 1
    w_grids = max(w_img - CROP + STRIDE - 1, 0) // STRIDE + 1
    preds = torch.zeros((1, 150, h_img, w_img), device=img.device)
    count = torch.zeros((1, 1, h_img, w_img), device=img.device)
    for hi in range(h_grids):
        for wi in range(w_grids):
            y1, x1 = hi * STRIDE, wi * STRIDE
            y2, x2 = min(y1 + CROP, h_img), min(x1 + CROP, w_img)
            y1, x1 = max(y2 - CROP, 0), max(x2 - CROP, 0)
            preds[:, :, y1:y2, x1:x2] += _encode_decode(vit, head, img[:, :, y1:y2, x1:x2], arch)
            count[:, :, y1:y2, x1:x2] += 1
    return preds / count


def whole(vit: Any, head: dict, x: torch.Tensor, arch: str) -> torch.Tensor:  # noqa: ANN401 - upstream ViT
    """The reference logits `[1, 150, H, W]` of an image `[1, 3, H, W]` in `[0, 1]` (the authors' test protocol)."""
    h, w = x.shape[-2:]
    s = SHORT / min(h, w)  # mmcv rescale: min(99999999 / long, 512 / short)
    y = F.interpolate(x, (int(h * s + 0.5), int(w * s + 0.5)), mode="bilinear", align_corners=False)
    mean, std = (torch.tensor(v, device=x.device)[:, None, None] for v in (IMAGENET_MEAN, IMAGENET_STD))
    return F.interpolate(_slide(vit, head, (y - mean) / std, arch), (h, w), mode="bilinear", align_corners=False)


@functools.cache
def _on_path(checkout: Path) -> None:
    sys.path.insert(0, str(reference_tree(checkout, REF_COMMIT)))


def reference(checkout: Path, card: Any, variant: Any) -> tuple[Any, dict, str]:  # noqa: ANN401 - registry records
    """The upstream ViT (CPU float32, eval) and the head tensors of a variant, and its arch."""
    _on_path(checkout)
    from dinov2.hub import backbones  # noqa: PLC0415  # reference checkout on sys.path

    arch = variant.params["arch"]
    backbone, head_file = registry.weights(card, variant)
    vit = getattr(backbones, f"dinov2_{arch}14")(pretrained=False)
    vit.load_state_dict(torch.load(backbone, map_location="cpu", weights_only=True))
    sd = torch.load(head_file, map_location="cpu", weights_only=True)["state_dict"]
    head = {k: sd[f"decode_head.bn.{k}"] for k in ("weight", "bias", "running_mean", "running_var")}
    head |= {"conv_w": sd["decode_head.conv_seg.weight"], "conv_b": sd["decode_head.conv_seg.bias"]}
    return vit.eval(), head, arch


def main() -> None:
    archs = set(sys.argv[2:])
    x = torch.from_numpy(image()).permute(2, 0, 1)[None].float() / 255
    for card, variant in registry.cards(TASK).values():
        if card.family != FAMILY or (archs and variant.params["arch"] not in archs):
            continue
        vit, head, arch = reference(Path(sys.argv[1]), card, variant)
        with torch.inference_mode():
            logits = whole(vit, head, x, arch)
        write(
            variant.ref,
            {"logits": logits[0].numpy()},
            family=FAMILY,
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, torch {torch.__version__}",
            task=TASK,
        )
        print(f"  {variant.ref}: classes {sorted(set(logits[0].argmax(0).ravel().tolist()))[:8]}")


if __name__ == "__main__":
    main()
