"""Segment Anything's prompt side, for families that put another image encoder under it.

- `net/`: upstream prompt encoder, two-way transformer and mask decoder (unchanged).
- input: upstream `ResizeLongestSide` (PIL bilinear on uint8, longest side `long_side`), normalise, zero-pad
  bottom-right to a square. On the CPU PIL itself resizes (bit-exact); elsewhere `pil_bilinear` (separable antialiased
  passes with uint8 rounding in between, like PIL).
- `decode`: upstream `PromptEncoder.forward` + `MaskDecoder.predict_masks` without host read-backs, so a CUDA graph
  can capture it (`fastforks.shared.cuda_graphs`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from fastforks._core.preprocess import scale

if TYPE_CHECKING:
    from fastforks._core.cards import Io

    from .net.mask_decoder import MaskDecoder
    from .net.prompt_encoder import PromptEncoder


def preprocess_shape(h: int, w: int, long_side: int) -> tuple[int, int]:
    """Upstream `ResizeLongestSide.get_preprocess_shape`: `(h, w)` scaled so the longer side is `long_side`."""
    s = long_side / max(h, w)
    return int(h * s + 0.5), int(w * s + 0.5)


def pil_bilinear(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """PIL `BILINEAR` resize of a float `[B, C, H, W]` 0..255 tensor (horizontal pass, round, vertical pass, round)."""
    (h, w), (nh, nw) = x.shape[-2:], size
    if nw != w:
        x = F.interpolate(x, (h, nw), mode="bilinear", align_corners=False, antialias=True).round_().clamp_(0, 255)
    if nh != h:
        x = F.interpolate(x, (nh, nw), mode="bilinear", align_corners=False, antialias=True).round_().clamp_(0, 255)
    return x


def _pil_resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """PIL itself (CPU, bit-exact with upstream), on a float `[1, 3, H, W]` 0..255 tensor."""
    arr = x[0].permute(1, 2, 0).to(torch.uint8).numpy()
    out = np.asarray(Image.fromarray(arr).resize(size[::-1], Image.Resampling.BILINEAR))
    return torch.from_numpy(out.copy()).permute(2, 0, 1)[None].float()


def prepare(image: torch.Tensor, io: Io, long_side: int, *, exact: bool) -> torch.Tensor:
    """`[1, 3, H, W]` RGB in `[0, 1]` -> the normalised, zero-padded `[1, 3, long_side, long_side]` float32 input.

    Upstream takes uint8 images, so values are rounded to 0..255 first. `exact`: resize with PIL (CPU tensors only).
    """
    size = preprocess_shape(*image.shape[-2:], long_side)
    x = image.mul(255).round_()
    if size != tuple(image.shape[-2:]):
        x = _pil_resize(x, size) if exact else pil_bilinear(x, size)
    x = scale(x.div_(255), io)
    return F.pad(x, (0, long_side - size[1], 0, long_side - size[0]))


def embed_points(pe: PromptEncoder, points: torch.Tensor, labels: torch.Tensor, *, pad: bool) -> torch.Tensor:
    """Upstream `PromptEncoder._embed_points` with `torch.where` for its boolean-mask writes (same values).

    A masked write reads the mask back on the host, which a CUDA graph capture refuses.
    """
    points = points + 0.5  # shift to the pixel centre
    if pad:
        points = torch.cat([points, points.new_zeros((len(points), 1, 2))], dim=1)
        labels = torch.cat([labels, labels.new_full((len(labels), 1), -1)], dim=1)
    emb = pe.pe_layer.forward_with_coords(points, pe.input_image_size)
    lab = labels[..., None]
    zero = emb.new_zeros(())
    kinds = [pe.not_a_point_embed, *pe.point_embeddings[:2]]  # labels -1, 0, 1
    emb = torch.where(lab == -1, zero, emb)
    for label, kind in zip((-1, 0, 1), kinds, strict=True):
        emb = emb + torch.where(lab == label, cast("torch.nn.Embedding", kind).weight, zero)
    return emb


def decode(  # noqa: PLR0913 - one argument per prompt kind
    pe: PromptEncoder,
    decoder: MaskDecoder,
    embedding: torch.Tensor,
    *,
    points: torch.Tensor | None,
    labels: torch.Tensor | None,
    boxes: torch.Tensor | None,
    masks: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream `PromptEncoder.forward` + `MaskDecoder.predict_masks`: all four masks `[P, 4, h, w]` and scores.

    Prompts in the 1024 frame. Index 0 is the single-mask output, 1-3 the multimask output.
    """
    parts = []
    if points is not None and labels is not None:
        parts.append(embed_points(pe, points, labels, pad=boxes is None))
    if boxes is not None:
        parts.append(pe._embed_boxes(boxes))  # noqa: SLF001 - upstream's own forward calls it
    n = len(parts[0]) if parts else 1 if masks is None else len(masks)
    sparse = torch.cat(parts, dim=1) if parts else embedding.new_empty((n, 0, pe.embed_dim))
    if masks is None:
        dense = pe.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(n, -1, *pe.image_embedding_size)
    else:
        dense = pe._embed_masks(masks)  # noqa: SLF001 - upstream's own forward calls it
    return decoder.predict_masks(
        image_embeddings=embedding,
        image_pe=pe.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
    )
