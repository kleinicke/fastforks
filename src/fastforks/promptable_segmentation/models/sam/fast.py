"""SAM input handling and the portable image-encoder forward (`segment_anything/fast.py`, "friendly" path).

- input: upstream `ResizeLongestSide` (PIL bilinear on uint8, longest side `image_size`), normalise, zero-pad
  bottom-right to a square. On the CPU PIL itself resizes (bit-exact); on CUDA `pil_bilinear` (separable antialiased
  passes with uint8 rounding in between, like PIL; <= 0.1 % of pixels differ by 1 LSB).
- encoder: same maths as upstream `ImageEncoderViT.forward`; the attention with the decomposed relative-position bias
  `rel_h[q, kh] + rel_w[q, kw]` runs through SDPA by folding the bias into an augmented dot product
  (`q' = [q * scale, rel_h, rel_w]`, `k' = [k, onehot(kh), onehot(kw)]`, padded to a multiple of 8; `v` zero-padded to
  the same head dim so the flash kernel takes it), never materialising the N x N attention matrix. The neck's
  `LayerNorm2d`s run channels-last through `F.layer_norm` (float32 statistics for float16 activations).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from fastforks.promptable_segmentation._runner import scale

from .net.image_encoder import get_rel_pos

if TYPE_CHECKING:
    from fastforks._core.cards import Io


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


def window_partition(x: torch.Tensor, ws: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """`[B, H, W, C]` -> zero-padded `[B * windows, ws, ws, C]` and the padded size (upstream `window_partition`)."""
    b, h, w, c = x.shape
    ph, pw = (ws - h % ws) % ws, (ws - w % ws) % ws
    if ph or pw:
        x = F.pad(x, (0, 0, 0, pw, 0, ph))
    hp, wp = h + ph, w + pw
    x = x.view(b, hp // ws, ws, wp // ws, ws, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, c)
    return x, (hp, wp)


def window_unpartition(x: torch.Tensor, ws: int, pad_hw: tuple[int, int], hw: tuple[int, int]) -> torch.Tensor:
    """Inverse of `window_partition` (upstream `window_unpartition`)."""
    (hp, wp), (h, w) = pad_hw, hw
    b = x.shape[0] // (hp * wp // ws // ws)
    x = x.view(b, hp // ws, wp // ws, ws, ws, -1).permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, -1)
    return x[:, :h, :w] if (hp > h or wp > w) else x


def onehot_keys(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """`[h * w, h + w]`: row `(kh, kw)` has a 1 in column `kh` and in column `h + kw`."""
    eye_h = torch.eye(h, device=device, dtype=dtype)
    eye_w = torch.eye(w, device=device, dtype=dtype)
    return torch.cat([eye_h.repeat_interleave(w, 0), eye_w.repeat(h, 1)], 1)


def augmented(
    qkv: torch.Tensor, nh: int, rel: tuple[torch.Tensor, torch.Tensor, torch.Tensor], hw: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Packed qkv `[B, N, 3C]` -> SDPA operands `q', k', v'` `[B, heads, N, Dp]` (to run with `scale=1`), head dim.

    `rel`: the relative-position tables `Rh [h, h, d]`, `Rw [w, w, d]` and `onehot_keys(h, w)`.
    """
    rh, rw, onehot = rel
    b, n, c3 = qkv.shape
    d = c3 // 3 // nh
    h, w = hw
    q, k, v = qkv.view(b, n, 3, nh, d).unbind(2)
    r_q = q.reshape(b, h, w, nh, d)
    rel_h = torch.einsum("bhwnc,hkc->bnhwk", r_q, rh).reshape(b, nh, n, h)
    rel_w = torch.einsum("bhwnc,wkc->bnhwk", r_q, rw).reshape(b, nh, n, w)
    pad = -(d + h + w) % 8
    qt = q.transpose(1, 2)
    zeros = qt.new_zeros(b, nh, n, pad)
    qa = torch.cat([qt * d**-0.5, rel_h, rel_w, zeros], -1)
    ka = torch.cat([k.transpose(1, 2), onehot.expand(b, nh, n, h + w), zeros], -1)
    va = torch.cat([v.transpose(1, 2), qt.new_zeros(b, nh, n, h + w + pad)], -1)
    return qa, ka, va, d


Rel = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def rel_tables(blk: Any, tokens: int, dtype: torch.dtype) -> Rel:  # noqa: ANN401 - upstream Block
    """The block's relative-position tables `Rh`, `Rw` and `onehot_keys` (window, or `tokens x tokens` if global)."""
    a, side = blk.attn, int(blk.window_size) or tokens
    with torch.no_grad():
        rh = get_rel_pos(side, side, a.rel_pos_h).to(dtype)
        rw = get_rel_pos(side, side, a.rel_pos_w).to(dtype)
    return rh, rw, onehot_keys(side, side, a.rel_pos_h.device, dtype)


def attention(blk: Any, x: torch.Tensor, rel: Rel) -> torch.Tensor:  # noqa: ANN401 - upstream Block
    """Upstream `Attention.forward` (relative positions) through SDPA: `[B, H, W, C]` -> `[B, H, W, C]`."""
    a = blk.attn
    b, h, w, c = x.shape
    qkv = a.qkv(x.reshape(b, h * w, c))
    qa, ka, va, d = augmented(qkv, a.num_heads, rel, (h, w))
    o = F.scaled_dot_product_attention(qa, ka, va, scale=1.0)[..., :d]
    return a.proj(o.transpose(1, 2).reshape(b, h, w, c))


def encode(enc: Any, x: torch.Tensor, rel: list[Rel]) -> torch.Tensor:  # noqa: ANN401 - upstream ImageEncoderViT
    """Upstream `ImageEncoderViT.forward` in the encoder's dtype: `[B, 3, S, S]` -> float32 `[B, 256, S/16, S/16]`.

    `rel`: per block, `rel_tables` for its window (the input is always the padded square, so they are fixed).
    """
    x = enc.patch_embed(x.to(enc.pos_embed.dtype))
    x = x + enc.pos_embed
    for blk, r in zip(enc.blocks, rel, strict=True):
        shortcut = x
        y = blk.norm1(x)
        if blk.window_size > 0:
            hw = (y.shape[1], y.shape[2])
            y, pad_hw = window_partition(y, blk.window_size)
            y = window_unpartition(attention(blk, y, r), blk.window_size, pad_hw, hw)
        else:
            y = attention(blk, y, r)
        x = shortcut + y
        x = x + blk.mlp(blk.norm2(x))
    n = enc.neck  # conv 1x1, LayerNorm2d, conv 3x3, LayerNorm2d
    y = F.linear(x, n[0].weight.flatten(1))
    y = F.layer_norm(y, (y.shape[-1],), n[1].weight, n[1].bias, n[1].eps)
    y = n[2](y.permute(0, 3, 1, 2))
    y = F.layer_norm(y.permute(0, 2, 3, 1), (y.shape[1],), n[3].weight, n[3].bias, n[3].eps)
    return y.float().permute(0, 3, 1, 2).contiguous()
