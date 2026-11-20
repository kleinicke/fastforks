"""Shared parts of the Swin super-resolution networks (SwinIR, Swin2SR, HAT, DAT).

`DropPath`, `to_2tuple`, `trunc_normal_` replace the timm / BasicSR helpers the vendored networks import (inference
only). `shift_mask` is SwinIR's `SwinTransformerBlock.calculate_mask` (the 0 / -100 shifted-window mask), made on the
device once per size. `relative_bias` gathers a block's relative position bias once, `window_attention` is the window
attention as one `scaled_dot_product_attention` call with that bias (+ the mask) as its additive mask. `pad_heads`
zero-pads a block's head dimension to a multiple of 8 in its `qkv` / `proj` weights at load: SwinIR's 30 (180 / 6,
240 / 8) keeps the fused attention kernels out and SDPA falls back to its math path, which materialises the scores
(2.1 GB and 28 ms per block of a 520 x 520 image, SwinIR-M; padded: 0.6 GB, 5 ms). The scores are unchanged (zero
rows of q and k), the extra output channels meet zero columns of `proj`. `attention` is the SDPA call itself, padding
the head size at run time where the weights are not padded (HAT, DAT) and copying strided inputs. `patch_embed` is
`PatchEmbed.forward` with contiguous tokens. `checkpoint_upsampler` reads a checkpoint's reconstruction layers
(`load(path)`).
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "DropPath",
    "attention",
    "checkpoint_upsampler",
    "pad_heads",
    "patch_embed",
    "relative_bias",
    "shift_mask",
    "to_2tuple",
    "trunc_normal_",
    "window_attention",
]

trunc_normal_ = nn.init.trunc_normal_


def to_2tuple(x: int | tuple[int, int]) -> tuple[int, int]:
    """`to_2tuple` of timm, for ints and pairs."""
    return x if isinstance(x, tuple) else (x, x)


def DropPath(drop_prob: float) -> nn.Module:  # noqa: N802 - the timm class name the vendored nets call
    """`DropPath` of timm at inference: the identity (stochastic depth only acts in training)."""
    del drop_prob
    return nn.Identity()


def shift_mask(  # noqa: PLR0913 - window geometry
    h: int, w: int, window: int, shift: int, device: torch.device, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """The cached `_shift_mask` (dynamo cannot trace through lru_cache; the compiled graph holds the mask anyway)."""
    fn = _shift_mask.__wrapped__ if torch.compiler.is_compiling() else _shift_mask
    return fn(h, w, window, shift, device, dtype=dtype)


@functools.lru_cache(maxsize=16)
def _shift_mask(  # noqa: PLR0913 - the map size, window, shift and where / how the mask is cached
    h: int, w: int, window: int, shift: int, device: torch.device, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """`[nW, window**2, window**2]` 0 / -100 mask of the shifted windows of an `h` x `w` map (SwinIR's), in `dtype`.

    Cached in the attention's dtype: float16 holds 0 / -100 exactly, and a float32 copy per size plus a cast per call
    cost 1.5 x its float16 size (64 + 32 MB at 512 x 512, window 8).
    """
    img = torch.zeros(1, h, w, 1, device=device)
    cnt = 0
    for hs in (slice(0, -window), slice(-window, -shift), slice(-shift, None)):
        for ws in (slice(0, -window), slice(-window, -shift), slice(-shift, None)):
            img[:, hs, ws, :] = cnt
            cnt += 1
    m = img.view(1, h // window, window, w // window, window, 1).permute(0, 1, 3, 2, 4, 5).reshape(-1, window * window)
    m = m.unsqueeze(1) - m.unsqueeze(2)
    return m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0).to(dtype)


@functools.lru_cache(maxsize=16)
def shifted_window_regions(h: int, w: int, window: int, shift: int, device: torch.device) -> torch.Tensor:
    """One region label per window token, in the order returned by window partition."""
    y, x = (torch.arange(n, device=device) for n in (h, w))
    rows = (y >= h - window).to(torch.uint8) + (y >= h - shift)
    cols = (x >= w - window).to(torch.uint8) + (x >= w - shift)
    regions = 3 * rows[:, None] + cols[None, :]
    return regions.view(h // window, window, w // window, window).permute(0, 2, 1, 3).reshape(-1, window**2)


def relative_bias(table: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """`[nH, N, N]` relative position bias of one block (`table[index]`, as the attention gathers it per call)."""
    n = index.shape[0]
    return table[index.reshape(-1)].view(n, n, -1).permute(2, 0, 1).contiguous()


def pad_heads(qkv: nn.Linear, proj: nn.Linear, heads: int, multiple: int = 8) -> None:
    """Zero-pad the per-head dimension of `qkv` (rows) and `proj` (columns) to a multiple of `multiple`, in place."""
    d = qkv.out_features // (3 * heads)
    dp = -(-d // multiple) * multiple
    if dp == d:
        return
    with torch.no_grad():
        w = qkv.weight.new_zeros(3, heads, dp, qkv.in_features)
        w[:, :, :d] = qkv.weight.view(3, heads, d, qkv.in_features)
        qkv.weight = nn.Parameter(w.view(3 * heads * dp, qkv.in_features), requires_grad=False)
        if qkv.bias is not None:
            b = qkv.bias.new_zeros(3, heads, dp)
            b[:, :, :d] = qkv.bias.view(3, heads, d)
            qkv.bias = nn.Parameter(b.view(-1), requires_grad=False)
        qkv.out_features = 3 * heads * dp
        wp = proj.weight.new_zeros(proj.out_features, heads, dp)
        wp[:, :, :d] = proj.weight.view(proj.out_features, heads, d)
        proj.weight = nn.Parameter(wp.view(proj.out_features, heads * dp), requires_grad=False)
        proj.in_features = heads * dp


def _border_attention(  # noqa: PLR0913 - QKV, bias, region labels and SDPA scale
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, regions: torch.Tensor, *, scale: float
) -> torch.Tensor:
    """Apply full additive masks only to windows containing multiple shifted regions."""
    n = q.shape[0]
    out = attention(q, k, v, bias.expand(n, -1, -1, -1), scale)
    border = torch.where(regions.ne(regions[:, :1]).any(dim=-1))[0]
    windows = regions.shape[0]
    if border.numel():
        labels = regions.index_select(0, border)
        shift_mask = labels[:, :, None].ne(labels[:, None, :]).to(q.dtype) * -100.0
        weights = bias + shift_mask[:, None]
        batch = n // windows
        if batch > 1:
            offsets = torch.arange(batch, device=q.device) * windows
            border = (offsets[:, None] + border[None, :]).flatten()
            weights = weights.repeat(batch, 1, 1, 1)
        selected = attention(*(tensor.index_select(0, border) for tensor in (q, k, v)), weights, scale)
        out.index_copy_(0, border, selected)
    return out


def window_attention(  # noqa: PLR0913 - the block's projections, head count, scale, bias and mask
    x: torch.Tensor,
    qkv: nn.Linear,
    proj: nn.Linear,
    *,
    heads: int,
    scale: float,
    bias: torch.Tensor,
    mask: torch.Tensor | None,
    cosine: torch.Tensor | None = None,
) -> torch.Tensor:
    """Swin window attention of `[B * nW, N, C]` windows: `proj(softmax(q k^T * scale + bias (+ mask)) v)`.

    `bias` is `relative_bias(...)` `[nH, N, N]`. `mask` is an additive shifted-window mask `[nW, N, N]`, compact
    region labels `[nW, N]`, or None. The head dimension is `qkv`'s, optionally padded by `pad_heads`. For Swin V2,
    `cosine` is the clamped, exponentiated logit scale `[nH, 1, 1]`. It multiplies q after L2-normalisation of q
    and k (pass `scale=1`).
    """
    b_, n, _ = x.shape
    d = qkv.out_features // (3 * heads)
    q, k, v = qkv(x).reshape(b_, n, 3, heads, d).permute(2, 0, 3, 1, 4).unbind(0)
    if cosine is not None:  # zero padded head columns leave the norms unchanged
        q = F.normalize(q, dim=-1) * cosine.to(q.dtype)
        k = F.normalize(k, dim=-1)
    add = bias.to(q.dtype)[None]
    if mask is not None and mask.ndim == 2:  # noqa: PLR2004 - compact labels are [windows, tokens]
        out = _border_attention(q, k, v, add, mask, scale=scale)
        del q, k, v
    else:
        if mask is not None:
            nw = mask.shape[0]
            add = add + mask.to(q.dtype)[:, None]
            if b_ != nw:
                add = add.repeat(b_ // nw, 1, 1, 1)
        out = attention(q, k, v, add.expand(b_, -1, -1, -1), scale)
    return proj(out.transpose(1, 2).reshape(b_, n, heads * d))


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, scale: float) -> torch.Tensor:
    """`scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)`, head size padded to a multiple of 8 (CUDA).

    Zero q / k columns add nothing to the scores, the zero v columns are cut off. Inputs whose head dimension is
    strided are copied: the fused kernels need it contiguous, else SDPA falls back to the unfused math path (HAT-S
    overlapping attention, head size 24: 9.9 GB and 1.7 s instead of 2.0 GB and 1.1 s at 512 x 512).
    """
    d = q.shape[-1]
    pad = -d % 8 if q.is_cuda else 0
    if pad:
        q, k, v = (F.pad(t, (0, pad)) for t in (q, k, v))
    else:
        q, k, v = (t if t.stride(-1) == 1 else t.contiguous() for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    return out[..., :d] if pad else out


def patch_embed(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """`PatchEmbed.forward` (`[B, C, H, W]` -> `[B, H * W, C]` tokens, + its norm) with contiguous tokens.

    Upstream returns a transposed view; each group adds its input to it (`patch_embed(...) + x`), and the sum takes
    the first operand's strides, so the whole residual stream ran transposed and every residual add was a strided
    kernel (HAT 512 x 512: 0.7 of 2.3 s).
    """
    x = x.flatten(2).transpose(1, 2).contiguous()
    return x if m.norm is None else m.norm(x)  # ty: ignore[call-non-callable]


def checkpoint_upsampler(sd: Mapping[str, torch.Tensor]) -> tuple[str, int] | None:
    """`(upsampler, scale)` of a checkpoint from its reconstruction layers, or None (denoising: not run here)."""
    if "conv_up1.weight" in sd:
        return "nearest+conv", 4 if "conv_up2.weight" in sd else 2
    convs = [v for k, v in sd.items() if k.startswith("upsample.") and k.endswith(".weight")]
    if not convs:
        return None
    if "conv_before_upsample.0.weight" not in sd:  # one conv to scale² x 3 channels
        return "pixelshuffledirect", round(math.sqrt(convs[0].shape[0] / 3))
    ratio = convs[0].shape[0] // sd["conv_before_upsample.0.weight"].shape[0]
    return "pixelshuffle", 3 if ratio == 9 else 2 ** len(convs)  # noqa: PLR2004 - x3: one conv to 9 x channels
