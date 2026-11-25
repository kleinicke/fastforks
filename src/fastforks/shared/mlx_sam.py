"""SAM's image encoder (ViTDet backbone with decomposed relative positions) written in MLX, channels last.

Shared by SAM and SAM-HQ (`MlxSamEncoder(enc, rel, dtype, hq=...)`). Written out instead of replayed from an
exported graph: the export, translation and segment tuning were most of the first call (2.3 s on an M1 Pro).

- Attention is MLX's fused `scaled_dot_product_attention` at the head dim (64 / 80) with upstream's bias
  `rel_h[q, kh] + rel_w[q, kw]` as a dense additive mask in the activation dtype (as upstream adds it). A global block
  (64 x 64 tokens) builds that mask in query-row chunks of at most `MASK_BYTES`, one after the other (`mx.depends`),
  so its N x N matrix is never resident at once.
- Windowed blocks partition the normalised map, zero-padded to whole 14 x 14 windows (the padded tokens are keys,
  as upstream's).
- LayerNorm is `mx.fast.layer_norm` (float32 statistics), the MLP's exact GELU float32; the neck's LayerNorm2d is the
  same LayerNorm over channels. The embedding (and SAM-HQ's feature) is returned in float32.
- SAM-HQ: the first global block's output and `MaskDecoderHQ`'s two 2 x 2 / stride-2 transposed-conv stacks
  (`embedding_encoder(embedding) + compress_vit_feat(early)`), in float32: each transposed conv is one matmul to the
  four output pixels per input pixel.
"""

# ruff: noqa: ANN401 - the upstream encoder's torch modules

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import torch

__all__ = ["MlxSamEncoder"]

MASK_BYTES = 64 << 20  # bound of one global block's mask chunk; the result does not depend on it
FUSED_DIMS = (64, 80, 128)  # head dims MLX's attention runs fused (SAM: 64 for B / L, 80 for H)


def _arr(t: torch.Tensor, dtype: mx.Dtype) -> mx.array:
    a = mx.array(t.detach().double().cpu().numpy().astype(np.float32)).astype(dtype)
    mx.eval(a)
    return a


def _linear(layer: Any, dtype: mx.Dtype) -> Callable[[mx.array], mx.array]:
    w = _arr(layer.weight.T, dtype)
    b = _arr(layer.bias, dtype) if layer.bias is not None else None
    return (lambda x: x @ w) if b is None else (lambda x: mx.addmm(b, x, w))


def _layer_norm(layer: Any, dtype: mx.Dtype) -> Callable[[mx.array], mx.array]:
    w, b, eps = _arr(layer.weight, dtype), _arr(layer.bias, dtype), float(layer.eps)
    return lambda x: mx.fast.layer_norm(x, w, b, eps)


def _gelu(x: mx.array) -> mx.array:
    """Exact (erf) GELU computed in float32."""
    y = x.astype(mx.float32)
    return (y * (0.5 + 0.5 * mx.erf(y * 0.7071067811865476))).astype(x.dtype)


def _rel(q: mx.array, rh: mx.array, rw: mx.array) -> tuple[mx.array, mx.array]:
    """Queries `[B, h, w, heads, d]` -> `rel_h [B, heads, h, w, kh]`, `rel_w [B, heads, h, w, kw]`."""
    b, h, w, n, d = q.shape
    qh = q.transpose(1, 0, 2, 3, 4).reshape(h, b * w * n, d)
    rel_h = (qh @ rh.transpose(0, 2, 1)).reshape(h, b, w, n, -1).transpose(1, 3, 0, 2, 4)
    qw = q.transpose(2, 0, 1, 3, 4).reshape(w, b * h * n, d)
    rel_w = (qw @ rw.transpose(0, 2, 1)).reshape(w, b, h, n, -1).transpose(1, 3, 2, 0, 4)
    return rel_h, rel_w


def _sdpa(q: mx.array, k: mx.array, v: mx.array, mask: mx.array, scale: float) -> mx.array:
    """Fused attention `[B, heads, L, d]`; an unfused head dim is zero-padded to the next fused one (same scores)."""
    d = q.shape[-1]
    if d in FUSED_DIMS or d > FUSED_DIMS[-1]:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    pad = [(0, 0)] * 3 + [(0, next(f for f in FUSED_DIMS if f > d) - d)]
    out = mx.fast.scaled_dot_product_attention(mx.pad(q, pad), mx.pad(k, pad), mx.pad(v, pad), scale=scale, mask=mask)
    return out[..., :d]


def _attention(attn: Any, rel: Sequence[torch.Tensor], dtype: mx.Dtype) -> Callable[[mx.array], mx.array]:
    """Upstream `Attention.forward` with relative positions: `[B, h, w, C]` -> `[B, h, w, C]`."""
    qkv, proj = _linear(attn.qkv, dtype), _linear(attn.proj, dtype)
    heads = attn.num_heads
    rh, rw = _arr(rel[0], dtype), _arr(rel[1], dtype)
    item = 2 if dtype == mx.float16 else 4

    def apply(x: mx.array) -> mx.array:
        b, h, w, c = x.shape
        d, n = c // heads, h * w
        t = qkv(x).reshape(b, h, w, 3, heads, d)
        q, k, v = t[..., 0, :, :], t[..., 1, :, :], t[..., 2, :, :]
        rel_h, rel_w = _rel(q, rh, rw)
        q, k, v = (a.reshape(b, n, heads, d).transpose(0, 2, 1, 3) for a in (q, k, v))
        rows = max(1, min(h, MASK_BYTES // (b * heads * w * n * item)))
        outs: list[mx.array] = []
        for r in range(0, h, rows):
            mask = rel_h[:, :, r : r + rows, :, :, None] + rel_w[:, :, r : r + rows, :, None, :]
            mask = mask.reshape(b, heads, -1, n)
            qi = q[:, :, r * w : (r + rows) * w]
            if outs:
                qi = mx.depends(qi, outs[-1])
            outs.append(_sdpa(qi, k, v, mask, d**-0.5))
        o = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)
        return proj(o.transpose(0, 2, 1, 3).reshape(b, h, w, c))

    return apply


def _windowed(attend: Callable[[mx.array], mx.array], ws: int) -> Callable[[mx.array], mx.array]:
    """Attention within zero-padded `ws x ws` windows (upstream `window_partition` / `window_unpartition`)."""

    def apply(x: mx.array) -> mx.array:
        b, h, w, c = x.shape
        ph, pw = -h % ws, -w % ws
        if ph or pw:
            x = mx.pad(x, [(0, 0), (0, ph), (0, pw), (0, 0)])
        nh, nw = (h + ph) // ws, (w + pw) // ws
        x = x.reshape(b, nh, ws, nw, ws, c).transpose(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, c)
        y = attend(x).reshape(b, nh, nw, ws, ws, c).transpose(0, 1, 3, 2, 4, 5).reshape(b, h + ph, w + pw, c)
        return y[:, :h, :w] if ph or pw else y

    return apply


def _block(blk: Any, rel: Sequence[torch.Tensor], dtype: mx.Dtype) -> Callable[[mx.array], mx.array]:
    norm1, norm2 = _layer_norm(blk.norm1, dtype), _layer_norm(blk.norm2, dtype)
    attend = _attention(blk.attn, rel, dtype)
    if blk.window_size > 0:
        attend = _windowed(attend, int(blk.window_size))
    lin1, lin2 = _linear(blk.mlp.lin1, dtype), _linear(blk.mlp.lin2, dtype)

    def apply(x: mx.array) -> mx.array:
        x = x + attend(norm1(x))
        return x + lin2(_gelu(lin1(norm2(x))))

    return apply


def _conv_t2(layer: Any) -> Callable[[mx.array], mx.array]:
    """`ConvTranspose2d(kernel 2, stride 2)` in float32: `[B, h, w, C]` -> `[B, 2h, 2w, O]`."""
    cin, o = layer.weight.shape[:2]
    w = _arr(layer.weight.permute(0, 2, 3, 1).reshape(cin, 4 * o), mx.float32)  # [C, (a, b, O)]
    bias = _arr(layer.bias, mx.float32)

    def apply(x: mx.array) -> mx.array:
        b, h, ww, _ = x.shape
        y = (x @ w).reshape(b, h, ww, 2, 2, o).transpose(0, 1, 3, 2, 4, 5).reshape(b, 2 * h, 2 * ww, o)
        return y + bias

    return apply


def _upsample(seq: Any) -> Callable[[mx.array], mx.array]:
    """`MaskDecoderHQ`'s `ConvTranspose2d, LayerNorm2d, GELU, ConvTranspose2d` in float32."""
    up1, norm, up2 = _conv_t2(seq[0]), _layer_norm(seq[1], mx.float32), _conv_t2(seq[3])
    return lambda x: up2(_gelu(norm(up1(x))))


class MlxSamEncoder:
    """`[B, S, S, 3]` normalised, padded images (float32) -> float32 embedding `[B, S/16, S/16, 256]` (channels last).

    `rel`: per block, the `(Rh, Rw, ...)` relative-position tables of its window (`sam.fast.rel_tables`). With `hq`
    (a `MaskDecoderHQ`) the call returns `(embedding, hq_features [B, S/4, S/4, 32])`.
    """

    def __init__(self, enc: Any, rel: Sequence[Sequence[torch.Tensor]], dtype: str, hq: Any = None) -> None:
        """The MLX layers of the upstream `ImageEncoderViT` `enc`, in `dtype` (the neck's output float32)."""
        d = mx.float16 if dtype == "float16" else mx.float32
        conv = enc.patch_embed.proj
        p, c = conv.kernel_size[0], conv.out_channels
        patch_w = _arr(conv.weight.permute(2, 3, 1, 0).reshape(-1, c), d)  # [(row, col, channel), C]
        patch_b = _arr(conv.bias, d)
        pos = _arr(enc.pos_embed[0], d)
        blocks = [_block(blk, r, d) for blk, r in zip(enc.blocks, rel, strict=True)]
        first_global = next((i for i, blk in enumerate(enc.blocks) if blk.window_size == 0), None)
        if hq is not None and first_global is None:
            msg = "HQ-SAM needs a global-attention block (its early feature)"
            raise ValueError(msg)
        n = enc.neck  # conv 1x1, LayerNorm2d, conv 3x3, LayerNorm2d
        neck1 = _arr(n[0].weight.flatten(1).T, d)
        neck_norm1, neck_norm2 = _layer_norm(n[1], d), _layer_norm(n[3], d)
        neck3 = _arr(n[2].weight.permute(0, 2, 3, 1), d)  # OHWI
        hq_embed = _upsample(hq.embedding_encoder) if hq is not None else None
        hq_vit = _upsample(hq.compress_vit_feat) if hq is not None else None

        def forward(x: mx.array) -> mx.array | tuple[mx.array, mx.array]:
            b, s = x.shape[0], x.shape[1]
            t = s // p
            x = x.astype(d).reshape(b, t, p, t, p, 3).transpose(0, 1, 3, 2, 4, 5).reshape(b, t, t, -1)
            x = mx.addmm(patch_b, x, patch_w) + pos
            early = None
            for i, blk in enumerate(blocks):
                x = blk(x)
                if i == first_global:
                    early = x
            y = neck_norm2(mx.conv2d(neck_norm1(x @ neck1), neck3, padding=1)).astype(mx.float32)
            if hq_embed is None or hq_vit is None or early is None:
                return y
            return y, hq_embed(y) + hq_vit(early.astype(mx.float32))

        self.forward = mx.compile(forward)

    def __call__(self, x: mx.array) -> mx.array | tuple[mx.array, mx.array]:
        """The embedding (and SAM-HQ's feature) of a batch of encoder inputs, channels last."""
        return self.forward(x)
