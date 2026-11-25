"""SAM 2's image encoder (Hiera trunk + the folded FPN neck of `fast.py`) written in MLX, channels last.

Written out instead of replayed from an exported graph: the export, translation and segment tuning were most of the
first call (1.6 to 3.8 s on an M1 Pro; 0.15 to 0.6 s like this).

- A block is `fast._block_f32_stream`: in float16 the residual stream stays float32 (the norms read it, attention and
  the MLP run in float16, the adds are float32), and a stage transition's projection, whose pooled output *is* the
  stream there, runs in float32 with float32 weights.
- Attention is MLX's fused `scaled_dot_product_attention` (float32 softmax inside), which has a fused kernel at head
  dims 64, 80 and 128 only. Hiera's head dim is 96 (Tiny, Small), 56 (Base Plus) or 72 (Large). 56 is padded to 64 in
  the weights (`PAD_TO`), as zero output columns of the q / k / v projection and zero input rows of the output
  projection: the scores and the result are the same sums plus zeros, and Base Plus is 10 % faster for it. Padding
  96 to 128 or 72 to 80 was measured slower than the unfused path (the wider GEMMs cost more than the kernel saves),
  so those stay as they are.
- Windowed blocks partition the normalised map, zero-padded to whole windows (the padded tokens are keys, as
  upstream's); the query pooling of a stage's first block is a 2 x 2 max over each window.
- The blocks are evaluated one by one, since one lazy graph of the trunk keeps every block's temporaries alive at
  once. From stage 2 on a block is started (`mx.async_eval`) before the one before it is waited for, so the GPU never
  idles while Python builds the next graph. The stage-1 blocks (256 x 256 tokens, 0.3 to 0.5 GB of temporaries each)
  are evaluated alone: two of them in flight were 3 to 5 % faster and 0.3 to 0.5 GB more peak memory.
"""

# ruff: noqa: ANN401 - the upstream trunk's torch modules

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx, to_mx, to_torch
from fastforks.shared.mlx_vit import gelu

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

    from .fast import Lin, Neck

__all__ = ["MlxEncoder"]

SYNC_TOKENS = 1 << 16  # blocks on maps this large are evaluated alone (see the module docstring)
PAD_TO = {56: 64}  # head dims padded to one MLX's attention runs fused, where that was measured faster


def _arr(t: torch.Tensor | np.ndarray, dtype: mx.Dtype) -> mx.array:
    v = t if isinstance(t, np.ndarray) else t.detach().double().cpu().numpy()
    a = mx.array(np.ascontiguousarray(v, dtype=np.float32)).astype(dtype)
    mx.eval(a)
    return a


def _linear(layer: Any, dtype: mx.Dtype) -> Callable[[mx.array], mx.array]:
    w, b = _arr(layer.weight.T, dtype), _arr(layer.bias, dtype)
    return lambda x: mx.addmm(b, x, w)


def _layer_norm(layer: Any, dtype: mx.Dtype) -> Callable[[mx.array], mx.array]:
    w, b, eps = _arr(layer.weight, dtype), _arr(layer.bias, dtype), float(layer.eps)
    return lambda x: mx.fast.layer_norm(x, w, b, eps)


def _pool(x: mx.array) -> mx.array:
    """2 x 2 / stride-2 max pooling of `[B, H, W, C]` (even sides)."""
    b, h, w, c = x.shape
    return x.reshape(b, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))


def _attention(attn: Any, dtype: mx.Dtype, pad_to: dict[int, int]) -> Callable[[mx.array], mx.array]:
    """`MultiScaleAttention.forward` on windows `[B, H, W, C]` -> `[B, H', W', C_out]` (the queries pooled or not)."""
    heads = attn.num_heads
    d = attn.dim_out // heads
    dp = pad_to.get(d, d)
    w = attn.qkv.weight.detach().double().numpy().reshape(3, heads, d, attn.dim)
    b = attn.qkv.bias.detach().double().numpy().reshape(3, heads, d)
    o = attn.proj.weight.detach().double().numpy().T.reshape(heads, d, attn.dim_out)
    if dp != d:
        w = np.pad(w, ((0, 0), (0, 0), (0, dp - d), (0, 0)))
        b = np.pad(b, ((0, 0), (0, 0), (0, dp - d)))
        o = np.pad(o, ((0, 0), (0, dp - d), (0, 0)))
    qkv_w, qkv_b = _arr(w.reshape(3 * heads * dp, attn.dim).T, dtype), _arr(b.reshape(-1), dtype)
    out_w, out_b = _arr(o.reshape(heads * dp, attn.dim_out), dtype), _arr(attn.proj.bias, dtype)
    pooled, scale = attn.q_pool is not None, d**-0.5

    def run(x: mx.array) -> mx.array:
        n, h, wd, _ = x.shape
        qkv = mx.addmm(qkv_b, x.reshape(n, h * wd, -1), qkv_w).reshape(n, h * wd, 3, heads, dp)
        q = qkv[:, :, 0]
        if pooled:
            q = _pool(q.reshape(n, h, wd, heads * dp))
            h, wd = h // 2, wd // 2
            q = q.reshape(n, h * wd, heads, dp)
        k, v = qkv[:, :, 1].transpose(0, 2, 1, 3), qkv[:, :, 2].transpose(0, 2, 1, 3)
        y = mx.fast.scaled_dot_product_attention(q.transpose(0, 2, 1, 3), k, v, scale=scale)
        return mx.addmm(out_b, y.transpose(0, 2, 1, 3).reshape(n, h, wd, heads * dp), out_w)

    return run


def _partition(x: mx.array, window: int) -> tuple[mx.array, tuple[int, int]]:
    """`[B, H, W, C]` -> windows `[B * n, window, window, C]` (zero-padded bottom / right) and the padded size."""
    b, h, w, c = x.shape
    ph, pw = -h % window, -w % window
    if ph or pw:
        x = mx.pad(x, ((0, 0), (0, ph), (0, pw), (0, 0)))
    hp, wp = h + ph, w + pw
    x = x.reshape(b, hp // window, window, wp // window, window, c).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, window, window, c), (hp, wp)


def _unpartition(x: mx.array, window: int, padded: tuple[int, int], size: tuple[int, int]) -> mx.array:
    """Windows `[B * n, window, window, C]` of a `padded` map -> `[B, H, W, C]` cropped to `size`."""
    (hp, wp), (h, w) = padded, size
    b = x.shape[0] // ((hp // window) * (wp // window))
    x = x.reshape(b, hp // window, wp // window, window, window, -1).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(b, hp, wp, -1)[:, :h, :w]


def _block(blk: Any, dtype: mx.Dtype, stream: mx.Dtype, pad_to: dict[int, int]) -> Callable[[mx.array], mx.array]:
    """`MultiScaleBlock.forward` on a `stream`-dtype map, attention and MLP in `dtype`."""
    norm1, norm2 = _layer_norm(blk.norm1, stream), _layer_norm(blk.norm2, stream)
    proj = _linear(blk.proj, stream) if blk.dim != blk.dim_out else None
    attn = _attention(blk.attn, dtype, pad_to)
    fc1, fc2 = (_linear(layer, dtype) for layer in blk.mlp.layers)
    window, pooled = blk.window_size, blk.q_stride is not None

    def run(x: mx.array) -> mx.array:
        h = norm1(x)
        shortcut = x
        if proj is not None:
            shortcut = proj(h)
            shortcut = _pool(shortcut) if pooled else shortcut
        h = h.astype(dtype)
        size, win = h.shape[1:3], window
        padded = size
        if window:
            h, padded = _partition(h, window)
        h = attn(h)
        if pooled:
            win, size = window // 2, shortcut.shape[1:3]
            padded = (size[0] + -size[0] % win, size[1] + -size[1] % win) if win else size
        if window:
            h = _unpartition(h, win, padded, size)
        x = shortcut + h.astype(stream)
        return x + fc2(gelu(fc1(norm2(x).astype(dtype)))).astype(stream)

    return run


class MlxEncoder:
    """Hiera trunk and neck of one loaded SAM 2 on MLX: `fast.trunk` + `fast.neck`."""

    def __init__(
        self, trunk: Any, neck: Neck, pos: torch.Tensor, dtype_name: str, *, pad_to: dict[int, int] | None = None
    ) -> None:
        """Copy the weights of upstream `Hiera` (`trunk`), the folded `neck` and the position map `[1, T, T, C]`.

        `pad_to` replaces `PAD_TO` (the tests pad their small trunk's head dims).
        """
        dtype = getattr(mx, dtype_name)
        stream = mx.float32
        self.dtype, self.stream = dtype, stream
        conv = trunk.patch_embed.proj
        self.stride, self.padding = conv.stride[0], conv.padding[0]
        self.patch_w = _arr(conv.weight.permute(0, 2, 3, 1), dtype)
        self.patch_b = _arr(conv.bias, dtype)
        self.pos = _arr(pos, dtype)
        self.blocks = [_block(blk, dtype, stream, PAD_TO if pad_to is None else pad_to) for blk in trunk.blocks]
        self.stage_ends = set(trunk.stage_ends)

        def lin(t: Lin) -> Callable[[mx.array], mx.array]:
            w, b = _arr(t[0].T, dtype), _arr(t[1], dtype)
            return lambda x: mx.addmm(b, x.astype(dtype), w).astype(mx.float32)

        self.top, self.mid, self.s1, self.s0 = lin(neck.top), lin(neck.mid), lin(neck.s1), lin(neck.s0)
        self.no_mem = _arr(neck.no_mem, mx.float32)

    def features(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Normalised `[B, S, S, 3]` -> float32 channels-last `(image_embed, feat_s1, feat_s0)`."""
        x = mx.conv2d(x.astype(self.dtype), self.patch_w, stride=self.stride, padding=self.padding)
        x = (x + self.patch_b + self.pos).astype(self.stream)
        feats, prev = [], None
        for i, blk in enumerate(self.blocks):
            big = x.shape[1] * x.shape[2] >= SYNC_TOKENS
            x = blk(x)
            if big:
                mx.eval(x)
                prev = None
            else:
                mx.async_eval(x)
                if prev is not None:
                    mx.eval(prev)
                prev = x
            if i in self.stage_ends:
                feats.append(x)
        f0, f1, f2, f3 = feats
        top = self.top(f3)
        b, h, w, c = top.shape
        up = mx.broadcast_to(top[:, :, None, :, None], (b, h, 2, w, 2, c)).reshape(b, 2 * h, 2 * w, c)
        return self.mid(f2) + up + self.no_mem, self.s1(f1), self.s0(f0)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Normalised NCHW `[B, 3, S, S]` -> float32 NCHW `image_embed`, `feat_s1`, `feat_s0` (as `fast.neck`)."""
        out = self.features(to_mx(image.float().permute(0, 2, 3, 1)))
        mx.eval(*out)
        keys = ("image_embed", "feat_s1", "feat_s0")
        return {k: to_torch(a).permute(0, 3, 1, 2).contiguous() for k, a in zip(keys, out, strict=True)}
