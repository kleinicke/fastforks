"""SegFormer on Apple MLX, written in `mlx.core` operations (`MlxSegFormer(net, dtype)`): a normalised image -> logits.

`_Net.forward` (the transformers modules with merged key / value projections and the folded all-MLP head), channels
last. Written out instead of replayed from an exported graph: the export, translation and segment tuning were most of
a 1-6 s first call whose network runs in 0.04-0.35 s.

- each block is one `mx.compile`d function per stage (its weights arguments, so a stage's blocks share one trace):
  pre-norm efficient attention (the spatial reduction a space-to-depth reshape and one GEMM, MLX's fused attention),
  pre-norm Mix-FFN (the depthwise 3 x 3 the shared depthwise kernel, the exact GELU in float32);
- the head: per stage one GEMM at the stage's resolution (projection, fuse slice and batch norm folded by `_Net`),
  the torch-identical bilinear resize (`mlx_seg.resize`), the sum, ReLU and the classifier.

Each block is evaluated as it is scheduled, one in flight (`_step`): MLX frees a command buffer's transients only
when it completes, and one lazy graph of the whole encoder holds far more than its live set.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared.mlx_seg import resize

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["MlxSegFormer"]

ONE = (1, 1)


def _gelu(x: mx.array) -> mx.array:
    """Exact (erf) GELU, in float32."""
    y = x.astype(mx.float32)
    return (y * 0.5 * (1 + mx.erf(y * 0.7071067811865476))).astype(x.dtype)


def _layer_norm(x: mx.array, w: mx.array, b: mx.array, eps: float) -> mx.array:
    """`mx.fast.layer_norm`, written out (float32 statistics) below 256 channels (the fused kernel 4-6x slower)."""
    if x.shape[-1] >= 256:  # noqa: PLR2004
        return mx.fast.layer_norm(x, w, b, eps)
    xf = x.astype(mx.float32)
    d = xf - xf.mean(axis=-1, keepdims=True)
    return (d * mx.rsqrt(mx.square(d).mean(axis=-1, keepdims=True) + eps)).astype(x.dtype) * w + b


@functools.cache
def _norm(eps: float) -> Callable[..., mx.array]:
    """`_layer_norm` compiled (the patch embeddings' and stages' norms, outside the blocks)."""
    return mx.compile(lambda x, w, b: _layer_norm(x, w, b, eps))


def _space_to_depth(x: mx.array, r: int) -> mx.array:
    """`[B, H, W, C]` -> `[B, H / r, W / r, r * r * C]` (rows `(i, j, c)`), the map cropped as a stride-`r` conv."""
    b, h, w, c = x.shape
    h, w = h // r, w // r
    x = x[:, : h * r, : w * r].reshape(b, h, r, w, r, c).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(b, h, w, r * r * c)


def _block_fn(heads: int, sr: int, eps: tuple[float, float, float], scale: float) -> Callable[..., mx.array]:
    """`_Net._block` on `[B, H, W, C]` tokens, compiled."""
    e1, e2, er = eps

    def block(x: mx.array, *w: mx.array) -> mx.array:
        n1w, n1b, wq, bq, wkv, bkv, wo, bo, n2w, n2b, w1, b1, wd, bd, w2, b2, *red = w
        b, h, wd_, c = x.shape
        y = _layer_norm(x, n1w, n1b, e1)
        q = mx.addmm(bq, y, wq).reshape(b, h * wd_, heads, -1).transpose(0, 2, 1, 3)
        if sr > 1:
            wr, br, nrw, nrb = red
            y = _layer_norm(mx.addmm(br, _space_to_depth(y, sr), wr), nrw, nrb, er)
        kv = mx.addmm(bkv, y.reshape(b, -1, c), wkv).reshape(b, -1, 2, heads, c // heads).transpose(2, 0, 3, 1, 4)
        o = mx.fast.scaled_dot_product_attention(q, kv[0], kv[1], scale=scale)
        x = x + mx.addmm(bo, o.transpose(0, 2, 1, 3).reshape(b, h, wd_, c), wo)
        y = mx.addmm(b1, _layer_norm(x, n2w, n2b, e2), w1)
        y = _gelu(ops.depthwise(y, wd, bd, ONE, ONE, ONE))
        return x + mx.addmm(b2, y, w2)

    return mx.compile(block)


_PENDING: list[mx.array] = []  # the step in flight (`_step`)


def _step(x: mx.array) -> mx.array:
    """Schedule `x` and wait for the step before it (pipeline depth 1: the GPU stays busy, the transients bounded)."""
    mx.async_eval(x)
    if _PENDING:
        mx.eval(_PENDING.pop())
    _PENDING.append(x)
    return x


class MlxSegFormer:
    """The network: normalised `[B, 3, H, W]` float32 -> logits `[B, H / 4, W / 4, classes]` float32 (`_Net`)."""

    def __init__(self, net: Any, dtype: str) -> None:  # noqa: ANN401 - `model._Net` (float32, folded head)
        """The weights of `net` in `dtype`."""
        self.dtype = getattr(mx, dtype)
        self.stages = []
        fns: dict[tuple, Callable[..., mx.array]] = {}
        for st in net.stages:
            pe = st.patch_embeddings
            embed = (self._a(pe.proj.weight.permute(0, 2, 3, 1)), self._a(pe.proj.bias), *self._ln(pe.layer_norm))
            conv = (pe.proj.stride[0], pe.proj.padding[0], pe.layer_norm.eps)
            blocks = []
            for b in st.blocks:
                a, mlp = b.attention, b.mlp
                dw = mlp.dwconv.dwconv
                if dw.out_channels % 4:
                    msg = "SegFormer MLX: Mix-FFN width not a multiple of 4"
                    raise NotImplementedError(msg)
                w = [*self._ln(b.layernorm_before), *self._lin(a.q_proj.weight, a.q_proj.bias)]
                w += [*self._lin(b._kv_w, b._kv_b), *self._lin(a.o_proj.weight, a.o_proj.bias)]  # noqa: SLF001
                w += [*self._ln(b.layernorm_after), *self._lin(mlp.fc1.weight, mlp.fc1.bias)]
                w += [self._a(dw.weight[:, 0].permute(1, 2, 0)), self._a(dw.bias)]
                w += self._lin(mlp.fc2.weight, mlp.fc2.bias)
                sr, er = a.sequence_reduction_ratio, 0.0
                if sr > 1:
                    r = a.sequence_reduction
                    wr = r.sequence_reduction.weight.permute(2, 3, 1, 0).reshape(-1, r.sequence_reduction.out_channels)
                    w += [self._a(wr), self._a(r.sequence_reduction.bias), *self._ln(r.layer_norm)]
                    er = r.layer_norm.eps
                key = (a.num_attention_heads, sr, b.layernorm_before.eps, b.layernorm_after.eps, er, a.scaling)
                if key not in fns:
                    fns[key] = _block_fn(key[0], sr, key[2:5], a.scaling)
                mx.eval(w)
                blocks.append((fns[key], w))
            norm = (*self._ln(st.layer_norm), st.layer_norm.eps)
            self.stages.append((embed, conv, blocks, norm))
        self.head = [self._a(getattr(net, f"head_w{i}").T) for i in range(len(net.stages))]
        self.head_b = self._a(net.head_b)
        self.cls_w, self.cls_b = self._lin(net.cls_w, net.cls_b)
        mx.eval(self.head, self.head_b, self.cls_w, self.cls_b, [s[0] for s in self.stages])

    def _a(self, t: Any) -> mx.array:  # noqa: ANN401 - a torch tensor
        return mx.array(np.ascontiguousarray(t.detach().float().numpy())).astype(self.dtype)

    def _lin(self, w: Any, b: Any) -> list[mx.array]:  # noqa: ANN401 - torch tensors
        return [self._a(w.T), self._a(b)]

    def _ln(self, m: Any) -> list[mx.array]:  # noqa: ANN401 - a `LayerNorm`
        return [self._a(m.weight), self._a(m.bias)]

    def __call__(self, image: mx.array) -> mx.array:
        """`_Net.forward`."""
        try:
            return self._forward(image)
        finally:
            _PENDING.clear()

    def _forward(self, image: mx.array) -> mx.array:
        x = mx.transpose(image, (0, 2, 3, 1)).astype(self.dtype)
        feats = []
        for (pw, pb, nw, nb), (stride, pad, eps), blocks, (fw, fb, feps) in self.stages:
            x = _norm(eps)(mx.conv2d(x, pw, stride=stride, padding=pad) + pb, nw, nb)
            for fn, w in blocks:
                x = _step(fn(x, *w))
            x = _step(_norm(feps)(x, fw, fb))
            feats.append(x)
        size = feats[0].shape[1:3]
        acc = None
        for f, w in zip(feats, self.head, strict=True):
            y = resize(f @ w, size)
            acc = y if acc is None else acc + y
            acc = _step(acc)
        assert acc is not None  # noqa: S101 - four stages
        return mx.addmm(self.cls_b, mx.maximum(acc + self.head_b, 0), self.cls_w).astype(mx.float32)
