"""LoMa on Apple MLX, written in `mlx.core` operations (`MlxLoMa(net, dtype)`).

An image -> keypoints and descriptions; two images' -> matches.

`Matcher.describe` and `Matcher.match` without an exported graph: the DaD detector and the DeDoDe descriptor are
`fastforks.shared.dedode.mlx`'s (the descriptor sampled at the keypoints as `sparse.sample`), the DINOv2 of B / L /
G / R `mlx_vit`'s, the transformer layers `fastforks.shared.lightglue.mlx`'s (in the run dtype; the cross attention as
MLX's fused attention, which accumulates in float32: the explicit similarity overflowed float16). Keypoint sampling
(softmax over all pixels, 3 x 3 non-maximum suppression, top-k, sub-pixel refinement), the assignment (the product
of the row and column softmaxes) and the mutual-maximum filter are float32. The images' resizes (bicubic with
antialiasing) stay PyTorch's on the CPU, as `Matcher._resize`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.shared.dedode.mlx import MlxDescriptor, MlxDetector
from fastforks.shared.lightglue.mlx import _heads, _Layer, _Linear, _merge

if TYPE_CHECKING:
    from .model import Matcher

__all__ = ["MlxLoMa"]

EVAL_EVERY = 3  # transformer layers per evaluation


def _cast(obj: Any, dtype: mx.Dtype) -> list[mx.array]:  # noqa: ANN401 - the `lightglue.mlx` layer objects
    """Every `mx.array` attribute of `obj` and of its nested `lightglue.mlx` helpers in `dtype` (evaluated)."""
    out = []
    for k, v in vars(obj).items():
        if isinstance(v, mx.array):
            setattr(obj, k, v.astype(dtype))
            out.append(getattr(obj, k))
        elif type(v).__module__ == _Layer.__module__:
            out += _cast(v, dtype)
    mx.eval(out)
    return out


class _CrossLayer(_Layer):
    """`lightglue.mlx`'s layer with the cross attention as fused attention.

    LoMa's similarities pass float16's range (descriptions near 20 after four layers: NaN from the fifth), which the
    fused kernel accumulates in float32.
    """

    def cross_attn(self, x0: mx.array, x1: mx.array) -> tuple[mx.array, mx.array]:
        """`CrossBlock.forward`: the softmax per direction of the scaled similarities."""
        qk0, qk1 = (_heads(self.to_qk(x), self.heads)[None] for x in (x0, x1))
        v0, v1 = (_heads(self.to_v(x), self.heads)[None] for x in (x0, x1))
        scale = self.cross_scale**2  # `to_qk` scaled on both sides
        m0 = mx.fast.scaled_dot_product_attention(qk0, qk1, v1, scale=scale)[0]
        m1 = mx.fast.scaled_dot_product_attention(qk1, qk0, v0, scale=scale)[0]
        return self.cross_ffn(x0, self.to_out(_merge(m0))), self.cross_ffn(x1, self.to_out(_merge(m1)))


def _linspace(n: int) -> mx.array:
    """`torch.linspace(-1 + 1 / n, 1 - 1 / n, n)`: the normalised pixel centres."""
    return mx.array(np.linspace(-1 + 1 / n, 1 - 1 / n, n).astype(np.float32))


def sample_keypoints(logits: mx.array, k: int, temp: float) -> mx.array:
    """`net.dad.sample_keypoints` with a 3 x 3 window: float32 logits `[H, W]` -> normalised keypoints `[k, 2]`."""
    h, w = logits.shape
    probs = mx.softmax(logits.reshape(-1)).reshape(h, w)
    padded = mx.pad(probs, 1)  # zeros: below every probability, as max-pool's -inf padding
    peak = probs
    for dy in range(3):
        for dx in range(3):
            peak = mx.maximum(peak, padded[dy : dy + h, dx : dx + w])
    flat = mx.where(probs == peak, probs, 0).reshape(-1)
    k = min(k, h * w)
    top = mx.argpartition(-flat, k - 1)[:k]
    inds = top[mx.argsort(-flat[top])]  # `torch.topk`'s order: best first
    y, x = inds // w, inds % w
    lp = mx.pad(logits, 1).reshape(-1)
    patch = mx.stack([lp[(y + dy) * (w + 2) + x + dx] for dy in range(3) for dx in range(3)])  # [9, k], row-major
    weights = mx.softmax(patch / temp, axis=0)
    steps = mx.array([-1.0, 0.0, 1.0])
    ox = (weights * mx.tile(steps, 3)[:, None]).sum(axis=0) * (2 / w)
    oy = (weights * mx.repeat(steps, 3)[:, None]).sum(axis=0) * (2 / h)
    return mx.stack([_linspace(w)[x] + ox, _linspace(h)[y] + oy], axis=-1)


class MlxLoMa:
    """The `Matcher` `net` (eval, float32, on the CPU) on MLX in `dtype` (the name)."""

    def __init__(self, net: Matcher, dtype: str) -> None:
        """Copy the weights of `net` to MLX."""
        self.dtype = getattr(mx, dtype)
        self.detector = MlxDetector(net.detector, dtype)
        self.descriptor = MlxDescriptor(net.descriptor, dtype)
        m = net.matcher
        self.input_proj = _Linear(m.input_proj) if hasattr(m.input_proj, "weight") else None
        self.posenc = _Linear(m.posenc.Wr)
        self.layers = [_CrossLayer(layer) for layer in m.transformers]
        for obj in [self.input_proj, *self.layers]:
            if obj is not None:
                _cast(obj, self.dtype)
        self.final_proj = _Linear(m.log_assignment[-1].final_proj)  # float32

    def keypoints(self, image: mx.array, k: int, temp: float) -> mx.array:
        """The normalised `[1, H, W, 3]` detector image -> normalised keypoints `[k, 2]`."""
        return sample_keypoints(self.detector(image), k, temp)

    def descriptions(self, image: mx.array, keypoints: mx.array) -> mx.array:
        """The `[1, S, S, 3]` descriptor image and normalised keypoints -> float32 descriptions `[N, C]`."""
        return self.descriptor(image, keypoints)

    def _encoding(self, kpts: mx.array) -> mx.array:
        p = self.posenc(kpts)
        return mx.repeat(mx.stack([mx.cos(p), mx.sin(p)])[:, None], 2, axis=-1).astype(self.dtype)

    def match(self, k0: mx.array, k1: mx.array, d0: mx.array, d1: mx.array, threshold: float) -> tuple[mx.array, ...]:
        """`LoMaNet` and `filter_matches`: the match index per keypoint of image 1 (-1: none) and its score."""
        descs = [d.astype(self.dtype) for d in (d0, d1)]
        if self.input_proj is not None:
            descs = [self.input_proj(d) for d in descs]
        encs = [self._encoding(k) for k in (k0, k1)]
        for i, layer in enumerate(self.layers):
            descs = list(layer(*descs, *encs))
            if i % EVAL_EVERY == EVAL_EVERY - 1:
                mx.eval(descs)  # bounds the queued graph
        scale = descs[0].shape[-1] ** 0.25
        p0, p1 = (self.final_proj(d.astype(mx.float32)) / scale for d in descs)
        # the row and column softmaxes as log-sum-exps along the last axis of the matrix and of its transpose
        sim, sim_t = p0 @ p1.T, p1 @ p0.T
        lse0, lse1 = mx.logsumexp(sim, axis=1, keepdims=True), mx.logsumexp(sim_t, axis=1, keepdims=True)
        scores = mx.exp(sim - lse0) * mx.exp(sim - lse1.T)
        scores_t = mx.exp(sim_t - lse1) * mx.exp(sim_t - lse0.T)
        m0, m1 = mx.argmax(scores, axis=1).astype(mx.int32), mx.argmax(scores_t, axis=1).astype(mx.int32)
        best = mx.max(scores, axis=1)
        mutual = m1[m0] == mx.arange(m0.shape[0])
        mscores = mx.where(mutual, best, 0)
        return mx.where(mutual & (mscores > threshold), m0, -1), mscores
