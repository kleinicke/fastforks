"""LightGlue's forward on Apple MLX, eager (`MlxLightGlue(net)`): one pair, float32.

The keypoint counts differ per pair and shrink per layer (point pruning), so an exported graph would be built anew for
every pair; these are the vendored network's own steps (`net/lightglue.py`, its CPU branch: pruning at any keypoint
count, the cross attention as a similarity matrix with a softmax per direction) written with `mlx.core`, which
needs no fixed shapes. The weights are read from the PyTorch module once. Each layer's pruning masks and the final
assignment are the only host reads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx

if TYPE_CHECKING:
    from .net.lightglue import LightGlue

__all__ = ["Features", "MlxLightGlue"]

Features = tuple[np.ndarray, mx.array, np.ndarray]
"""One image: keypoints `[N, 2]` in pixels, descriptors `[N, D]` on MLX, the image's `(width, height)`."""


class _Linear:
    """`nn.Linear` with its weight transposed once."""

    def __init__(self, layer: Any) -> None:  # noqa: ANN401 - torch modules of the vendored net
        self.w = mx.array(layer.weight.detach().float().numpy().T.copy())
        self.b = None if layer.bias is None else mx.array(layer.bias.detach().float().numpy())

    def __call__(self, x: mx.array) -> mx.array:
        y = x @ self.w
        return y if self.b is None else y + self.b


class _Ffn:
    """The blocks' `Linear, LayerNorm, GELU, Linear` on `cat(x, message)`."""

    def __init__(self, ffn: Any) -> None:  # noqa: ANN401 - torch modules of the vendored net
        self.fc1, self.fc2 = _Linear(ffn[0]), _Linear(ffn[3])
        self.g, self.b, self.eps = (
            mx.array(ffn[1].weight.detach().numpy()),
            mx.array(ffn[1].bias.detach().numpy()),
            float(ffn[1].eps),
        )

    def __call__(self, x: mx.array, message: mx.array) -> mx.array:
        y = mx.fast.layer_norm(self.fc1(mx.concatenate([x, message], axis=-1)), self.g, self.b, self.eps)
        return x + self.fc2(y * (0.5 + 0.5 * mx.erf(y * 0.7071067811865476)))  # exact GELU


def _heads(x: mx.array, heads: int) -> mx.array:
    """`[N, heads * d]` as `[heads, N, d]`."""
    return x.reshape(x.shape[0], heads, -1).transpose(1, 0, 2)


def _merge(x: mx.array) -> mx.array:
    """`[heads, N, d]` as `[N, heads * d]`."""
    return x.transpose(1, 0, 2).reshape(x.shape[1], -1)


class _Layer:
    """One `TransformerLayer`: self attention with the rotary encoding per image, then cross attention."""

    def __init__(self, layer: Any) -> None:  # noqa: ANN401 - torch modules of the vendored net
        s, c = layer.self_attn, layer.cross_attn
        self.heads = int(s.num_heads)
        self.qkv, self.out, self.self_ffn = _Linear(s.Wqkv), _Linear(s.out_proj), _Ffn(s.ffn)
        self.to_qk, self.to_v, self.to_out, self.cross_ffn = (
            _Linear(c.to_qk),
            _Linear(c.to_v),
            _Linear(c.to_out),
            _Ffn(c.ffn),
        )
        self.cross_scale = float(c.scale) ** 0.5

    def self_attn(self, x: mx.array, enc: mx.array) -> mx.array:
        """`SelfBlock.forward` of `[N, D]` with the encoding `[2, heads, N, d]` (cos, sin)."""
        qkv = self.qkv(x).reshape(x.shape[0], self.heads, -1, 3).transpose(1, 0, 2, 3)  # [heads, N, d, 3]
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]

        def rotary(t: mx.array) -> mx.array:
            pairs = t.reshape(*t.shape[:-1], -1, 2)
            half = mx.stack([-pairs[..., 1], pairs[..., 0]], axis=-1).reshape(t.shape)
            return t * enc[0] + half * enc[1]

        q, k = rotary(q), rotary(k)
        context = mx.fast.scaled_dot_product_attention(q[None], k[None], v[None], scale=q.shape[-1] ** -0.5)[0]
        return self.self_ffn(x, self.out(_merge(context)))

    def cross_attn(self, x0: mx.array, x1: mx.array) -> tuple[mx.array, mx.array]:
        """`CrossBlock.forward`: the similarity matrix with a softmax per direction.

        The matrix is computed once per direction: MLX's softmax along the second to last axis costs more than the
        transposed product (4096 x 4096: 6.4 ms against 2.9 ms).
        """
        qk0, qk1 = (_heads(self.to_qk(x), self.heads) * self.cross_scale for x in (x0, x1))
        v0, v1 = (_heads(self.to_v(x), self.heads) for x in (x0, x1))
        m0 = mx.softmax(qk0 @ qk1.swapaxes(-1, -2), axis=-1) @ v1  # [heads, N0, N1] @ [heads, N1, d]
        m1 = mx.softmax(qk1 @ qk0.swapaxes(-1, -2), axis=-1) @ v0
        return self.cross_ffn(x0, self.to_out(_merge(m0))), self.cross_ffn(x1, self.to_out(_merge(m1)))

    def __call__(self, d0: mx.array, d1: mx.array, e0: mx.array, e1: mx.array) -> tuple[mx.array, mx.array]:
        return self.cross_attn(self.self_attn(d0, e0), self.self_attn(d1, e1))


class MlxLightGlue:
    """The matcher `net` (eval, weights loaded) on MLX: `(keypoints, descriptors, image size)` twice -> matches."""

    def __init__(self, net: LightGlue) -> None:
        """Copy the weights of `net` to MLX."""
        conf = net.conf
        if conf.add_scale_ori or conf.mp:
            msg = "MlxLightGlue: scale / orientation inputs and mixed precision are not ported"
            raise NotImplementedError(msg)
        self.conf = conf
        self.input_proj = _Linear(net.input_proj) if hasattr(net.input_proj, "weight") else (lambda x: x)
        self.posenc = _Linear(net.posenc.Wr)
        self.layers = [_Layer(layer) for layer in net.transformers]
        assignment: Any = net.log_assignment
        confidence: Any = net.token_confidence
        thresholds: Any = net.confidence_thresholds
        self.matchability = [_Linear(a.matchability) for a in assignment]
        self.final_proj = [_Linear(a.final_proj) for a in assignment]
        self.token = [_Linear(t.token[0]) for t in confidence]
        self.thresholds = [float(t) for t in thresholds]

    def _encoding(self, kpts: np.ndarray, size: np.ndarray) -> mx.array:
        """`normalize_keypoints` and the Fourier encoding: `[2, 1, N, d]` (cos, sin; per head the same)."""
        size = size.astype(np.float32)
        k = (kpts.astype(np.float32) - size / 2) / (size.max() / 2)
        p = self.posenc(mx.array(k))
        return mx.repeat(mx.stack([mx.cos(p), mx.sin(p)])[:, None], 2, axis=-1)

    def _adapt(self, i: int, descs: list[mx.array], points: int) -> tuple[bool, list[np.ndarray | None]]:
        """After layer `i`: whether to stop (adaptive depth) and per image the indices to keep (adaptive width)."""
        conf = self.conf
        early_stop, pruning = conf.depth_confidence > 0, conf.width_confidence > 0
        if not (early_stop or pruning):
            return False, [None, None]
        tokens = [mx.sigmoid(self.token[i](d))[:, 0] for d in descs] if early_stop else []
        keep = []
        if pruning:
            for k, d in enumerate(descs):
                mask = mx.sigmoid(self.matchability[i](d))[:, 0] > (1 - conf.width_confidence)
                keep.append(mask | (tokens[k] <= self.thresholds[i]) if early_stop else mask)
        mx.eval(descs, tokens, keep)  # one host read per layer
        if early_stop:
            unsure = sum(int((np.asarray(t) < self.thresholds[i]).sum()) for t in tokens)
            if 1.0 - unsure / points > conf.depth_confidence:
                return True, [None, None]
        return False, [np.flatnonzero(np.asarray(k)) for k in keep] if pruning else [None, None]

    def __call__(self, image0: Features, image1: Features) -> tuple[np.ndarray, np.ndarray]:
        """Index pairs `[S, 2]` (int64) and their scores `[S]`: `LightGlue.forward`'s `matches[0]`, `scores[0]`.

        An image is `(keypoints [N, 2] in pixels, descriptors [N, D], its (width, height))`.
        """
        empty = np.empty((0, 2), np.int64), np.empty((0,), np.float32)
        points = len(image0[0]) + len(image1[0])
        if len(image0[0]) == 0 or len(image1[0]) == 0:
            return empty
        descs = [self.input_proj(image[1]) for image in (image0, image1)]
        encs = [self._encoding(image[0], image[2]) for image in (image0, image1)]
        inds = [np.arange(len(image[0])) for image in (image0, image1)]
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            descs = list(layer(*descs, *encs))
            if i == last:
                break
            stop, keeps = self._adapt(i, descs, points)
            if stop:
                last = i
                break
            for k, keep in enumerate(keeps):
                if keep is not None and len(keep) < len(inds[k]):
                    j = mx.array(keep)
                    inds[k], descs[k], encs[k] = inds[k][keep], descs[k][j], encs[k][:, :, j]
            if len(inds[0]) == 0 or len(inds[1]) == 0:
                return empty
        # `MatchAssignment` of the last layer run and `filter_matches` (the dustbin row and column are not read).
        # Per direction its own (transposed) matrix, reduced along the last axis: MLX's reductions along the first
        # axis are several times slower.
        proj = self.final_proj[last]
        scale = descs[0].shape[-1] ** 0.25
        p0, p1 = (proj(d) / scale for d in descs)
        z0, z1 = (-mx.logaddexp(-self.matchability[last](d), 0.0) for d in descs)  # log sigmoid
        sim, sim_t = p0 @ p1.T, p1 @ p0.T
        lse0, lse1 = mx.logsumexp(sim, axis=1, keepdims=True), mx.logsumexp(sim_t, axis=1, keepdims=True)
        scores = (sim - lse0) + (sim - lse1.T) + z0 + z1.T
        scores_t = (sim_t - lse1) + (sim_t - lse0.T) + z1 + z0.T
        m0, m1 = mx.argmax(scores, axis=1), mx.argmax(scores_t, axis=1)
        best = mx.exp(mx.max(scores, axis=1))
        mx.eval(m0, m1, best)
        m0, m1, best = np.asarray(m0).astype(np.int64), np.asarray(m1), np.asarray(best)
        valid = np.flatnonzero((m1[m0] == np.arange(len(m0))) & (best > self.conf.filter_threshold))
        return np.stack([inds[0][valid], inds[1][m0[valid]]], axis=-1), best[valid]
