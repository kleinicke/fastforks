"""XFeat's keypoint selection, descriptor sampling and nearest-neighbour matching on Apple MLX.

The steps of upstream `XFeat.detectAndCompute` after the network (`net/xfeat.py`), on the dense maps of `mlx_net`,
without the padded keypoint list: the reliability score is computed for every pixel (the 5x5 maximum and the
two `grid_sample`s at integer pixel positions are fixed gathers), so the keypoints are the positive entries of one
score map. That map is the only host read per image (ranking a few thousand candidates is cheaper there than a
device sort of every pixel); the descriptors stay on MLX for the matcher.
"""

from __future__ import annotations

from functools import lru_cache, reduce

import numpy as np

from fastforks._core.mlx import mx

__all__ = ["detect", "mutual_nearest", "score_map"]

_F = np.float32
_Table = tuple[mx.array | None, mx.array, mx.array, mx.array, mx.array]


def _unnormalized(n: int, full: int, size: int) -> np.ndarray:
    """Sample positions of the pixels `0..n-1` in a map of `size`: `InterpolateSparse2d.normgrid` then `grid_sample`.

    Float32 step by step as PyTorch computes it: `2 * (x / (full - 1)) - 1`, then `((g + 1) * size - 1) / 2`
    (`align_corners=False`).
    """
    g = _F(2) * (np.arange(n, dtype=_F) / _F(full - 1)) - _F(1)
    return ((g + _F(1)) * _F(size) - _F(1)) / _F(2)


@lru_cache(maxsize=8)
def _score_tables(h: int, w: int) -> tuple[_Table, _Table]:
    """Per axis (y, x) of an `h x w` heatmap: its nearest tap (index, in bounds), the bilinear taps of the h/8 map."""
    tables: list[_Table] = []
    for n in (h, w):
        near = np.rint(_unnormalized(n, n, n))  # round half to even, as `grid_sample`'s nearest
        inside = (near >= 0) & (near <= n - 1)
        pos = _unnormalized(n, n, n // 8)
        low = np.floor(pos)
        index = np.clip(near, 0, n - 1).astype(np.int32)
        tables.append(
            (
                None if np.array_equal(index, np.arange(n)) else mx.array(index),  # every pixel its own tap
                mx.array(inside.astype(_F)),
                mx.array(low.astype(np.int32) + 1),  # index into the map padded by one zero on each side
                mx.array(low + _F(1) - pos),  # weight of the low tap
                mx.array(pos - low),
            )
        )
    return tables[0], tables[1]


def _local_max(x: mx.array, radius: int) -> mx.array:
    """`MaxPool2d(2 * radius + 1, stride=1, padding=radius)` of `[H, W]`: a running maximum per axis."""
    for axis in (0, 1):
        n = x.shape[axis]
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        p = mx.pad(x, pad, constant_values=-float("inf"))
        x = reduce(mx.maximum, [p[k : k + n] if axis == 0 else p[:, k : k + n] for k in range(2 * radius + 1)])
    return x


def score_map(heat: mx.array, reliability: mx.array, threshold: float) -> mx.array:
    """`[H, W]`: upstream's keypoint score where a pixel is a keypoint, else 0.

    A keypoint is a 5x5 maximum of `heat` above `threshold` (upstream `NMS`); its score is `heat` (nearest) times
    `reliability` `[H/8, W/8]` (bilinear), both read through `InterpolateSparse2d`, whose grid is normalised by
    `size - 1`: the nearest tap of the last row and column falls outside the map (score 0).
    """
    h, w = heat.shape
    (ny, iny, y0, wy0, wy1), (nx, inx, x0, wx0, wx1) = _score_tables(h, w)
    peak = (heat == _local_max(heat, 2)) & (heat > threshold)
    nearest = heat if ny is None else heat[ny]
    nearest = (nearest if nx is None else nearest[:, nx]) * (iny[:, None] * inx[None])
    rel = mx.pad(reliability, 1)
    top, bottom = rel[y0], rel[y0 + 1]
    bilinear = (
        top[:, x0] * (wx0[None] * wy0[:, None])
        + top[:, x0 + 1] * (wx1[None] * wy0[:, None])
        + bottom[:, x0] * (wx0[None] * wy1[:, None])
        + bottom[:, x0 + 1] * (wx1[None] * wy1[:, None])
    )
    return mx.where(peak, nearest * bilinear, 0.0)


def _cubic(t: np.ndarray) -> np.ndarray:
    """PyTorch's four cubic convolution weights (A = -0.75) of the fractions `t`: `[N, 4]` float32."""
    a = _F(-0.75)

    def inner(x: np.ndarray) -> np.ndarray:  # |x| <= 1
        return ((a + _F(2)) * x - (a + _F(3))) * x * x + _F(1)

    def outer(x: np.ndarray) -> np.ndarray:  # 1 < |x| < 2
        return ((a * x - _F(5) * a) * x + _F(8) * a) * x - _F(4) * a

    u = _F(1) - t
    return np.stack([outer(t + _F(1)), inner(t), inner(u), outer(u + _F(1))], axis=-1).astype(_F)


def sample_bicubic(feats: mx.array, xs: np.ndarray, ys: np.ndarray, h: int, w: int) -> mx.array:
    """`InterpolateSparse2d('bicubic')` of `[h8, w8, C]` at the pixels `(xs, ys)` of the `h x w` image: `[N, C]`."""
    fh, fw, c = feats.shape
    taps = []
    for v, full, size in ((ys, h, fh), (xs, w, fw)):
        g = _F(2) * (v.astype(_F) / _F(full - 1)) - _F(1)
        pos = ((g + _F(1)) * _F(size) - _F(1)) / _F(2)
        low = np.floor(pos)
        idx = low.astype(np.int64)[:, None] + np.arange(-1, 3)
        weight = _cubic(pos - low) * ((idx >= 0) & (idx <= size - 1))  # zeros padding
        taps.append((np.clip(idx, 0, size - 1), weight.astype(_F)))
    (iy, wy), (ix, wx) = taps
    flat = feats.reshape(fh * fw, c)  # a tap's channels are one row
    index = mx.array((iy[:, :, None] * fw + ix[:, None, :]).astype(np.int32))  # [N, 4, 4]
    rows = (flat[index] * mx.array(wx)[:, None, :, None]).sum(axis=2)  # along x first, as PyTorch
    return (rows * mx.array(wy)[:, :, None]).sum(axis=1)


def detect(feats: mx.array, score: mx.array, top_k: int) -> tuple[np.ndarray, mx.array]:
    """Keypoints `[N, 2]` (int64 `(x, y)` in the network image, best first) and their unit descriptors `[N, 64]`.

    `feats` `[H/8, W/8, 64]` (unit length per pixel, upstream's `M1`), `score` `[H, W]` (`score_map`). At most
    `top_k` keypoints of positive score, as upstream keeps.
    """
    h, w = score.shape
    scores = np.asarray(score).reshape(-1)
    found = np.flatnonzero(scores > 0)
    found = found[found > 0]  # pixel (0, 0) is upstream's padding value: never a keypoint
    values = scores[found]
    if len(found) > top_k:  # rank only what can be kept: the candidates are several times `top_k`
        keep = values >= np.partition(values, len(values) - top_k)[len(values) - top_k]
        found, values = found[keep], values[keep]
    found = found[np.argsort(-values, kind="stable")[:top_k]]
    ys, xs = np.divmod(found, w)
    desc = sample_bicubic(feats, xs, ys, h, w)
    desc = desc / mx.maximum(mx.linalg.norm(desc, axis=-1, keepdims=True), 1e-12)
    return np.stack([xs, ys], axis=-1), desc


def mutual_nearest(desc1: mx.array, desc2: mx.array) -> tuple[np.ndarray, np.ndarray]:
    """Upstream `match(min_cossim=-1)`: mutual nearest neighbours `[S, 2]` (int64) and their cosine similarities."""
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return np.empty((0, 2), np.int64), np.empty((0,), _F)
    sim = desc1 @ desc2.T
    # the transposed product again: MLX's argmax along the first axis costs more than a second matrix product
    to2, to1, best = mx.argmax(sim, axis=1), mx.argmax(desc2 @ desc1.T, axis=1), mx.max(sim, axis=1)
    mx.eval(to2, to1, best)
    to2, to1, best = np.asarray(to2).astype(np.int64), np.asarray(to1), np.asarray(best)
    idx1 = np.flatnonzero(to1[to2] == np.arange(len(to2)))
    return np.stack([idx1, to2[idx1]], axis=-1), best[idx1]
