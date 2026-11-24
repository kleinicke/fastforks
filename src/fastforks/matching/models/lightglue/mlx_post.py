"""DISK's keypoints and descriptors from the U-Net's maps on Apple MLX (`net.disk.DISK.forward` after the U-Net).

- `peaks`: the heatmap where a pixel is a keypoint candidate (`nms2d_mask` and the score threshold), on MLX;
- `detect`: one host read of that map, upstream's top-n rule on the candidates (a few thousand), and the
  descriptors gathered from the dense map on MLX (they stay there for the matcher).

The keypoints are `heatmap_to_keypoints`'s exactly (same order: row-major), the descriptors equal up to rounding.
"""

from __future__ import annotations

from functools import reduce

import numpy as np

from fastforks._core.mlx import mx

__all__ = ["detect", "peaks"]

_EPS = 1e-12  # fixed: `F.normalize`'s default eps (upstream's call)


def peaks(heat: mx.array, window: int, threshold: float) -> mx.array:
    """`[H, W]` heatmap -> the heat where it is above `threshold` and above all its window's neighbours, else -inf.

    `nms2d_mask`: strict maxima, none within `window // 2` of the border (the +inf padding fails those). The maximum
    of the neighbours is the other rows' running maxima and the own row's without the centre.
    """
    r = window // 2
    h, w = heat.shape
    p = mx.pad(heat, [(r, r), (r, r)], constant_values=float("inf"))
    rows = reduce(mx.maximum, [p[:, k : k + w] for k in range(window)])
    others = [rows[k : k + h] for k in range(window) if k != r]
    others += [p[r : r + h, k : k + w] for k in range(window) if k != r]
    keep = (heat > reduce(mx.maximum, others)) & (heat > threshold)
    return mx.where(keep, heat, -float("inf"))


def detect(peak: mx.array, descriptors: mx.array, n: int) -> tuple[np.ndarray, mx.array]:
    """Keypoints `[N, 2]` (int64 `(x, y)`) of the `peaks` map and their unit descriptors `[N, D]` (lazy).

    `descriptors` is the dense map `[D, H', W']` (at least the peak map's size). `heatmap_to_keypoints`'s rule: the
    candidates above the `n + 1`-th largest score, then the first `n` of them.
    """
    w = peak.shape[1]
    flat = np.asarray(peak).reshape(-1)
    found = np.flatnonzero(flat > -np.inf)
    if len(found):
        score = flat[found]
        k = min(n + 1, len(found))
        found = found[score > -np.partition(-score, k - 1)[k - 1]][:n]
    ys, xs = np.divmod(found, w)
    d, _, wide = descriptors.shape
    desc = descriptors.reshape(d, -1)[:, mx.array(ys * wide + xs)].T
    desc = desc.astype(mx.float32)
    return np.stack([xs, ys], axis=-1), desc / mx.maximum(mx.linalg.norm(desc, axis=-1, keepdims=True), _EPS)
