"""OpenCV's `cv2.resize(..., interpolation=INTER_LINEAR)` of uint8 images on the device, as mmcv-based detectors use.

mmdetection resizes the uint8 image on the CPU with OpenCV; this reproduces its 8-bit path on any device with integer
arithmetic:

- coefficients: source position `(d + 0.5) * scale - 0.5` in float32, split into the left tap and a fraction, both
  weights rounded to 11 bits (`INTER_RESIZE_COEF_BITS`). Columns outside the image take the border pixel with weight
  one; rows outside keep their two weights and clamp the row indices (two truncations instead of one)
- horizontal pass in integers (`S[x0] * a0 + S[x1] * a1`), vertical pass as OpenCV's
  `(((b0 * (r0 >> 4)) >> 16) + ((b1 * (r1 >> 4)) >> 16) + 2) >> 2`
- an exact 2x shrink in both axes is OpenCV's fast area path instead: the rounded mean of each 2 x 2 block
"""

from __future__ import annotations

import functools
from typing import cast

import numpy as np
import torch

_COEF_BITS = 11  # fixed: OpenCV INTER_RESIZE_COEF_BITS
_AREA_FAST = 2  # fixed: OpenCV resizes an exact 2x shrink with INTER_LINEAR as 2 x 2 area means


@functools.lru_cache(maxsize=64)
def _taps(n_in: int, n_out: int, *, rows: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Left / right source indices `[n_out]` (int64) and their 11-bit weights (int32) along one axis."""
    scale = n_in / n_out  # double, as OpenCV's `1. / inv_scale`
    f = ((np.arange(n_out, dtype=np.float64) + 0.5) * scale - 0.5).astype(np.float32)
    s = np.floor(f).astype(np.int64)
    f = f - s.astype(np.float32)
    if not rows:
        low, high = s < 0, s >= n_in - 1
        f[low | high] = 0.0
        s[low] = 0
        s[high] = n_in - 1
    one = np.float32(1 << _COEF_BITS)
    a0 = np.rint((np.float32(1.0) - f) * one).astype(np.int32)  # cvRound: to nearest, ties to even
    a1 = np.rint(f * one).astype(np.int32)
    return (
        torch.from_numpy(np.clip(s, 0, n_in - 1)),
        torch.from_numpy(np.clip(s + 1, 0, n_in - 1)),
        torch.from_numpy(a0),
        torch.from_numpy(a1),
    )


@functools.lru_cache(maxsize=64)
def _device_taps(n_in: int, n_out: int, device: torch.device, *, rows: bool) -> tuple[torch.Tensor, ...]:
    """`_taps` on the device (row weights as a column), cached: no host copies per call."""
    taps = _taps.__wrapped__ if torch.compiler.is_compiling() else _taps  # dynamo cannot trace through lru_cache
    t = [v.to(device) for v in taps(n_in, n_out, rows=rows)]
    if rows:
        t[2], t[3] = t[2][:, None], t[3][:, None]
    return tuple(t)


def cv2_linear(x: torch.Tensor, h: int, w: int, scale: float = 1.0) -> torch.Tensor:
    """`cv2.resize` (INTER_LINEAR) of `[B, C, H, W]` 8-bit levels to `h x w`; uint8 levels.

    `x` is uint8, or float whose values times `scale` round to the levels (0..255: float levels with `scale` 1, an
    image in [0, 1] with `scale` 255). Only the source pixels the taps read are converted (no full-size copy).
    """
    hin, win = x.shape[-2:]

    def levels(t: torch.Tensor) -> torch.Tensor:
        return t.mul(scale).round_().to(torch.int32) if t.is_floating_point() else t  # uint8 promotes in the products

    if (hin, win) == (h, w):
        return levels(x).to(torch.uint8)
    if hin == _AREA_FAST * h and win == _AREA_FAST * w:
        v = cast("torch.Tensor", sum(levels(x[..., i::2, j::2]).to(torch.int32) for i in (0, 1) for j in (0, 1)))
        return ((v + 2) >> 2).to(torch.uint8)
    taps = _device_taps.__wrapped__ if torch.compiler.is_compiling() else _device_taps
    x0, x1, a0, a1 = taps(win, w, x.device, rows=False)
    y0, y1, b0, b1 = taps(hin, h, x.device, rows=True)
    out = []
    for y, b in ((y0, b0), (y1, b1)):
        rows = x.index_select(-2, y)
        r = levels(rows.index_select(-1, x0)) * a0  # horizontal pass, int32
        r += levels(rows.index_select(-1, x1)) * a1
        r >>= 4
        r *= b
        r >>= 16
        out.append(r)
    r = out[0].add_(out[1]).add_(2)
    return (r >> 2).to(torch.uint8)
