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


def cv2_linear(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """`cv2.resize` (INTER_LINEAR) of `[B, C, H, W]` uint8 levels (float, values 0..255) to `h x w`; float32 levels."""
    hin, win = x.shape[-2:]
    if (hin, win) == (h, w):
        return x.float()
    v = x.to(torch.int32)
    if hin == _AREA_FAST * h and win == _AREA_FAST * w:
        v = v[..., 0::2, 0::2] + v[..., 0::2, 1::2] + v[..., 1::2, 0::2] + v[..., 1::2, 1::2]
        return ((v + 2) >> 2).float()
    taps = _taps.__wrapped__ if torch.compiler.is_compiling() else _taps  # dynamo cannot trace through lru_cache
    x0, x1, a0, a1 = (t.to(x.device) for t in taps(win, w, rows=False))
    y0, y1, b0, b1 = (t.to(x.device) for t in taps(hin, h, rows=True))
    r0, r1 = v[..., y0, :], v[..., y1, :]
    r0 = r0[..., x0] * a0 + r0[..., x1] * a1
    r1 = r1[..., x0] * a0 + r1[..., x1] * a1
    b0, b1 = b0[:, None], b1[:, None]
    return ((((b0 * (r0 >> 4)) >> 16) + ((b1 * (r1 >> 4)) >> 16) + 2) >> 2).float()
