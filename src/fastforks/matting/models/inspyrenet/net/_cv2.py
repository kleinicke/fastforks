"""The OpenCV / kornia calls of the vendored `modules/layers.py`, without OpenCV or kornia.

`getGaussianKernel` and `getStructuringElement(MORPH_ELLIPSE, ...)` compute what OpenCV's do (float64 `[k, 1]`
Gaussian for `sigma > 0`; OpenCV's ellipse rows). `dilation` / `erosion` (kornia) serve only `Transition`, the
pyramid blending of `threshold` inference, which transparent-background never runs (`threshold=None`) and fastforks
does not port: they raise.
"""

import math

import numpy as np

MORPH_ELLIPSE = 2


def getGaussianKernel(ksize, sigma):  # noqa: N802 - OpenCV's name
    x = np.arange(ksize, dtype=np.float64) - (ksize - 1) / 2
    k = np.exp(-(x**2) / (2 * sigma**2))
    return (k / k.sum())[:, None]


def getStructuringElement(shape, ksize):  # noqa: N802 - OpenCV's name
    if shape != MORPH_ELLIPSE:
        raise NotImplementedError(shape)
    w, h = ksize
    r, c = h // 2, w // 2
    inv_r2 = 1 / (r * r) if r else 0
    out = np.zeros((h, w), np.uint8)
    for i in range(h):
        dy = i - r
        if abs(dy) <= r:
            dx = round(c * math.sqrt((r * r - dy * dy) * inv_r2))
            out[i, max(c - dx, 0) : min(c + dx + 1, w)] = 1
    return out


def dilation(*_args, **_kwargs):
    msg = "InSPyReNet pyramid blending (threshold inference) is not ported"
    raise NotImplementedError(msg)


erosion = dilation
