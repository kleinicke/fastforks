"""OpenCV's `cv2.resize` of uint8 images (`INTER_CUBIC`, `INTER_AREA`) on the device, as upstream DA3 preprocesses.

Upstream resizes the uint8 image on the CPU with OpenCV; this reproduces it on any device as two separable tap
gathers with OpenCV's weights:

- `INTER_CUBIC`: Keys kernel a = -0.75, replicated borders
- `INTER_AREA` (shrinking): OpenCV's `computeResizeAreaTab` box-overlap weights

Horizontal pass first, rounded once at the end; cubic in float64 (float32 on MPS), area in float32 as OpenCV. Matches
OpenCV 5.0 up to a few rounding ties (one level); OpenCV 4 computes uint8 cubic in 11-bit fixed point, which differs by
one level on a few percent of the pixels.
"""

from __future__ import annotations

import functools
import math
from typing import Literal

import torch

_CUBIC_A = -0.75  # fixed: OpenCV's bicubic kernel parameter
_AREA_EPS = 1e-3  # fixed: OpenCV computeResizeAreaTab's threshold for partial source cells

Kind = Literal["cubic", "area"]


@functools.lru_cache(maxsize=64)
def _cubic(n_in: int, n_out: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Source indices `[n_out, 4]` (int64) and float64 weights of OpenCV's INTER_CUBIC along one axis."""
    scale = n_in / n_out
    a = _CUBIC_A
    idx: list[list[int]] = []
    wts: list[list[float]] = []
    for o in range(n_out):
        f = (o + 0.5) * scale - 0.5
        sx = math.floor(f)
        t = f - sx
        c = [
            ((a * (t + 1) - 5 * a) * (t + 1) + 8 * a) * (t + 1) - 4 * a,
            ((a + 2) * t - (a + 3)) * t * t + 1,
            ((a + 2) * (1 - t) - (a + 3)) * (1 - t) * (1 - t) + 1,
        ]
        idx.append([min(max(sx - 1 + k, 0), n_in - 1) for k in range(4)])
        wts.append([*c, 1 - sum(c)])
    return torch.tensor(idx), torch.tensor(wts, dtype=torch.float64)


@functools.lru_cache(maxsize=64)
def _area(n_in: int, n_out: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Source indices `[n_out, T]` (int64) and float32 weights of OpenCV's INTER_AREA (shrinking) along one axis."""
    scale = n_in / n_out
    taps: list[list[tuple[int, float]]] = []  # (source index, weight) per output
    for o in range(n_out):
        f1 = o * scale
        f2 = f1 + scale
        cell = min(scale, n_in - f1)
        s2 = min(math.floor(f2), n_in - 1)
        s1 = min(math.ceil(f1), s2)
        row = []
        if s1 - f1 > _AREA_EPS:
            row.append((s1 - 1, (s1 - f1) / cell))
        row += [(s, 1.0 / cell) for s in range(s1, s2)]
        if f2 - s2 > _AREA_EPS:
            row.append((s2, min(f2 - s2, 1.0, cell) / cell))
        taps.append(row)
    t = max(len(r) for r in taps)
    pad = [row + [(0, 0.0)] * (t - len(row)) for row in taps]
    idx = torch.tensor([[s for s, _ in row] for row in pad])
    return idx, torch.tensor([[w for _, w in row] for row in pad], dtype=torch.float32)  # OpenCV's float tables


def _gather(x: torch.Tensor, idx: torch.Tensor, wts: torch.Tensor) -> torch.Tensor:
    """Resample the last axis of `x`: `out[..., o] = sum_k x[..., idx[o, k]] * wts[o, k]`."""
    return (x[..., idx.to(x.device)] * wts.to(x.device, x.dtype)).sum(-1)


def cv2_resize(x: torch.Tensor, h: int, w: int, kind: Kind) -> torch.Tensor:
    """`cv2.resize` of `[B, C, H, W]` uint8 levels (any float / int dtype, values 0..255) to `h x w`; float32 levels.

    `kind` applies to both axes, as OpenCV's interpolation flag does.
    """
    taps, dtype = (_cubic, torch.float64) if kind == "cubic" else (_area, torch.float32)
    if x.device.type == "mps":  # no float64 there; 4 taps of levels <= 255 differ by ~1e-5, only exact ties round apart
        dtype = torch.float32
    if torch.compiler.is_compiling():  # dynamo cannot trace through lru_cache; the graph holds the tables anyway
        taps = taps.__wrapped__
    (ix, wx), (iy, wy) = taps(x.shape[-1], w), taps(x.shape[-2], h)
    v = _gather(_gather(x.to(dtype), ix, wx).transpose(-1, -2), iy, wy).transpose(-1, -2)
    return v.round_().clamp_(0, 255).float()
