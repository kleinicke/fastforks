"""Sort-free k-th smallest values for exported graphs (fastforks; TensorRT's TopK takes at most 3840 elements).

`kth_smallest` bisects on the value: the k-th smallest entry v_k is the smallest x with count(v <= x) > k. The
invariant count(v <= lo) <= k < count(v <= hi) holds from the start (lo below the minimum, hi the maximum), so hi ends
within range * 2^-64 of v_k; it is v_k exactly once lo and hi are adjacent floats, which 64 halvings reach unless
|v_k| is below ~1e-12 of the range (depth, disparity and confidence maps: exact).
"""

from __future__ import annotations

import math

import torch

STEPS = 64


def kth_smallest(
    v: torch.Tensor, k: int | torch.Tensor, valid: torch.Tensor | None = None, steps: int = STEPS
) -> torch.Tensor:
    """The k-th smallest (0-based) entry per row of `v` `[..., N]` among the `valid` ones (default all): `[...]`.

    `k` is an int or an integer tensor of shape `v.shape[:-1]` with 0 <= k < count(valid) per row (rows outside that
    get a value of no meaning). Same result as `torch.where(valid, v, inf).sort(-1).values[..., k]` for finite values;
    computed in float32 (counts are exact up to 2^24 entries per row), returned in `v`'s dtype.
    """
    x = v.float()
    big = torch.finfo(torch.float32).max
    if valid is None:
        lo, hi = x.amin(-1), x.amax(-1)
    else:
        lo, hi = torch.where(valid, x, big).amin(-1), torch.where(valid, x, -big).amax(-1)
    lo = lo - 1 - lo.abs()
    kf = k.float() if isinstance(k, torch.Tensor) else float(k)
    for _ in range(steps):
        mid = (lo + hi) * 0.5
        le = x <= mid[..., None]
        above = (le if valid is None else le & valid).float().sum(-1) > kf
        hi, lo = torch.where(above, mid, hi), torch.where(above, lo, mid)
    return hi.to(v.dtype)


def quantile(v: torch.Tensor, q: float) -> torch.Tensor:
    """`torch.quantile(v, q, dim=-1)` (linear interpolation) for finite `v` `[..., N]`, without a sort: `[...]`."""
    n = v.shape[-1]
    pos = q * (n - 1)
    i = min(math.floor(pos), n - 1)
    lo = kth_smallest(v, i).float()
    if pos == i:
        return lo.to(v.dtype)
    x = v.float()
    le = (x <= lo[..., None]).sum(-1)  # entries <= the i-th: the (i + 1)-th equals it when more than i + 1
    hi = torch.where(le > i + 1, lo, torch.where(x > lo[..., None], x, torch.inf).amin(-1))
    return torch.lerp(lo, hi, pos - i).to(v.dtype)
