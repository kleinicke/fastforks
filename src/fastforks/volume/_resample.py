"""nnU-Net v2's resampling and nonzero mask in PyTorch, on the volume's device.

Ported from nnunetv2 2.8.1 (Apache-2.0): `default_resampling.resample_data_or_seg_to_shape` (scikit-image
`resize`, i.e. `scipy.ndimage.zoom` with `grid_mode=True`, mode `nearest`, clipped to the input range; anisotropic
volumes in-plane per slice, then order `order_z` along the low-resolution axis),
`resample_torch.resample_torch_fornnunet` (PyTorch `interpolate`), `no_resampling.no_resampling_hack`, and
`cropping.create_nonzero_mask` (`binary_fill_holes`).

The zooms are separable linear maps, so each axis is one `[new, old]` matrix: the spline prefilter (scipy's 12-voxel
edge padding, then the cubic B-spline coefficients by a banded solve) times the interpolation weights, applied with a
matmul. The prefilter boundary of the padded signal decays as 0.268^12 (~1e-7) before it reaches the volume.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._core.errors import WeightsError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = ["RESAMPLERS", "fill_holes", "lowres_axis", "new_shape", "resample"]

ANISO_THRESHOLD = 3.0  # nnunetv2.configuration.ANISO_THRESHOLD
_PAD = 12  # scipy.ndimage._prepad_for_spline_filter
RESAMPLERS = ("resample_data_or_seg_to_shape", "resample_torch_fornnunet", "no_resampling_hack")


def new_shape(shape: Sequence[int], spacing: Sequence[float], target: Sequence[float]) -> tuple[int, ...]:
    """`compute_new_shape`: `round(old spacing / new spacing * size)` per axis."""
    return tuple(round(s / t * n) for s, t, n in zip(spacing, target, shape, strict=True))


def _lowres(spacing: Sequence[float]) -> int | None:
    big = max(spacing)
    axes = [i for i, s in enumerate(spacing) if big / s == 1]
    return axes[0] if len(axes) == 1 else None  # two or three equally coarse axes: not separate


def lowres_axis(current: Sequence[float], new: Sequence[float], force: bool | None = None) -> int | None:  # noqa: FBT001
    """`determine_do_sep_z_and_axis`: the axis resampled separately (order `order_z`), or `None`."""
    if force is not None:
        return _lowres(current) if force else None
    for sp in (current, new):
        if max(sp) / min(sp) > ANISO_THRESHOLD:
            return _lowres(sp)
    return None


def _cubic(t: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Cubic B-spline weights of the coefficients at `floor(x) - 1 .. floor(x) + 2` (`t = x - floor(x)`)."""
    t2, t3 = t * t, t * t * t
    return (1 - t) ** 3 / 6, (3 * t3 - 6 * t2 + 4) / 6, (-3 * t3 + 3 * t2 + 3 * t + 1) / 6, t3 / 6


@lru_cache(maxsize=64)
def _matrix(n: int, m: int, order: int) -> torch.Tensor:
    """`[m, n]` float64 map of `scipy.ndimage.zoom(order, mode='nearest', grid_mode=True)` along one axis."""
    x = (torch.arange(m, dtype=torch.float64) + 0.5) * (n / m) - 0.5
    rows = torch.arange(m)
    if order == 0:
        out = torch.zeros((m, n), dtype=torch.float64)
        out[rows, torch.floor(x + 0.5).long().clamp_(0, n - 1)] = 1.0
        return out
    if order == 1:
        out = torch.zeros((m, n), dtype=torch.float64)
        x = x.clamp(0, n - 1)
        i0 = torch.floor(x).long().clamp_(max=max(n - 2, 0))
        t = x - i0
        out.index_put_((rows, i0), 1 - t, accumulate=True)
        out.index_put_((rows, (i0 + 1).clamp_(max=n - 1)), t, accumulate=True)
        return out
    if order != 3:  # noqa: PLR2004
        msg = f"nnU-Net resampling of order {order}: only orders 0, 1 and 3 are implemented"
        raise WeightsError(msg)
    big = n + 2 * _PAD
    edge = torch.zeros((big, n), dtype=torch.float64)
    edge[torch.arange(big), (torch.arange(big) - _PAD).clamp_(0, n - 1)] = 1.0
    band = torch.diag(torch.full((big,), 4 / 6, dtype=torch.float64))
    band += torch.diag(torch.full((big - 1,), 1 / 6, dtype=torch.float64), 1)
    band += torch.diag(torch.full((big - 1,), 1 / 6, dtype=torch.float64), -1)
    band[0, 1] = band[-1, -2] = 2 / 6  # mirror boundary of the padded signal
    coef = torch.linalg.solve(band, edge)  # [big, n]: spline coefficients of the edge-padded signal
    xp = x + _PAD
    i = torch.floor(xp).long()
    weights = torch.zeros((m, big), dtype=torch.float64)
    for k, w in enumerate(_cubic(xp - i)):
        weights.index_put_((rows, (i + k - 1).clamp_(0, big - 1)), w, accumulate=True)
    return weights @ coef


def _along(x: torch.Tensor, dim: int, size: int, order: int) -> torch.Tensor:
    """Resample `x` along `dim` to `size` (one axis of `scipy.ndimage.zoom`), in float32."""
    n = x.shape[dim]
    if n == size:
        return x
    m = _matrix(n, size, order).to(x.device, torch.float32)
    return torch.movedim(torch.movedim(x, dim, -1) @ m.T, -1, dim)


def _zoom(x: torch.Tensor, dims: Sequence[int], shape: Sequence[int], order: int) -> torch.Tensor:
    """Skimage `resize(order, mode='edge', anti_aliasing=False)` of `x` over `dims`, clipped per remaining index."""
    lo = x.amin(dim=tuple(dims), keepdim=True)
    hi = x.amax(dim=tuple(dims), keepdim=True)
    for d, s in zip(dims, shape, strict=True):
        x = _along(x, d, s, order)
    return torch.maximum(torch.minimum(x, hi), lo)


def _default(
    x: torch.Tensor, shape: Sequence[int], current: Sequence[float], new: Sequence[float], kw: Mapping[str, Any]
) -> torch.Tensor:
    order, order_z = int(kw.get("order", 3)), int(kw.get("order_z", 0))
    axis = lowres_axis(current, new, kw.get("force_separate_z"))
    if axis is None:
        return _zoom(x, (1, 2, 3), shape, order)
    plane = [d for d in (1, 2, 3) if d != axis + 1]
    x = _zoom(x, plane, [shape[d - 1] for d in plane], order)  # per slice: clipped to each slice's range
    return _along(x, axis + 1, shape[axis], order_z)  # scipy map_coordinates: no clipping


def _torch(
    x: torch.Tensor, shape: Sequence[int], current: Sequence[float], new: Sequence[float], kw: Mapping[str, Any]
) -> torch.Tensor:
    axis = lowres_axis(current, new, kw.get("force_separate_z"))
    if axis is None:
        return F.interpolate(x[None].float(), size=tuple(shape), mode="trilinear", antialias=False)[0]
    plane = [d for d in (0, 1, 2) if d != axis]
    y = torch.movedim(x, axis + 1, 1).flatten(0, 1)  # (c, axis) -> one channel per slice
    y = F.interpolate(y[None].float(), size=[shape[d] for d in plane], mode="bilinear", antialias=False)[0]
    y = torch.movedim(y.unflatten(0, (x.shape[0], x.shape[axis + 1])), 1, axis + 1)
    return F.interpolate(y[None], size=tuple(shape), mode="nearest-exact")[0]


def resample(  # noqa: PLR0913, PLR0917 - nnU-Net's resampling-function signature
    x: torch.Tensor,
    shape: Sequence[int],
    current: Sequence[float],
    new: Sequence[float],
    fn: str = "resample_data_or_seg_to_shape",
    kwargs: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """nnU-Net's `resampling_fn_data` / `resampling_fn_probabilities` of float data `[C, *shape]`.

    Raises:
        WeightsError: An unknown resampling function, or a segmentation resampling (`is_seg`).
    """
    kw = dict(kwargs or {})
    if kw.get("is_seg"):
        msg = "nnU-Net segmentation resampling is not used at inference"
        raise WeightsError(msg)
    if fn == "no_resampling_hack" or tuple(x.shape[1:]) == tuple(shape):
        return x
    if fn == "resample_data_or_seg_to_shape":
        return _default(x.float(), shape, current, new, kw)
    if fn == "resample_torch_fornnunet":
        return _torch(x.float(), shape, current, new, kw)
    msg = f"nnU-Net resampling function {fn!r} is not supported ({', '.join(RESAMPLERS)})"
    raise WeightsError(msg)


def fill_holes(mask: torch.Tensor, check: int = 8) -> torch.Tensor:
    """`scipy.ndimage.binary_fill_holes` (6-connected): background not connected to the border becomes foreground."""
    out = ~mask
    reach = torch.zeros_like(mask)
    for d in range(mask.dim()):
        for i in (0, -1):
            idx = tuple(i if a == d else slice(None) for a in range(mask.dim()))
            reach[idx] = out[idx]
    done = False
    while not done:
        before = int(reach.sum())
        for _ in range(check):
            grow = reach.clone()
            for d in range(mask.dim()):
                n = mask.shape[d]
                if n > 1:
                    grow.narrow(d, 1, n - 1).logical_or_(reach.narrow(d, 0, n - 1))
                    grow.narrow(d, 0, n - 1).logical_or_(reach.narrow(d, 1, n - 1))
            reach = grow.logical_and_(out)
        done = int(reach.sum()) == before
    return ~reach
