"""Antialiased resizes as two matrix products in exported graphs (TensorRT has no antialiased `Resize`).

An antialiased bilinear / bicubic `F.interpolate` is linear and separable: at a fixed input and output size it is
`Wh @ x @ Ww.T` with two resampling matrices. `matmul_antialias()` makes `F.interpolate` emit that while a fixed-size
graph is traced; the matrices are built by torch's own antialiased kernel on identity bases (float64 on the CPU), so
the exported graph computes the same weights as the eager op. Only the exporting thread is affected, and only inside
the context: eager calls and the kernels backend keep `F.interpolate` as it is. Symbolic sizes (a dynamic export),
non-float inputs and other modes call the original function.
"""

from __future__ import annotations

import contextlib
import math
import threading
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Generator

_ORIGINAL = F.interpolate
_OPS = {"bilinear": "_upsample_bilinear2d_aa", "bicubic": "_upsample_bicubic2d_aa"}
_LOCK = threading.Lock()
_STATE = threading.local()
_USERS = 0
_CACHE: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = {}
_CACHE_SIZE = 16


def _pair(v: Any) -> tuple[Any, Any]:  # noqa: ANN401 - an int / float or a pair of them
    return (v[0], v[1]) if isinstance(v, (tuple, list)) else (v, v)


def resampling_matrices(  # noqa: PLR0913 - the arguments of the antialiased op
    in_hw: tuple[int, int],
    out_hw: tuple[int, int],
    mode: str,
    *,
    align_corners: bool,
    scales: tuple[float | None, float | None] = (None, None),
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(Wh [Ho, H], Ww_t [W, Wo])` with `interpolate(x, antialias=True) == Wh @ x @ Ww_t` (up to rounding).

    Built from torch's antialiased op on identity bases in float64: basis image c is row c of the identity, resized
    along W only (H has size 1, where the filters are the identity). The filter of an axis depends only on its sizes
    and scale, so the H weights come from the same W resize (torch 2.14's CPU kernel misreads `[1, C, H, 1]` inputs).
    """
    op = getattr(torch.ops.aten, _OPS[mode]).default
    (h, w), (ho, wo) = in_hw, out_hw

    def axis(n: int, m: int, scale: float | None) -> torch.Tensor:  # [n, m]: row c = the resized basis vector e_c
        return op(torch.eye(n, dtype=torch.float64)[:, None, None, :], [1, m], align_corners, None, scale)[:, 0, 0, :]

    with torch.inference_mode(mode=False), torch.no_grad():
        wh, ww_t = axis(h, ho, scales[0]).t(), axis(w, wo, scales[1])
        return wh.to(device, dtype).contiguous(), ww_t.to(device, dtype).contiguous()


def _matrices(key: tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    got = _CACHE.get(key)
    if got is None:
        from torch.utils._python_dispatch import _disable_current_modes  # noqa: PLC0415 - private, export only

        in_hw, out_hw, mode, align_corners, scales, device, dtype = key
        with _disable_current_modes():  # real constants, also while a fake / proxy mode traces the graph
            got = resampling_matrices(
                in_hw, out_hw, mode, align_corners=align_corners, scales=scales, device=device, dtype=dtype
            )
        if len(_CACHE) >= _CACHE_SIZE:
            del _CACHE[next(iter(_CACHE))]
        _CACHE[key] = got
    return got


def _out_size(x: torch.Tensor, size: Any, scale_factor: Any, *, recompute: bool | None) -> tuple[Any, ...] | None:  # noqa: ANN401
    """`((Ho, Wo), (scale_h, scale_w))` as `F.interpolate` computes them for a 4-D input, or None if not static ints."""
    h, w = x.shape[-2:]
    if size is not None:
        out, scales = _pair(size), (None, None)
    elif scale_factor is not None:
        sf = _pair(scale_factor)
        if not all(isinstance(s, (int, float)) for s in sf) or not (isinstance(h, int) and isinstance(w, int)):
            return None  # a symbolic size would be specialized by the float math below
        out = (math.floor(float(h * sf[0])), math.floor(float(w * sf[1])))
        scales = (None, None) if recompute else (float(sf[0]), float(sf[1]))
    else:
        return None
    if not all(isinstance(v, int) for v in (h, w, *out)):  # symbolic sizes (dynamic export): the original op
        return None
    return out, scales


def _interpolate(  # noqa: PLR0913, PLR0917 - F.interpolate's signature
    input: torch.Tensor,  # noqa: A002 - F.interpolate's argument name
    size: Any = None,  # noqa: ANN401
    scale_factor: Any = None,  # noqa: ANN401
    mode: str = "nearest",
    align_corners: bool | None = None,  # noqa: FBT001
    recompute_scale_factor: bool | None = None,  # noqa: FBT001
    antialias: bool = False,  # noqa: FBT001, FBT002 - F.interpolate's signature
) -> torch.Tensor:
    args = (input, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias)
    if not (antialias and getattr(_STATE, "on", False) and mode in _OPS and input.dim() == 4):  # noqa: PLR2004 - NCHW
        return _ORIGINAL(*args)
    if not input.is_floating_point():
        return _ORIGINAL(*args)
    sizes = _out_size(input, size, scale_factor, recompute=recompute_scale_factor)
    if sizes is None:
        return _ORIGINAL(*args)
    (ho, wo), scales = sizes
    h, w = input.shape[-2:]
    key = ((h, w), (ho, wo), mode, bool(align_corners), scales, input.device, input.dtype)
    wh, ww_t = _matrices(key)
    with torch.autocast(input.device.type, enabled=False):  # the input's dtype, as the antialiased op
        if ho * w * (h + wo) <= h * wo * (w + ho):  # cheaper order first (fewer multiply-adds)
            return torch.matmul(torch.matmul(wh, input), ww_t)
        return torch.matmul(wh, torch.matmul(input, ww_t))


@contextlib.contextmanager
def matmul_antialias() -> Generator[None]:
    """Inside the context, antialiased bilinear / bicubic `F.interpolate` calls of this thread trace as matmuls."""
    global _USERS  # noqa: PLW0603 - the patch is shared by concurrent exports
    with _LOCK:
        if _USERS == 0:
            F.interpolate = _interpolate  # ty: ignore[invalid-assignment] - same call signature, wider hints
        _USERS += 1
    prev = getattr(_STATE, "on", False)
    _STATE.on = True
    try:
        yield
    finally:
        _STATE.on = prev
        with _LOCK:
            _USERS -= 1
            if _USERS == 0:
                F.interpolate = _ORIGINAL
