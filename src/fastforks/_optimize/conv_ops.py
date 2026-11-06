# ruff: noqa: PLR0913, PLR0917 - the kernel argument lists
"""The `fastforks.kernels.fvm_conv` ops (fused conv, depthwise conv, pools, input packing, token GEMM `lin`).

Split from `ops` so that a kernels backend that only needs fvm_conv imports (and its card lists) only fvm_conv; `ops`
re-exports every name. Conventions as in `ops`: fp16 NCHW tensors in channels-last memory unless noted; `act` codes
0 none, 1 relu, 2 relu6, 3 silu, 4 hardswish, 5 gelu, 6 gelu (tanh).
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.kernels import current_stream, note_fallback
from fastforks.kernels import fvm_conv

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastforks.kernels.fvm_conv import FvmLibrary

_CL = torch.channels_last
_HALF = torch.float16
_MAX_DEVICES = 64

_LIBS: list[FvmLibrary | None] = [None] * _MAX_DEVICES


def _conv(t: torch.Tensor) -> FvmLibrary:
    d = t.get_device()
    lib = _LIBS[d]
    if lib is None:
        lib = _LIBS[d] = fvm_conv.load(t.device)
    return lib


def _check(rc: int, what: str) -> None:
    if rc:
        msg = f"{what} failed (rc {rc})"
        raise RuntimeError(msg)


def _p(t: torch.Tensor | None) -> int | None:
    return None if t is None else t.data_ptr()


def _empty(n: int, c: int, h: int, w: int, like: torch.Tensor) -> torch.Tensor:
    return torch.empty((n, c, h, w), dtype=_HALF, device=like.device, memory_format=_CL)


def _osz(length: int, k: int, s: int, p: int, *, ceil: bool = False) -> int:
    """PyTorch pooling output size (ceil mode: the last window must start inside the input or the left padding)."""
    o = (length + 2 * p - k + s - 1) // s + 1 if ceil else (length + 2 * p - k) // s + 1
    if ceil and (o - 1) * s >= length + p:
        o -= 1
    return o


def _pixel_ld(t: torch.Tensor) -> int:
    """Pixel stride of a channels-last `[N, C, H, W]` view with evenly spaced pixels (e.g. a channel slice), else 0.

    Size-1 dims may carry any stride (PyTorch keeps NCHW strides on 1x1 maps) and are skipped.
    """
    n, c, h, w = t.shape
    s0, s1, s2, s3 = t.stride()
    ld = s3 if w > 1 else s2 if h > 1 else s0 if n > 1 else -(-c // 8) * 8
    ok = ld >= c and (c == 1 or s1 == 1) and (h == 1 or s2 == w * ld) and (n == 1 or s0 == h * w * ld)
    return ld if ok else 0


def _cl_slice(t: torch.Tensor) -> int:
    """Pixel stride of a channels-last fp16 4-D tensor that may be a channel slice of a wider buffer; 0 if not one."""
    if t.dim() != 4 or t.dtype != _HALF:  # noqa: PLR2004
        return 0
    l3 = _pixel_ld(t)
    return l3 if l3 % 8 == 0 and t.data_ptr() % 16 == 0 else 0


def _out_ld(y: torch.Tensor, n: int, k: int, p: int, q: int, what: str) -> int:
    ldy = _pixel_ld(y) if tuple(y.shape) == (n, k, p, q) else 0
    if not (ldy and ldy % 8 == 0 and y.data_ptr() % 16 == 0 and y.dtype == _HALF):
        msg = f"{what}: out must be a 16-byte aligned channels-last fp16 channel slice of shape {(n, k, p, q)}"
        raise ValueError(msg)
    return ldy


# id(weight) -> (weakref to it, version, data_ptr, packed copy). Not a WeakKeyDictionary: its lookups compare keys
# with `==`, which for tensors is elementwise (a hit raised "Boolean value of Tensor ... is ambiguous").
_STEM_C8_W: dict[int, tuple[weakref.ref[torch.Tensor], int, int, torch.Tensor]] = {}


def _stem_c8_drop(key: int, ref: weakref.ref[torch.Tensor]) -> None:
    hit = _STEM_C8_W.get(key)
    if hit is not None and hit[0] is ref:  # a dead weight's entry, not a newer one that reuses the id
        del _STEM_C8_W[key]


def _stem_c8_weight(w: torch.Tensor) -> torch.Tensor:
    """[K, 8, R, S] -> [K/8][R][S][8 in][8 out] (cached per weight tensor, version and storage)."""
    key = id(w)
    hit = _STEM_C8_W.get(key)
    if hit is not None and hit[0]() is w and hit[1] == w._version and hit[2] == w.data_ptr():  # noqa: SLF001
        return hit[3]
    k, _, r, s = w.shape
    wt = w.contiguous().view(k // 8, 8, 8, r, s).permute(0, 3, 4, 2, 1).contiguous()
    ref = weakref.ref(w, lambda r, key=key: _stem_c8_drop(key, r))
    _STEM_C8_W[key] = (ref, w._version, w.data_ptr(), wt)  # noqa: SLF001
    return wt


def _conv_impl(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    res: torch.Tensor | None,
    padding: Sequence[int],
    stride: Sequence[int],
    act: int,
    cfg: int,
    out: torch.Tensor | None,
) -> torch.Tensor:
    ldx = _cl_slice(x)
    if not (ldx > x.shape[1] and x.shape[1] != 8):  # noqa: PLR2004 - a real slice (not 8 channels): read in place
        ldx = 0
        x = x.contiguous(memory_format=_CL)
    if not w.is_contiguous(memory_format=_CL):
        msg = "conv_fused: weight must be channels-last"
        raise ValueError(msg)
    if res is not None:
        res = res.contiguous(memory_format=_CL)
    n, c, h, wd = x.shape
    k, _, r, s = w.shape
    ph, pw = padding
    sh, sw = stride
    p, q = (h + 2 * ph - r) // sh + 1, (wd + 2 * pw - s) // sw + 1
    if out is not None:
        y, ldy = out, _out_ld(out, n, k, p, q, "conv_fused")
    else:
        y, ldy = _empty(n, k, p, q, x), k
    st = current_stream()
    if c == 8 and k <= 16 and res is None and out is None and cfg < 0:  # noqa: PLR2004 - direct 8-channel stem kernel
        wt = _stem_c8_weight(w)
        rc = _conv(x).stem_c8(
            x.data_ptr(), wt.data_ptr(), _p(bias), y.data_ptr(), n, h, wd, k, r, s, ph, pw, sh, sw, act, st
        )
        _check(rc, "fvm_conv stem_c8")
        return y
    if res is not None and res.shape != y.shape:
        msg = f"conv_fused: residual {tuple(res.shape)} vs output {tuple(y.shape)}"
        raise ValueError(msg)
    rc = _conv(x).conv_fused(
        cfg,
        x.data_ptr(),
        w.data_ptr(),
        _p(bias),
        _p(res),
        y.data_ptr(),
        n,
        h,
        wd,
        c,
        k,
        r,
        s,
        ph,
        pw,
        sh,
        sw,
        act,
        ldx,
        ldy,
        st,
    )
    _check(rc, f"fvm_conv conv_fused cfg {cfg} x {tuple(x.shape)} w {tuple(w.shape)}")
    return y


def conv_fused(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    res: torch.Tensor | None,
    padding: Sequence[int],
    stride: Sequence[int],
    act: int,
    cfg: int = -1,
) -> torch.Tensor:
    """act(conv(x, w) + bias (+ res)); w `[K, C, R, S]` channels-last; cfg -1 = tile config by the shape rule."""
    return _conv_impl(x, w, bias, res, padding, stride, act, cfg, None)


def conv_fused_into(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    res: torch.Tensor | None,
    padding: Sequence[int],
    stride: Sequence[int],
    act: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """`conv_fused` written into `out` (a channel slice of a channels-last buffer); returns `out`."""
    return _conv_impl(x, w, bias, res, padding, stride, act, -1, out)


def affine_act(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, act: int) -> torch.Tensor:
    """act(x * scale[c] + shift[c]) (fp32 scale / shift); x may be a channel slice; packed output."""
    n, c, h, w = x.shape
    ldx = _cl_slice(x)
    if not ldx:
        x = x.contiguous(memory_format=_CL)
        ldx = c
    if x.dtype != _HALF or c % 8 or scale.dtype != torch.float32:
        msg = "affine_act: fp16 x with C % 8 == 0 and fp32 scale / shift"
        raise ValueError(msg)
    y = _empty(n, c, h, w, x)
    rc = _conv(x).affine_act(
        x.data_ptr(), scale.data_ptr(), shift.data_ptr(), y.data_ptr(), n * h * w, c, ldx, act, current_stream()
    )
    _check(rc, "fvm_conv affine_act")
    return y


def maxpool(
    x: torch.Tensor,
    k: Sequence[int],
    stride: Sequence[int],
    pad: Sequence[int],
    ceil: bool = False,  # noqa: FBT001, FBT002
) -> torch.Tensor:
    """Max pool (PyTorch semantics, -inf padding, no dilation)."""
    if x.dtype != _HALF or x.shape[1] % 8:
        note_fallback("fvm_conv", "maxpool with C % 8 != 0 or not fp16")
        return F.max_pool2d(x, list(k), list(stride), list(pad), 1, ceil)
    x = x.contiguous(memory_format=_CL)
    n, c, h, w = x.shape
    p, q = _osz(h, k[0], stride[0], pad[0], ceil=ceil), _osz(w, k[1], stride[1], pad[1], ceil=ceil)
    y = _empty(n, c, p, q, x)
    rc = _conv(x).maxpool(
        x.data_ptr(), y.data_ptr(), n, h, w, c, p, q, k[0], k[1], stride[0], stride[1], pad[0], pad[1], current_stream()
    )
    _check(rc, "fvm_conv maxpool")
    return y


def avgpool(
    x: torch.Tensor,
    k: Sequence[int],
    stride: Sequence[int],
    pad: Sequence[int],
    count_include_pad: bool = True,  # noqa: FBT001, FBT002
) -> torch.Tensor:
    """Average pool (PyTorch semantics, no ceil mode)."""
    if x.dtype != _HALF or x.shape[1] % 8:
        note_fallback("fvm_conv", "avgpool with C % 8 != 0 or not fp16")
        return F.avg_pool2d(x, list(k), list(stride), list(pad), False, count_include_pad)  # noqa: FBT003
    x = x.contiguous(memory_format=_CL)
    n, c, h, w = x.shape
    p, q = (h + 2 * pad[0] - k[0]) // stride[0] + 1, (w + 2 * pad[1] - k[1]) // stride[1] + 1
    y = _empty(n, c, p, q, x)
    rc = _conv(x).avgpool(
        x.data_ptr(),
        y.data_ptr(),
        n,
        h,
        w,
        c,
        k[0],
        k[1],
        stride[0],
        stride[1],
        pad[0],
        pad[1],
        int(count_include_pad),
        current_stream(),
    )
    _check(rc, "fvm_conv avgpool")
    return y


def dwconv(
    x: torch.Tensor,
    w_rsc: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    act: int,
) -> torch.Tensor:
    """Depthwise conv, fp32 accumulation; w_rsc `[R, S, C]` contiguous."""
    x = x.contiguous(memory_format=_CL)
    if x.dtype != _HALF or x.shape[1] % 8 or not w_rsc.is_contiguous():
        msg = "dwconv: fp16 x with C % 8 == 0 and a contiguous [R, S, C] weight"
        raise ValueError(msg)
    n, c, h, w = x.shape
    r, s = w_rsc.shape[0], w_rsc.shape[1]
    p, q = (h + 2 * padding[0] - r) // stride[0] + 1, (w + 2 * padding[1] - s) // stride[1] + 1
    y = _empty(n, c, p, q, x)
    rc = _conv(x).dwconv(
        x.data_ptr(),
        w_rsc.data_ptr(),
        _p(bias),
        y.data_ptr(),
        n,
        h,
        w,
        c,
        r,
        s,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        act,
        current_stream(),
    )
    _check(rc, "fvm_conv dwconv")
    return y


def input_nhwc8(x: torch.Tensor) -> torch.Tensor:
    """fp32 / fp16 NCHW input (C <= 8) -> fp16 channels-last with the channels zero-padded to 8."""
    x = x.contiguous()
    n, c, h, w = x.shape
    if c > 8:  # noqa: PLR2004
        msg = "input_nhwc8: C <= 8"
        raise ValueError(msg)
    y = _empty(n, 8, h, w, x)
    rc = _conv(x).input_nhwc8(x.data_ptr(), y.data_ptr(), int(x.dtype == torch.float32), n, c, h * w, current_stream())
    _check(rc, "fvm_conv input_nhwc8")
    return y


# --- token GEMM -------------------------------------------------------------------------------------------------

LIN_MAX_ROWS = 2048  # residual GEMMs with fewer rows (batch * tokens) and ...
LIN_MIN_WORK = 1 << 29  # ... at least this many multiply-adds (rows * K * N) run cuBLAS + add


def lin(
    x: torch.Tensor, w4: torch.Tensor, b: torch.Tensor | None, res: torch.Tensor | None = None, act: int = 0
) -> torch.Tensor:
    """act(x @ w^T + b (+ res)) through the fused conv (1x1 conv over [1, K, M, 1], fp16 accumulate).

    Per-shape dispatch (documented on the cards): a GEMM with a residual, fewer than `LIN_MAX_ROWS` rows and at least
    `LIN_MIN_WORK` multiply-adds runs as `F.linear` + add. There the fused CUTLASS tile grid is too coarse for the GPU
    (e.g. SigLIP-so400m 378, 729 rows: attention projection 35.2 vs 38.8 us, MLP down-projection 97 vs 121 us on an
    RTX 5060 Ti) while the extra add launch is negligible. Smaller GEMMs are launch bound and the fused kernel saves a
    launch (DeiT-B 224: 2.88 vs 3.61 ms), larger ones favour the fused kernel (up to 1.7x from 4096 rows).
    """
    sh = x.shape
    k = sh[-1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    if res is not None and act == 0 and m < LIN_MAX_ROWS and m * k * w4.shape[0] >= LIN_MIN_WORK and x.is_cuda:
        # cuBLAS: faster here (coarse fused tile grid) and more accurate (fp32 accumulate); a dispatch, not a fallback
        y = F.linear(x, w4.reshape(w4.shape[0], k), b)
        return y + res.reshape(y.shape)
    r = None if res is None else res.reshape(m, -1).view(1, m, 1, -1).permute(0, 3, 1, 2)
    y = conv_fused(x2.view(1, m, 1, k).permute(0, 3, 1, 2), w4, b, r, [0, 0], [1, 1], act)
    return y.permute(0, 2, 3, 1).reshape(*sh[:-1], -1)
