# ruff: noqa: PLR0913, PLR0917 - the kernel argument lists
"""Tensor-level ops on the fast-vision-models kernel libraries (`fastforks.kernels.fvm_*`), for the kernels passes.

Each op does what the fast-vision-models torch host function did: shape / layout checks, the channel-slice detection
(a channel slice of a channels-last buffer is read or written in place), output allocation, and the same routing to
the plain-PyTorch op (or another kernel) for the cases the kernel does not cover; numerics are those of the
fast-vision-models `cuda-kernels` tier. Tensors are fp16 NCHW in channels-last memory unless noted; `act` codes:
0 none, 1 relu, 2 relu6, 3 silu, 4 hardswish, 5 gelu, 6 gelu (tanh).

The ops are plain Python functions (fx `call_function` targets of the rewritten graphs).
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import torch

from fastforks._core.kernels import current_stream, note_fallback
from fastforks.kernels import fvm_conv, fvm_extra, fvm_mb, fvm_r3

from .conv_ops import (  # the fvm_conv ops (re-exported: `ops.conv_fused` etc.)
    _CL,
    _HALF,
    _MAX_DEVICES,
    _check,
    _cl_slice,
    _empty,
    _osz,
    _out_ld,
    _p,
    affine_act,  # noqa: F401 - re-exported
    avgpool,  # noqa: F401 - re-exported
    conv_fused,
    conv_fused_into,
    dwconv,
    input_nhwc8,  # noqa: F401 - re-exported
    maxpool,  # noqa: F401 - re-exported
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastforks.kernels.fvm_conv import FvmLibrary


# cuBLASLt workspace of the 1x1 conv (`pw`): PyTorch 2.14's default (`getCUDABlasLtWorkspaceSize`), which the
# fast-vision-models op used; the algorithm heuristic depends on it.
PW_WORKSPACE = 1 << 20
PW_MIN_MACS = 150_000_000  # cuBLASLt 1x1 conv only from this many multiply-adds (smaller: CUTLASS conv)
PW_MAX_NARROW = 256  # ... and only if min(C, K) <= this
TC_MAX_C = 48  # small-C KxK convs on the gconv2 kernel (cat2) up to this many input channels
STEM_K = (16, 24, 32, 40, 48, 64)  # output channels of the direct stem kernels


_LIBS: dict[str, list[FvmLibrary | None]] = {n: [None] * _MAX_DEVICES for n in ("r3", "mb", "extra")}
_MODULES = {"r3": fvm_r3, "mb": fvm_mb, "extra": fvm_extra}


def _lib(name: str, t: torch.Tensor) -> FvmLibrary:
    d = t.get_device()
    cache = _LIBS[name]
    lib = cache[d]
    if lib is None:
        lib = cache[d] = _MODULES[name].load(t.device)
    return lib


def libraries(device: torch.device) -> list[str]:
    """Load all four libraries on `device`; returns their tags.

    Raises:
        KernelUnavailableError: A library has no usable binary for this GPU / driver.
    """
    return [m.load(device).tag for m in (fvm_conv, fvm_r3, fvm_mb, fvm_extra)]


# --- fvm_r3 -----------------------------------------------------------------------------------------------------


def gconv2(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    pad: Sequence[int],
    groups: int,
    act: int,
    cfg: int = -1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Grouped (or, groups 1, dense small-C) conv on tensor cores; w `[K, GW, R, S]` channels-last."""
    x = x.contiguous(memory_format=_CL)
    if not w.is_contiguous(memory_format=_CL) or x.dtype != _HALF:
        msg = "gconv2: fp16 x and a channels-last weight"
        raise ValueError(msg)
    n, c, h, wd = x.shape
    k, gw, r, s = w.shape
    kg = k // groups
    if not (c == gw * groups and k % groups == 0 and gw % 16 == 0 and kg % 16 == 0):
        msg = f"gconv2: group shape C {c} K {k} GW {gw} groups {groups}"
        raise ValueError(msg)
    p, q = (h + 2 * pad[0] - r) // stride[0] + 1, (wd + 2 * pad[1] - s) // stride[1] + 1
    if out is not None:
        y, ldy = out, _out_ld(out, n, k, p, q, "gconv2")
    else:
        y, ldy = _empty(n, k, p, q, x), k
    rc = _lib("r3", x).gconv2(
        x.data_ptr(),
        w.data_ptr(),
        _p(bias),
        y.data_ptr(),
        n,
        h,
        wd,
        c,
        k,
        gw,
        r,
        s,
        stride[0],
        stride[1],
        pad[0],
        pad[1],
        act,
        ldy,
        cfg,
        current_stream(),
    )
    _check(rc, "fvm_r3 gconv2")
    return y


def conv_tc(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    pad: Sequence[int],
    act: int,
    cfg: int = -1,
) -> torch.Tensor:
    """Dense conv (C % 16 == 0, K % 16 == 0) on the gconv2 kernel."""
    return gconv2(x, w, bias, stride, pad, 1, act, cfg)


def conv_tc_into(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    pad: Sequence[int],
    act: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """`conv_tc` written into the channel slice `out`; returns `out`."""
    return gconv2(x, w, bias, stride, pad, 1, act, -1, out)


def se_scale(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """X * sigmoid(s) with s `[N, C, 1, 1]` (squeeze-excite gate)."""
    n, c, h, w = x.shape
    if c % 8 or x.dtype != _HALF or s.numel() != n * c or x.dim() != 4:  # noqa: PLR2004
        return x * torch.sigmoid(s)
    x = x.contiguous(memory_format=_CL)
    s = s.reshape(n, c).contiguous()
    y = torch.empty_like(x, memory_format=_CL)
    rc = _lib("r3", x).se_scale(x.data_ptr(), s.data_ptr(), y.data_ptr(), n, h * w, c, current_stream())
    _check(rc, "fvm_r3 se_scale")
    return y


def dwconv2(
    x: torch.Tensor,
    w_rsc: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    act: int,
) -> torch.Tensor:
    """Depthwise conv with half2 accumulation per kernel row (3x3 / 5x5 / 7x7; other shapes: `dwconv`)."""
    x = x.contiguous(memory_format=_CL)
    if x.dtype != _HALF or x.shape[1] % 8 or not w_rsc.is_contiguous():
        msg = "dwconv2: fp16 x with C % 8 == 0 and a contiguous [R, S, C] weight"
        raise ValueError(msg)
    r, s = w_rsc.shape[0], w_rsc.shape[1]
    if r > 7 or s > 7:  # noqa: PLR2004
        msg = "dwconv2: taps <= 7"
        raise ValueError(msg)
    if not (r == s and r in {3, 5, 7}):
        return dwconv(x, w_rsc, bias, stride, padding, act)
    n, c, h, w = x.shape
    p, q = (h + 2 * padding[0] - r) // stride[0] + 1, (w + 2 * padding[1] - s) // stride[1] + 1
    y = _empty(n, c, p, q, x)
    rc = _lib("r3", x).dwconv2(
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
    _check(rc, "fvm_r3 dwconv2")
    return y


def stem_pool(
    x: torch.Tensor,
    w_t: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    pad: Sequence[int],
    act: int,
    pk: int,
    ps: int,
    pp: int,
    ceil: bool,  # noqa: FBT001
) -> torch.Tensor:
    """Stem conv + act + max pool in one kernel, from the raw fp32 / fp16 NCHW input (C <= 4); w_t `[R, S, C, K]`."""
    x = x.contiguous()
    n, c, h, w = x.shape
    r, s, _, k = w_t.shape
    if w_t.shape[2] != c or c > 4 or not w_t.is_contiguous() or w_t.dtype != _HALF:  # noqa: PLR2004
        msg = "stem_pool: weight [R, S, C <= 4, K] contiguous fp16"
        raise ValueError(msg)
    p, q = (h + 2 * pad[0] - r) // stride[0] + 1, (w + 2 * pad[1] - s) // stride[1] + 1
    p2, q2 = _osz(p, pk, ps, pp, ceil=ceil), _osz(q, pk, ps, pp, ceil=ceil)
    y = _empty(n, k, p2, q2, x)
    rc = _lib("r3", x).stem_pool(
        x.data_ptr(), int(x.dtype == torch.float32), w_t.data_ptr(), _p(bias), y.data_ptr(), n, c, h, w, k, r, s,
        stride[0], stride[1], pad[0], pad[1], act, pk, ps, pp, p2, q2, current_stream(),
    )  # fmt: skip
    _check(rc, f"fvm_r3 stem_pool (K {k})")
    return y


_PW_WS: dict[tuple[int, int], torch.Tensor] = {}
# fvm_r3 1.0 (build f54eb890e23c) writes the bias pointer into the cached plan's matmul descriptor and then runs it;
# ctypes drops the GIL, so two threads on one plan could run with each other's bias. Held around the host-side call
# only (the GEMM itself stays asynchronous on the stream).
_PW_LOCK = threading.Lock()


def _pw_impl(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    res: torch.Tensor | None,
    act: int,
    out: torch.Tensor | None,
    force: int,
) -> torch.Tensor:
    n, c, h, wd = x.shape
    k = w.shape[0]
    m = n * h * wd
    ldx = _cl_slice(x)
    use = (
        act in {0, 1}
        and c % 8 == 0
        and k % 8 == 0
        and ldx > 0
        and (force > 0 or (force == 0 and res is None and min(c, k) <= PW_MAX_NARROW and m * k * c >= PW_MIN_MACS))
    )
    ldr = 0
    if use and res is not None:
        ldr = _cl_slice(res) if res.is_contiguous(memory_format=_CL) else 0
        use = ldr > 0
    ldy = 0
    if use and out is not None:
        ldy = _cl_slice(out)
        use = ldy > 0
    if use:
        y = out if out is not None else _empty(n, k, h, wd, x)
        if out is None:
            ldy = k
        st = current_stream()
        key = (x.get_device(), st)
        ws = _PW_WS.get(key)
        if ws is None:
            ws = _PW_WS[key] = torch.empty(PW_WORKSPACE, dtype=torch.uint8, device=x.device)
        with _PW_LOCK:  # fvm_pw sets the bias pointer on a cached, shared cuBLASLt plan before its matmul
            rc = _lib("r3", x).pw(
                x.data_ptr(), w.data_ptr(), _p(bias), _p(res), y.data_ptr(), m, k, c, ldx, ldy,
                ldr if res is not None else 0, act, ws.data_ptr(), PW_WORKSPACE, st,
            )  # fmt: skip
        if rc == 0:
            return y
        if rc != 1:
            msg = f"fvm_r3 pw: cublasLtMatmul failed (status {rc - 1000})"
            raise RuntimeError(msg)
        note_fallback("fvm_r3", "pw without a cuBLASLt plan")
    if out is not None:
        return conv_fused_into(x, w, bias, res, (0, 0), (1, 1), act, out)
    return conv_fused(x, w, bias, res, (0, 0), (1, 1), act)


def pw(
    x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None, res: torch.Tensor | None, act: int, force: int = 0
) -> torch.Tensor:
    """1x1 conv: cuBLASLt with fused bias / relu for big narrow GEMMs (fixed size rule), else `conv_fused`."""
    return _pw_impl(x, w, bias, res, act, None, force)


def pw_into(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    res: torch.Tensor | None,
    act: int,
    out: torch.Tensor,
    force: int = 0,
) -> torch.Tensor:
    """`pw` written into the channel slice `out`; returns `out`."""
    return _pw_impl(x, w, bias, res, act, out, force)


def cat2(
    x: torch.Tensor,
    wm: torch.Tensor,
    bm: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w3: torch.Tensor,
    b3: torch.Tensor | None,
    act: int,
    mmax: int,
) -> torch.Tensor:
    """cat([act(conv1x1(x)), act(convKxK(x))], 1) as one merged or two buffer-sharing convs.

    One merged KxK conv (`wm`, 1x1 zero-padded) up to `mmax` output pixels, else both convs into one buffer.
    """
    m = x.shape[0] * x.shape[2] * x.shape[3]
    r, c = w3.shape[2], x.shape[1]
    pd, one = (r // 2, r // 2), (1, 1)
    tc = c % 16 == 0 and c <= TC_MAX_C
    if m <= mmax:
        if tc and wm.shape[0] % 16 == 0:
            return gconv2(x, wm, bm, one, pd, 1, act)
        return conv_fused(x, wm, bm, None, pd, one, act)
    k1, k3 = w1.shape[0], w3.shape[0]
    y = _empty(x.shape[0], k1 + k3, x.shape[2], x.shape[3], x)
    _pw_impl(x, w1, b1, None, act, y.narrow(1, 0, k1), 0)
    if tc and k3 % 16 == 0:
        gconv2(x, w3, b3, one, pd, 1, act, -1, y.narrow(1, k1, k3))
    else:
        conv_fused_into(x, w3, b3, None, pd, one, act, y.narrow(1, k1, k3))
    return y


# --- fvm_mb -----------------------------------------------------------------------------------------------------


def stem_direct(
    x: torch.Tensor, w_t: torch.Tensor, bias: torch.Tensor | None, stride: Sequence[int], pad: Sequence[int], act: int
) -> torch.Tensor:
    """Stem conv straight from the raw fp32 / fp16 NCHW input (C <= 4); w_t `[R, S, C, K]`, K in `STEM_K`."""
    x = x.contiguous()
    n, c, h, w = x.shape
    r, s, _, k = w_t.shape
    if w_t.shape[2] != c or c > 4 or not w_t.is_contiguous() or w_t.dtype != _HALF:  # noqa: PLR2004
        msg = "stem_direct: weight [R, S, C <= 4, K] contiguous fp16"
        raise ValueError(msg)
    p, q = (h + 2 * pad[0] - r) // stride[0] + 1, (w + 2 * pad[1] - s) // stride[1] + 1
    y = _empty(n, k, p, q, x)
    rc = _lib("mb", x).stem_direct(
        x.data_ptr(), int(x.dtype == torch.float32), w_t.data_ptr(), _p(bias), y.data_ptr(), n, c, h, w, k, r, s,
        stride[0], stride[1], pad[0], pad[1], act, current_stream(),
    )  # fmt: skip
    _check(rc, f"fvm_mb stem_direct (K {k})")
    return y


def linattn2(xa: torch.Tensor, xb: torch.Tensor, eps: float) -> torch.Tensor:
    """EfficientViT ReLU linear attention (head dim 32) over the heads of xa `[B, HA*96, H, W]` then xb."""
    ok = (
        xa.dtype == _HALF
        and xa.is_contiguous(memory_format=_CL)
        and xa.shape[1] % 96 == 0
        and xb.is_contiguous(memory_format=_CL)
        and xb.shape[1] % 96 == 0
        and xa.shape[2:] == xb.shape[2:]
    )
    if not ok:
        msg = "linattn2: fp16 channels-last [B, heads*96, H, W] x2"
        raise ValueError(msg)
    b, _, h, w = xa.shape
    ha, hb = xa.shape[1] // 96, xb.shape[1] // 96
    y = _empty(b, (ha + hb) * 32, h, w, xa)
    rc = _lib("mb", xa).linattn2(xa.data_ptr(), xb.data_ptr(), y.data_ptr(), b, h * w, ha, hb, eps, current_stream())
    _check(rc, "fvm_mb linattn2")
    return y


def chan_gather(x: torch.Tensor, out: torch.Tensor, gsrc: torch.Tensor, gdst: torch.Tensor) -> torch.Tensor:
    """out[p, gdst[b] + k] = x[p, gsrc[8 b + k]] (-1: 0) for every pixel; returns `out`."""
    ok = (
        x.is_contiguous(memory_format=_CL)
        and out.is_contiguous(memory_format=_CL)
        and x.shape[0] == out.shape[0]
        and x.shape[2:] == out.shape[2:]
        and gsrc.numel() == 8 * gdst.numel()
    )
    if not ok:
        msg = "chan_gather: channels-last x / out of the same size, 8 sources per destination block"
        raise ValueError(msg)
    npix = x.shape[0] * x.shape[2] * x.shape[3]
    rc = _lib("mb", x).chan_gather(
        x.data_ptr(),
        out.data_ptr(),
        gsrc.data_ptr(),
        gdst.data_ptr(),
        npix,
        gdst.numel(),
        x.shape[1],
        out.shape[1],
        current_stream(),
    )
    _check(rc, "fvm_mb chan_gather")
    return out


XDW_MAX_TAPS = 8  # fvm_mb xdw_kernel: the generic depthwise loops stop at 8 taps per axis (fvm_mb.cu)


def xdw(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    wd: torch.Tensor,
    bd: torch.Tensor,
    e: int,
    stride: Sequence[int],
    pad: Sequence[int],
    acts: Sequence[int],
    tile: int = 0,
) -> torch.Tensor:
    """act2(dw(act1(conv1x1(x, w1) + b1)) + bd) in one kernel; w1 `[Ep, Cin_p]`, wd `[R, S, Ep]`; e real channels."""
    x = x.contiguous(memory_format=_CL)
    n, cin, h, w = x.shape
    r, s, ep = wd.shape
    cin_p = w1.shape[1]
    ok = (
        x.dtype == _HALF
        and cin % 8 == 0
        and cin_p % 16 == 0
        and ep % 32 == 0
        and e % 8 == 0
        and e <= ep
        and w1.shape[0] == ep
        and r <= XDW_MAX_TAPS
        and s <= XDW_MAX_TAPS
    )
    if not ok:
        msg = "xdw: layout"
        raise ValueError(msg)
    p, q = (h + 2 * pad[0] - r) // stride[0] + 1, (w + 2 * pad[1] - s) // stride[1] + 1
    y = _empty(n, e, p, q, x)
    rc = _lib("mb", x).xdw(
        x.data_ptr(), w1.data_ptr(), b1.data_ptr(), wd.data_ptr(), bd.data_ptr(), y.data_ptr(), n, h, w, cin, cin_p,
        ep, e, r, s, stride[0], stride[1], pad[0], pad[1], acts[0], acts[1], tile, current_stream(),
    )  # fmt: skip
    _check(rc, "fvm_mb xdw")
    return y


def dwmult(
    x: torch.Tensor, w_rsk: torch.Tensor, bias: torch.Tensor | None, stride: Sequence[int], pad: Sequence[int], act: int
) -> torch.Tensor:
    """Depthwise conv with a channel multiplier (K = C * mult); w_rsk `[R, S, K]`."""
    x = x.contiguous(memory_format=_CL)
    n, c, h, w = x.shape
    r, s, k = w_rsk.shape
    if k % c or k % 8 or x.dtype != _HALF:
        msg = "dwmult: K % C == 0, K % 8 == 0, fp16"
        raise ValueError(msg)
    p, q = (h + 2 * pad[0] - r) // stride[0] + 1, (w + 2 * pad[1] - s) // stride[1] + 1
    y = _empty(n, k, p, q, x)
    rc = _lib("mb", x).dwmult(
        x.data_ptr(),
        w_rsk.data_ptr(),
        _p(bias),
        y.data_ptr(),
        n,
        h,
        w,
        c,
        k,
        r,
        s,
        stride[0],
        stride[1],
        pad[0],
        pad[1],
        act,
        current_stream(),
    )
    _check(rc, "fvm_mb dwmult")
    return y


def gconv(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    pad: Sequence[int],
    groups: int,
    act: int,
) -> torch.Tensor:
    """Grouped conv (K / groups == GW, GW % 16 == 0), fp16 accumulation; w `[K, GW, R, S]` channels-last."""
    x = x.contiguous(memory_format=_CL)
    if not w.is_contiguous(memory_format=_CL) or x.dtype != _HALF:
        msg = "gconv: fp16 x and a channels-last weight"
        raise ValueError(msg)
    n, c, h, wd = x.shape
    k, gw, r, s = w.shape
    if not (c == gw * groups and k % groups == 0 and k // groups == gw and gw % 16 == 0):
        msg = f"gconv: group shape C {c} K {k} GW {gw} groups {groups}"
        raise ValueError(msg)
    p, q = (h + 2 * pad[0] - r) // stride[0] + 1, (wd + 2 * pad[1] - s) // stride[1] + 1
    y = _empty(n, k, p, q, x)
    rc = _lib("mb", x).gconv(
        x.data_ptr(),
        w.data_ptr(),
        _p(bias),
        y.data_ptr(),
        n,
        h,
        wd,
        c,
        k,
        gw,
        r,
        s,
        stride[0],
        stride[1],
        pad[0],
        pad[1],
        act,
        current_stream(),
    )
    _check(rc, "fvm_mb gconv")
    return y


def mbconv_into(
    x: torch.Tensor,
    w1: torch.Tensor | None,
    b1: torch.Tensor | None,
    wd: torch.Tensor,
    bd: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    stride: Sequence[int],
    pad: Sequence[int],
    acts: Sequence[int],
    cin_off: int,
    cin: int,
    out: torch.Tensor,
    colmap: torch.Tensor | None,
    gsrc: torch.Tensor | None,
    gdst: torch.Tensor | None,
) -> torch.Tensor:
    """Fused inverted-residual block written into `out`; returns `out`.

    ShuffleNetV2 units: input channel slice, output column map and passthrough gather;
    w1 `[Ep, Cin_p]` or None, wd `[R, S, Ep]`, w2 `[Cout, Ep]`.
    """
    x = x.contiguous(memory_format=_CL)
    if x.dtype != _HALF:
        msg = "mbconv: fp16 x"
        raise ValueError(msg)
    n, ldx, h, w = x.shape
    c_in = cin if cin > 0 else ldx - cin_off
    r, s, ldwd = wd.shape
    ep = ldwd
    gproj = w2.dim() == 3  # noqa: PLR2004
    cout = ep if gproj else w2.shape[0]
    p, q = (h + 2 * pad[0] - r) // stride[0] + 1, (w + 2 * pad[1] - s) // stride[1] + 1
    cin_p = w1.shape[1] if w1 is not None else c_in
    ok = (
        cin_off % 8 == 0
        and cin_off + c_in <= ldx
        and ldx % 8 == 0
        and c_in % 8 == 0
        and cout % 8 == 0
        and ep % 32 == 0
        and (w2.shape == (ep // 32, 32, 32) if gproj else w2.shape[1] == ep)
        and (w1 is None or cin_p % 16 == 0)
        and (w1 is not None or ep >= c_in)
        and (w1 is None or b1 is not None)
    )
    out_ok = (
        out.is_contiguous(memory_format=_CL)
        and out.shape[0] == n
        and out.shape[2:] == (p, q)
        and out.shape[1] % 8 == 0
        and out.dtype == _HALF
    )
    if not ok or not out_ok:
        msg = "mbconv: channel layout / out buffer"
        raise ValueError(msg)
    ldy = out.shape[1]
    if colmap is None and cout > ldy:
        msg = "mbconv: Cout > out channels"
        raise ValueError(msg)
    if colmap is not None and (colmap.numel() != cout // 8 or colmap.dtype != torch.int32):
        msg = "mbconv: colmap [Cout / 8] int32"
        raise ValueError(msg)
    g = 0
    if gsrc is not None:
        if not (
            gdst is not None
            and gsrc.numel() % 8 == 0
            and gdst.numel() * 8 == gsrc.numel()
            and tuple(stride) == (1, 1)
            and (p, q) == (h, w)
        ):
            msg = "mbconv: gather"
            raise ValueError(msg)
        g = gsrc.numel()
    rc = _lib("mb", x).mbconv(
        x.data_ptr(), _p(w1), _p(b1), wd.data_ptr(), bd.data_ptr(), w2.data_ptr(), _p(b2), None, out.data_ptr(),
        _p(colmap), _p(gsrc), _p(gdst), g, n, h, w, ldx, cin_off, c_in, cin_p, r, s, ldwd, ep, cout, int(gproj), ldy,
        stride[0], stride[1], pad[0], pad[1], acts[0], acts[1], acts[2], 0, 0, current_stream(),
    )  # fmt: skip
    _check(rc, "fvm_mb mbconv")
    return out


# --- fvm_extra --------------------------------------------------------------------------------------------------


def rope_qk(qkv: torch.Tensor, emb: torch.Tensor, npt: int, heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Qkv `[B, N, 3*H*D]` -> q, k `[B, N, H, D]` with the rotary embedding applied.

    `emb` `[N - npt, 2*D]` (timm interleaved pairs) on all but the first `npt` tokens.
    """
    if not qkv.is_contiguous() or qkv.dtype != _HALF:
        msg = "rope_qk: contiguous fp16 qkv"
        raise ValueError(msg)
    b, n, c3 = qkv.shape
    d = c3 // 3 // heads
    e = emb.to(torch.float32).contiguous()
    if tuple(e.shape) != (n - npt, 2 * d):
        msg = f"rope_qk: emb shape {tuple(e.shape)} vs N - npt = {n - npt}"
        raise ValueError(msg)
    q = torch.empty((b, n, heads, d), dtype=_HALF, device=qkv.device)
    k = torch.empty_like(q)
    rc = _lib("extra", qkv).rope_qk(
        qkv.data_ptr(), e.data_ptr(), q.data_ptr(), k.data_ptr(), b, n, heads, d, npt, current_stream()
    )
    _check(rc, "fvm_extra rope_qk")
    return q, k


def swiglu_ln(gx: torch.Tensor, w: torch.Tensor, b: torch.Tensor, hid: int, eps: float) -> torch.Tensor:
    """Gx `[..., 2*Hp]` = [gate | x] -> LayerNorm(silu(gate) * x) over the first `hid` channels, `[..., Hp]`."""
    if not gx.is_contiguous() or gx.dtype != _HALF:
        msg = "swiglu_ln: contiguous fp16"
        raise ValueError(msg)
    hp = gx.shape[-1] // 2
    m = gx.numel() // gx.shape[-1]
    y = torch.empty((*gx.shape[:-1], hp), dtype=_HALF, device=gx.device)
    rc = _lib("extra", gx).swiglu_ln(
        gx.data_ptr(), w.data_ptr(), b.data_ptr(), y.data_ptr(), m, hid, hp, eps, current_stream()
    )
    _check(rc, "fvm_extra swiglu_ln")
    return y
