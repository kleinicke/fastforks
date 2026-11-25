"""Channels-last Winograd 3x3 stride-1 convolution on Apple MLX (`fastforks.shared.mlx_graph`, kernels mode).

MLX's dense 3x3 convolution is an implicit GEMM that runs at about 3.5 TFLOPS, slower than Metal Performance Shaders.
The Winograd form needs 2.25x (F(2,3)) or 4x (F(4,3)) fewer multiplications and runs as three dispatches:

1. `_IN`: every `(m+2) x (m+2)` input tile is transformed (`B^T d B`) into `(m+2)^2` rows of a `[freq, tiles, Cin]`
   matrix;
2. `mx.matmul`: one `[tiles, Cin] x [Cin, Cout]` GEMM per frequency (the transformed weights are constants, made once);
3. `_OUT`: the inverse transform (`A^T M A`) with bias, activation and an optional residual add in one pass.

Transforms and epilogue run in float32 registers; the tile matrices are stored in the activation dtype, so the
float16 result is a little less exact than the direct convolution's (F(4,3) about 4e-3 relative RMS, F(2,3) 7e-4).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

from fastforks._core.mlx import metal_kernel, mx

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_MATS = {
    2: {
        "BT": [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]],
        "G": [[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]],
        "AT": [[1, 1, 1, 0], [0, 1, -1, -1]],
    },
    4: {
        "BT": [
            [4, 0, -5, 0, 1, 0],
            [0, -4, -4, 1, 1, 0],
            [0, 4, -4, -1, 1, 0],
            [0, -2, -1, 2, 1, 0],
            [0, 2, -1, -2, 1, 0],
            [0, 4, 0, -5, 0, 1],
        ],
        "G": [
            [1 / 4, 0, 0],
            [-1 / 6, -1 / 6, -1 / 6],
            [-1 / 6, 1 / 6, -1 / 6],
            [1 / 24, 1 / 12, 1 / 6],
            [1 / 24, -1 / 12, 1 / 6],
            [0, 0, 1],
        ],
        "AT": [[1, 1, 1, 1, 1, 0], [0, 1, -1, 2, -2, 0], [0, 1, 1, 4, 4, 0], [0, 1, -1, 8, -8, 1]],
    },
}
TILES = tuple(_MATS)
CV = 2  # channels per thread: the transform keeps (m+2)^2 vectors of this width in registers
_TG = 256  # threads per threadgroup
_MIN_C = 16  # fewer channels: the transforms cost more than they save
_WIDE_C = 64  # from this many channels on a small image still wins (see `supported`)
_MIN_D = 2  # depth slices of the 3x3x3 form
_MIN_VOXELS = 1024  # smaller volumes: launch cost dominates


def _header(m: int) -> str:
    def arr(a: Sequence[Sequence[float]]) -> str:
        return "{" + ",".join(f"{float(v)!r}f" for r in a for v in r) + "}"

    return (
        '#define UR _Pragma("clang loop unroll(full)")\n'
        f"constant constexpr float BT[] = {arr(_MATS[m]['BT'])};\n"
        f"constant constexpr float AT[] = {arr(_MATS[m]['AT'])};\n"
    )


_IN_SRC = """
    constexpr int A = MO + 2, CC = C / CV, NK = K3 ? 3 : 1;
    uint cv = thread_position_in_grid.x, tx = thread_position_in_grid.y, tyn = thread_position_in_grid.z;
    if (cv >= NK * CC) return;
    // K3: the 3x3x3 convolution reads each depth slice `sd = d + kd - 1` for kd = 0..2, one channel block each
    uint kd = cv / CC, cc = cv % CC;
    uint ty = tyn % TH, nd = tyn / TH;
    int sd = K3 ? int(nd % D) + int(kd) - 1 : 0;
    uint n = K3 ? (nd / D) * D + uint(sd) : nd;
    bool dok = !K3 || (sd >= 0 && sd < D);
    int ih0 = int(ty + r0[0]) * MO - 1, iw0 = int(tx) * MO - 1;
    typedef vec<float, CV> F;
    F d[A][A];
    const device vec<T, CV>* xv = (const device vec<T, CV>*)x;
    UR for (int i = 0; i < A; ++i) UR for (int j = 0; j < A; ++j) {
        int ih = ih0 + i, iw = iw0 + j;
        bool ok = dok && ih >= 0 && ih < H && iw >= 0 && iw < W;
        d[i][j] = ok ? F(xv[((size_t(n) * H + ih) * W + iw) * CC + cc]) : F(0);
    }
    F t[A][A];
    UR for (int i = 0; i < A; ++i) UR for (int j = 0; j < A; ++j) {
        F s = F(0);
        UR for (int k = 0; k < A; ++k) s += BT[i * A + k] * d[k][j];
        t[i][j] = s;
    }
    size_t tile = (size_t(nd) * TH + ty) * TW + tx;
    device vec<T, CV>* vv = (device vec<T, CV>*)v;
    UR for (int i = 0; i < A; ++i) UR for (int j = 0; j < A; ++j) {
        F s = F(0);
        UR for (int k = 0; k < A; ++k) s += BT[j * A + k] * t[i][k];
        vv[(size_t(i * A + j) * NT + tile) * (NK * CC) + cv] = vec<T, CV>(s * (1.0f / float(SC)));
    }
"""

_OUT_SRC = """
    constexpr int A = MO + 2;
    uint cv = thread_position_in_grid.x, tx = thread_position_in_grid.y, tyn = thread_position_in_grid.z;
    if (cv >= CO / CV) return;
    uint ty = tyn % TH, n = tyn / TH;
    size_t tile = (size_t(n) * TH + ty) * TW + tx;
    typedef vec<float, CV> F;
    const device vec<T, CV>* mv = (const device vec<T, CV>*)m;
    F mm[A][A];
    UR for (int f = 0; f < A * A; ++f) mm[f / A][f % A] = F(mv[(size_t(f) * NT + tile) * (CO / CV) + cv]) * float(SC);
    F t[MO][A];
    UR for (int i = 0; i < MO; ++i) UR for (int j = 0; j < A; ++j) {
        F s = F(0);
        UR for (int k = 0; k < A; ++k) s += AT[i * A + k] * mm[k][j];
        t[i][j] = s;
    }
    F bias = F(((const device vec<T, CV>*)b)[cv]);
    const int row0 = int(r0[0]) * MO;
    F slope = F(((const device vec<T, CV>*)a)[cv]);
    device vec<T, CV>* ov = (device vec<T, CV>*)out;
    const device vec<T, CV>* rv = (const device vec<T, CV>*)r;
    UR for (int i = 0; i < MO; ++i) {
        int ol = int(ty) * MO + i, oh = row0 + ol;
        if (oh >= H) continue;
        UR for (int j = 0; j < MO; ++j) {
            int ow = int(tx) * MO + j;
            if (ow >= W) continue;
            F s = bias;
            UR for (int k = 0; k < A; ++k) s += AT[j * A + k] * t[i][k];
            size_t at = ((size_t(n) * H + oh) * W + ow) * (CO / CV) + cv;
            if (ACT == 1) s = max(s, F(0));
            else if (ACT == 2) s = select(s * slope, s, s > F(0));
            if (RES) s += F(rv[at]);
            ov[((size_t(n) * BH + ol) * W + ow) * (CO / CV) + cv] = vec<T, CV>(s);
        }
    }
"""

_K: dict[int, tuple[Callable[..., list[mx.array]], Callable[..., list[mx.array]]]] = {}


def _scale(m: int, dtype: mx.Dtype) -> int:
    """Float16 range guard: `v` and the GEMM output are stored divided by this, the output transform multiplies it back.

    F(4,3)'s input transform grows a tile up to 100x (|B^T| row sums 10), so float16 activations above ~650 overflowed;
    F(2,3) grows 4x. Powers of two: exact in any dtype.
    """
    if dtype != mx.float16:
        return 1
    return 32 if m == 4 else 4


def _kernels(m: int) -> tuple[Callable[..., list[mx.array]], Callable[..., list[mx.array]]]:
    if m not in _K:
        _K[m] = (
            metal_kernel(
                name=f"conv_wino{m}_in", input_names=["x", "r0"], output_names=["v"], source=_IN_SRC, header=_header(m)
            ),
            metal_kernel(
                name=f"conv_wino{m}_out",
                input_names=["m", "b", "a", "r", "r0"],
                output_names=["out"],
                source=_OUT_SRC,
                header=_header(m),
            ),
        )
    return _K[m]


def weights(w: mx.array, m: int, dtype: mx.Dtype) -> mx.array:
    """Winograd-domain weights `[(m+2)^2, Cin, Cout]` from an OHWI 3x3 kernel (`G g G^T` in float64, then rounded)."""
    g = np.array(w.astype(mx.float32), dtype=np.float64)
    gm = np.array(_MATS[m]["G"], dtype=np.float64)
    u = np.einsum("ai,oijc,bj->abco", gm, g, gm)
    a = m + 2
    out = mx.array(u.reshape(a * a, g.shape[3], g.shape[0]).astype(np.float32)).astype(dtype)
    mx.eval(out)
    return out


def weights_graph(w: mx.array, m: int, dtype: mx.Dtype) -> mx.array:
    """`weights(w, m, dtype)` as MLX ops (float32 `kron(G, G)` GEMM), for a weight that may be a graph node.

    Computed inside the replayed graph on every call instead of held: a large layer's transformed weights (4x the
    kernel for F(4,3)) then cost one layer's transient, not their size for the life of the model. float32 accumulation,
    so the result differs from `weights` (float64) by float32 rounding only.
    """
    o, kh, kw, c = w.shape
    gm = np.array(_MATS[m]["G"], dtype=np.float64)
    kron = mx.array(np.kron(gm, gm).astype(np.float32))  # [(a, b), (i, j)]
    wm = mx.transpose(w.astype(mx.float32), (1, 2, 3, 0)).reshape(kh * kw, c * o)
    return mx.matmul(kron, wm).reshape((m + 2) ** 2, c, o).astype(dtype)


def supported(x_shape: tuple[int, ...], cout: int) -> bool:
    """Whether `conv3x3` handles an NHWC input of this shape (channels even) and is likely to beat `mx.conv2d`.

    Measured with chained layers (launch cost amortised): 16-32 channels need a large image (`h*w >= 48^2`); from 64
    channels on it wins as soon as the layer has about 2^25 multiply-adds per tap (`h*w*cin*cout`: 256 channels at
    24^2, 128 at 48^2, 512 at 16^2), and not below (64 channels at 32^2 ties, 128 at 32^2 loses).
    """
    _, h, w, cin = x_shape
    if cin % CV or cout % CV or cin < _MIN_C or cout < _MIN_C:
        return False
    return h * w >= 48 * 48 or (min(cin, cout) >= _WIDE_C and h * w * cin * cout >= 1 << 25)


def supported3d(x_shape: tuple[int, ...], cout: int, itemsize: int = 2) -> bool:
    """Whether `conv3d` handles an NDHWC input of this shape (even channels, at least 16) within its memory cap.

    The 3x3x3 form materialises the transformed input with the three depth taps stacked (`36 x tiles x 3*Cin`) and the
    product (`36 x tiles x Cout`), about 10 activations' worth; layers whose two tensors exceed `_BAND_BYTES` fall back
    to `mx.conv3d` (whose own temporaries are tiny), so the extra memory stays bounded per layer.
    """
    n, d, h, w, cin = x_shape
    if cin % CV or cout % CV or cin < _MIN_C or cout < _MIN_C or d < _MIN_D or d * h * w < _MIN_VOXELS:
        return False
    tiles = n * d * -(-h // 4) * -(-w // 4)
    return 36 * tiles * (3 * cin + cout) * itemsize <= _BAND_BYTES


_BAND_BYTES = (
    int(os.environ.get("FF_BAND_MB", "192")) << 20
)  # target size of one band's transformed activations: bounds the peak of huge images


def _as(a: mx.array, dtype: mx.Dtype) -> mx.array:
    """`a` in the activation dtype.

    The kernels read bias, slope and residual as `T`: an fp16 bias under fp32 activations (the fp32 islands of an fp16
    net) would be reinterpreted, not converted.
    """
    return a if a.dtype == dtype else a.astype(dtype)


def _band(x: mx.array, u: mx.array, bias, m, act, slope, residual, r0: int, bth: int):  # noqa: ANN001, ANN202, PLR0913
    n, h, wd, c = x.shape
    a = m + 2
    co = u.shape[2]
    tw = -(-wd // m)
    nt = n * bth * tw
    bh = min(bth * m, h - r0 * m)
    k_in, k_out = _kernels(m)
    off = mx.array([r0], dtype=mx.uint32)
    tx = min(_TG // (c // CV), tw) if c // CV < _TG else 1
    v = k_in(
        inputs=[x, off],
        template=[
            ("T", x.dtype),
            ("C", c),
            ("H", h),
            ("W", wd),
            ("TH", bth),
            ("TW", tw),
            ("NT", nt),
            ("MO", m),
            ("SC", _scale(m, x.dtype)),
            ("CV", CV),
            ("K3", 0),
            ("D", 1),
        ],
        grid=(c // CV, tw, n * bth),
        threadgroup=(min(c // CV, 256), max(1, tx), 1),
        output_shapes=[(a * a, nt, c)],
        output_dtypes=[x.dtype],
    )[0]
    p = mx.matmul(v, u)
    zero = mx.zeros((co,), dtype=x.dtype)
    txo = min(_TG // (co // CV), tw) if co // CV < _TG else 1
    return k_out(
        inputs=[
            p,
            _as(bias, x.dtype) if bias is not None else zero,
            _as(slope, x.dtype) if slope is not None else zero,
            _as(residual, x.dtype) if residual is not None else zero,
            off,
        ],
        template=[
            ("T", x.dtype), ("CO", co), ("H", h), ("W", wd), ("TH", bth), ("TW", tw), ("NT", nt),
            ("MO", m), ("SC", _scale(m, x.dtype)), ("CV", CV), ("ACT", act), ("RES", residual is not None), ("BH", bh),
        ],
        grid=(co // CV, tw, n * bth),
        threadgroup=(min(co // CV, 256), max(1, txo), 1),
        output_shapes=[(n, bh, wd, co)],
        output_dtypes=[x.dtype],
    )[0]  # fmt: skip


def weights3d(w: mx.array, m: int, dtype: mx.Dtype) -> mx.array:
    """Winograd-domain weights `[(m+2)^2, 3*Cin, Cout]` of an ODHWI 3x3x3 kernel (the depth taps stacked along K)."""
    return mx.concatenate([weights(w[:, kd], m, dtype) for kd in range(3)], axis=1)


def conv3d(
    x: mx.array,
    u: mx.array,
    bias: mx.array | None = None,
    *,
    act: int = 0,
    slope: mx.array | None = None,
) -> mx.array:
    """NDHWC 3x3x3 convolution, stride 1, zero padding 1, with `u = weights3d(..., 4, dtype)`, F(4,3) per depth slice.

    Output slice `d` needs input slices `d-1, d, d+1`: the input transform writes them side by side along the channel
    axis (zeros past the ends) so a single batched GEMM with `K = 3*Cin` sums the three taps; the inverse transform
    (with bias and `act` as in `conv3x3`) runs per slice. The whole volume is one band: see `supported3d` for the cap.
    """
    n, d, h, wd, c = x.shape
    m = 4
    co = u.shape[2]
    th, tw = -(-h // m), -(-wd // m)
    nt = n * d * th * tw
    k_in, k_out = _kernels(m)
    off = mx.array([0], dtype=mx.uint32)
    c3 = 3 * c
    tx = min(_TG // (c3 // CV), tw) if c3 // CV < _TG else 1
    v = k_in(
        inputs=[x.reshape(n * d, h, wd, c), off],
        template=[
            ("T", x.dtype), ("C", c), ("H", h), ("W", wd), ("TH", th), ("TW", tw), ("NT", nt),
            ("MO", m), ("SC", _scale(m, x.dtype)), ("CV", CV), ("K3", 1), ("D", d),
        ],
        grid=(c3 // CV, tw, n * d * th),
        threadgroup=(min(c3 // CV, 256), max(1, tx), 1),
        output_shapes=[(36, nt, c3)],
        output_dtypes=[x.dtype],
    )[0]  # fmt: skip
    p = mx.matmul(v, u)
    zero = mx.zeros((co,), dtype=x.dtype)
    txo = min(_TG // (co // CV), tw) if co // CV < _TG else 1
    y = k_out(
        inputs=[
            p,
            _as(bias, x.dtype) if bias is not None else zero,
            _as(slope, x.dtype) if slope is not None else zero,
            zero,
            off,
        ],
        template=[
            ("T", x.dtype), ("CO", co), ("H", h), ("W", wd), ("TH", th), ("TW", tw), ("NT", nt),
            ("MO", m), ("SC", _scale(m, x.dtype)), ("CV", CV), ("ACT", act), ("RES", False), ("BH", h),
        ],
        grid=(co // CV, tw, n * d * th),
        threadgroup=(min(co // CV, 256), max(1, txo), 1),
        output_shapes=[(n * d, h, wd, co)],
        output_dtypes=[x.dtype],
    )[0]  # fmt: skip
    return y.reshape(n, d, h, wd, co)


def conv3x3(  # noqa: PLR0913
    x: mx.array,
    u: mx.array,
    bias: mx.array | None = None,
    *,
    m: int = 4,
    act: int = 0,
    slope: mx.array | None = None,
    residual: mx.array | None = None,
    bands: int | None = None,
) -> mx.array:
    """NHWC 3x3 convolution, stride 1, zero padding 1, with `u = weights(...)`; bias, activation and residual fused.

    `act`: 0 none, 1 ReLU, 2 PReLU with the per-channel `slope` `[Cout]` (a LeakyReLU is a PReLU with a constant slope).
    `bands`: the image is processed in this many bands of tile rows (default: as many as keep one band's transformed
    activations under `_BAND_BYTES`): the transformed tensors of a huge image never exist whole.
    """
    n, h, wd, c = x.shape
    th = -(-h // m)
    if bands is None:
        v_bytes = (m + 2) ** 2 * n * th * -(-wd // m) * max(c, u.shape[2]) * x.dtype.size
        bands = max(1, v_bytes // _BAND_BYTES)
    bands = min(bands, th)
    bth = -(-th // bands)
    outs = [_band(x, u, bias, m, act, slope, residual, r0, min(bth, th - r0)) for r0 in range(0, th, bth)]
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)


# ---- direct convolution for few channels --------------------------------------------------------------------------
# MLX's implicit GEMM runs a conv with 3-32 input channels at full resolution at about 1 TFLOPS (RIFE's 16 -> 16 at
# 576x960: 2.6 ms, MPS 0.95; its 28 -> 16 stride-2 head at 1152x1920: 8.1 ms, MPS 4.2). This kernel register-blocks
# PX output pixels of a row x COB output channels per thread, reads each input row segment once per (kh, channel
# block) and the weights as uniform loads, accumulating in float32.
_DIRECT_SRC = """
    constexpr int XW = (PX - 1) * SW + (KW - 1) * DW + 1;
    constexpr int CB = C / CV, QB = COB / 4;
    uint cob = thread_position_in_grid.x, oxb = thread_position_in_grid.y, nh = thread_position_in_grid.z;
    if (cob >= CO / COB || oxb * PX >= OW) return;
    uint n = nh / OH, oy = nh % OH;
    int ox0 = int(oxb) * PX;
    float4 acc[PX][QB];
    UR for (int q = 0; q < QB; ++q) {
        uint c4 = (cob * QB + q) * 4;  // b / a: element reads (MLX may place small arrays in constant memory)
        float4 bb = HASB ? float4(b[c4], b[c4 + 1], b[c4 + 2], b[c4 + 3]) : float4(0);
        UR for (int p = 0; p < PX; ++p) acc[p][q] = bb;
    }
    const device vec<T, CV>* xv = (const device vec<T, CV>*)x;
    const device vec<T, 4>* wv = (const device vec<T, 4>*)w;
    int ix0 = ox0 * SW - PW;
    UR for (int kh = 0; kh < KH; ++kh) {
        int iy = int(oy) * SH - PH + kh * DH;
        if (iy < 0 || iy >= H) continue;
        size_t row = (size_t(n) * H + iy) * W;
        for (int cb = 0; cb < CB; ++cb) {
            vec<float, CV> xr[XW];
            UR for (int j = 0; j < XW; ++j) {
                int ix = ix0 + j;
                xr[j] = (ix >= 0 && ix < W) ? vec<float, CV>(xv[(row + ix) * CB + cb]) : vec<float, CV>(0);
            }
            UR for (int kw = 0; kw < KW; ++kw) UR for (int cc = 0; cc < CV; ++cc) {
                size_t wo = ((size_t(kh * KW + kw) * C + cb * CV + cc) * CO) / 4 + cob * QB;
                UR for (int q = 0; q < QB; ++q) {
                    float4 wq = float4(wv[wo + q]);
                    UR for (int p = 0; p < PX; ++p) acc[p][q] += xr[p * SW + kw * DW][cc] * wq;
                }
            }
        }
    }
    device vec<T, 4>* ov = (device vec<T, 4>*)out;
    UR for (int p = 0; p < PX; ++p) {
        int ox = ox0 + p;
        if (ox >= OW) break;
        size_t o = ((size_t(n) * OH + oy) * OW + ox) * (CO / 4) + cob * QB;
        UR for (int q = 0; q < QB; ++q) {
            float4 s = acc[p][q];
            if (ACT == 1) s = max(s, 0.0f);
            else if (ACT == 2) {
                uint c4 = (cob * QB + q) * 4;
                s = select(s * float4(a[c4], a[c4 + 1], a[c4 + 2], a[c4 + 3]), s, s > 0.0f);
            }
            ov[o + q] = vec<T, 4>(s);
        }
    }
"""
_DIRECT: list[object] = []
DIRECT_MAX_CIN = 32  # wider inputs: MLX's GEMM (or Winograd) is as fast or faster


def direct_supported(cin: int, cout: int, k: tuple[int, int], stride: tuple[int, int]) -> bool:
    """Whether `direct` takes a dense conv of `cin -> cout` channels, kernel `k`, `stride` and is likely faster.

    Measured (M1 Pro, fp16, single image, 288x480 .. 1280x1280): 2-2.4x faster than `mx.conv2d` for 3-28 input
    channels at stride 2 (to 48 outputs) and for 16 at stride 1 (to 32 outputs); 32 -> 32 at stride 1 ties and 48 or
    more channels lose, as do 7x7 stems with 64 outputs.
    """
    if cout % 4 or k[0] * k[1] > 9 or k[0] * k[1] == 1:
        return False
    if stride == (2, 2):
        return cin <= DIRECT_MAX_CIN and cout <= 48
    return stride == (1, 1) and cin <= 16 and cout <= 32


def direct(  # noqa: PLR0913
    x: mx.array,
    w: mx.array,
    bias: mx.array | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int] = (1, 1),
    *,
    act: int = 0,
    slope: mx.array | None = None,
) -> mx.array:
    """NHWC dense convolution with an OHWI kernel `w`, bias and `act` (as in `conv3x3`) fused; float32 accumulation.

    Odd input channel counts are zero-padded to a multiple of 4 (a 3-channel image stem).
    """
    if not _DIRECT:
        _DIRECT.append(
            metal_kernel(
                name="conv_direct",
                input_names=["x", "w", "b", "a"],
                output_names=["out"],
                source=_DIRECT_SRC,
                header='#define UR _Pragma("clang loop unroll(full)")\n',
            )
        )
    n, h, wd, c = x.shape
    co, kh, kw, _ = w.shape
    wt = mx.transpose(w, (1, 2, 3, 0))  # [KH, KW, C, CO]: one output-channel vector per (tap, input channel)
    if c % 2:
        cp = -(-c // 4) * 4
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, cp - c)])
        wt = mx.pad(wt, [(0, 0), (0, 0), (0, cp - c), (0, 0)])
        c = cp
    oh = (h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    ow = (wd + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    cob = 8 if co % 8 == 0 else 4
    px = 2
    ncb, owb = co // cob, -(-ow // px)
    zero = mx.zeros((co,), dtype=x.dtype)
    return _DIRECT[0](  # ty: ignore[call-non-callable]
        inputs=[
            x,
            _as(wt, x.dtype),
            _as(bias, x.dtype) if bias is not None else zero,
            _as(slope, x.dtype) if slope is not None else zero,
        ],
        template=[
            ("T", x.dtype), ("C", c), ("CO", co), ("H", h), ("W", wd), ("OH", oh), ("OW", ow), ("KH", kh), ("KW", kw),
            ("SH", stride[0]), ("SW", stride[1]), ("PH", padding[0]), ("PW", padding[1]), ("DH", dilation[0]),
            ("DW", dilation[1]), ("PX", px), ("COB", cob), ("CV", 4 if c % 4 == 0 else 2), ("HASB", bias is not None),
            ("ACT", act),
        ],
        grid=(ncb, owb, n * oh),
        threadgroup=(ncb, max(1, min(owb, _TG // ncb)), 1),
        output_shapes=[(n, oh, ow, co)],
        output_dtypes=[x.dtype],
    )[0]  # fmt: skip
