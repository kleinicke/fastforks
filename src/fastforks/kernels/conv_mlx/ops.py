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

import numpy as np

from fastforks._core.mlx import metal_kernel, mx

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


def _header(m: int) -> str:
    def arr(a: list[list[float]]) -> str:
        return "{" + ",".join(f"{float(v)!r}f" for r in a for v in r) + "}"

    return (
        '#define UR _Pragma("clang loop unroll(full)")\n'
        f"constant constexpr float BT[] = {arr(_MATS[m]['BT'])};\n"
        f"constant constexpr float AT[] = {arr(_MATS[m]['AT'])};\n"
    )


_IN_SRC = """
    constexpr int A = MO + 2;
    uint cv = thread_position_in_grid.x, tx = thread_position_in_grid.y, tyn = thread_position_in_grid.z;
    if (cv >= C / CV) return;
    uint ty = tyn % TH, n = tyn / TH;
    int ih0 = int(ty + r0[0]) * MO - 1, iw0 = int(tx) * MO - 1;
    typedef vec<float, CV> F;
    F d[A][A];
    const device vec<T, CV>* xv = (const device vec<T, CV>*)x;
    UR for (int i = 0; i < A; ++i) UR for (int j = 0; j < A; ++j) {
        int ih = ih0 + i, iw = iw0 + j;
        bool ok = ih >= 0 && ih < H && iw >= 0 && iw < W;
        d[i][j] = ok ? F(xv[((size_t(n) * H + ih) * W + iw) * (C / CV) + cv]) : F(0);
    }
    F t[A][A];
    UR for (int i = 0; i < A; ++i) UR for (int j = 0; j < A; ++j) {
        F s = F(0);
        UR for (int k = 0; k < A; ++k) s += BT[i * A + k] * d[k][j];
        t[i][j] = s;
    }
    size_t tile = (size_t(n) * TH + ty) * TW + tx;
    device vec<T, CV>* vv = (device vec<T, CV>*)v;
    UR for (int i = 0; i < A; ++i) UR for (int j = 0; j < A; ++j) {
        F s = F(0);
        UR for (int k = 0; k < A; ++k) s += BT[j * A + k] * t[i][k];
        vv[(size_t(i * A + j) * NT + tile) * (C / CV) + cv] = vec<T, CV>(s);
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
    UR for (int f = 0; f < A * A; ++f) mm[f / A][f % A] = F(mv[(size_t(f) * NT + tile) * (CO / CV) + cv]);
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

_K: dict[int, tuple[object, object]] = {}


def _kernels(m: int) -> tuple[object, object]:
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


def supported(x_shape: tuple[int, ...], cout: int) -> bool:
    """Whether `conv3x3` handles an NHWC input of this shape (channels even) and is likely to beat `mx.conv2d`."""
    _, h, w, cin = x_shape
    return cin % CV == 0 and cout % CV == 0 and cin >= _MIN_C and cout >= _MIN_C and h * w >= 48 * 48


_BAND_BYTES = 192 << 20  # target size of one band's transformed activations: bounds the peak of huge images


def _band(x: mx.array, u: mx.array, bias, m, act, slope, residual, r0: int, bth: int):  # noqa: ANN001, ANN202, PLR0913, PLR0917
    n, h, wd, c = x.shape
    a = m + 2
    co = u.shape[2]
    tw = -(-wd // m)
    nt = n * bth * tw
    bh = min(bth * m, h - r0 * m)
    k_in, k_out = _kernels(m)
    off = mx.array([r0], dtype=mx.uint32)
    tx = min(_TG // (c // CV), tw) if c // CV < _TG else 1
    v = k_in(  # ty: ignore[call-non-callable]
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
            ("CV", CV),
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
            bias if bias is not None else zero,
            slope.astype(x.dtype) if slope is not None else zero,
            residual if residual is not None else zero,
            off,
        ],
        template=[
            ("T", x.dtype), ("CO", co), ("H", h), ("W", wd), ("TH", bth), ("TW", tw), ("NT", nt),
            ("MO", m), ("CV", CV), ("ACT", act), ("RES", residual is not None), ("BH", bh),
        ],
        grid=(co // CV, tw, n * bth),
        threadgroup=(min(co // CV, 256), max(1, txo), 1),
        output_shapes=[(n, bh, wd, co)],
        output_dtypes=[x.dtype],
    )[0]  # fmt: skip


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
