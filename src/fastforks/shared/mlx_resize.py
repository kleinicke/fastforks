"""PyTorch's CPU `F.interpolate` bit for bit, without torch: bicubic and bilinear (`align_corners=True`), NCHW float32.

The MLX ports that resized their inputs and outputs with torch on the CPU run these instead, so an MLX call needs no
torch and gives the same numbers. Taps and weights are computed on the host as torch computes them (float32 with
fused multiply-adds: arm64 builds of torch contract `a * b + c`), the sums run in the same order with fused
multiply-adds: in a Metal kernel for images and maps (`bicubic`, `bilinear_aligned`), in NumPy for small tables
(`bicubic_host`). The NumPy fused multiply-add rounds twice (exact float64 product + sum, then float32), which
differs from one rounding only in rare halfway cases (none in the tests).

- `bicubic`: `F.interpolate(x, size, mode="bicubic", align_corners=False)` (a = -0.75, borders replicated), optionally
  followed by `(x - mean) / std` per channel (`fastforks._core.preprocess.scale`).
- `bilinear_aligned`: `F.interpolate(x, size, mode="bilinear", align_corners=True)` (both of torch's CPU kernels; not
  its single-thread 3-channel case, `torch.get_num_threads() == 1`).
- `bicubic_host`: `bicubic` by size or by scale factors (`scale_factor=`, as DINOv2's `interpolate_pos_encoding`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fastforks._core.mlx import metal_kernel, mx, tuning

if TYPE_CHECKING:
    from collections.abc import Sequence

_A = np.float32(-0.75)  # torch's bicubic coefficient
_F32 = np.float32
_SMALL = 128  # torch's bilinear: outputs with OH + OW <= 128 run its channels-last kernel


def _fma(a: np.ndarray | np.floating, b: np.ndarray | np.floating, c: np.ndarray | np.floating) -> np.ndarray:
    """float32 `a * b + c` (the float64 product of two float32 values is exact)."""
    return np.asarray(np.asarray(a, np.float64) * b + c, np.float32)


def _cc1(x: np.ndarray) -> np.ndarray:
    """Torch's `cubic_convolution1` (`|x| <= 1`)."""
    return _fma(_fma(_F32(_A + 2), x, -_F32(_A + 3)) * x, x, _F32(1))


def _cc2(x: np.ndarray) -> np.ndarray:
    """Torch's `cubic_convolution2` (`1 < |x| < 2`)."""
    return _fma(_fma(_fma(_A, x, -_F32(5) * _A), x, _F32(8) * _A), x, -_F32(4) * _A)


def cubic_taps(n_in: int, n_out: int, scale: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Bicubic (`align_corners=False`) taps of one axis: source indices `[4, n_out]` and their float32 weights.

    `scale`: torch's `scale_factor` for this axis (`None`: by size).
    """
    s = _F32(1.0 / scale) if scale else _F32(n_in) / _F32(n_out)
    real = _fma(np.full(n_out, s), np.arange(n_out, dtype=np.float32) + _F32(0.5), _F32(-0.5))
    i = np.floor(real)
    t = (real - i).astype(np.float32)
    i = i.astype(np.int64)
    u = _F32(1) - t  # torch rounds 1 - t, then adds 1 for the last tap (not 2 - t)
    w = np.stack([_cc2(t + _F32(1)), _cc1(t), _cc1(u), _cc2(u + _F32(1))])
    idx = np.stack([np.clip(i + k, 0, n_in - 1) for k in (-1, 0, 1, 2)])
    return idx, w


def linear_taps_aligned(n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear (`align_corners=True`) taps of one axis: source indices `[2, n_out]` and their float32 weights."""
    s = _F32(n_in - 1) / _F32(n_out - 1) if n_out > 1 else _F32(0)
    real = s * np.arange(n_out, dtype=np.float32)
    i0 = np.minimum(np.floor(real).astype(np.int64), n_in - 1)
    l1 = (real - i0).astype(np.float32)
    return np.stack([i0, np.minimum(i0 + 1, n_in - 1)]), np.stack([_F32(1) - l1, l1])


def bicubic_host(x: np.ndarray, size: Sequence[int], scale: Sequence[float] | None = None) -> np.ndarray:
    """`F.interpolate(x, size, mode="bicubic")` of a float32 `[..., H, W]` array (`scale`: the scale factors)."""
    sy, sx = scale if scale is not None else (None, None)
    iy, wy = cubic_taps(x.shape[-2], size[0], sy)
    ix, wx = cubic_taps(x.shape[-1], size[1], sx)
    out = None
    for j in range(4):
        r = x[..., iy[j], :]
        acc = r[..., ix[0]] * wx[0]
        for k in range(1, 4):
            acc = _fma(r[..., ix[k]], wx[k], acc)
        out = acc * wy[j][:, None] if out is None else _fma(acc, wy[j][:, None], out)
    assert out is not None  # noqa: S101 - four taps
    return out


# one thread per output value of `[N * C, OH, OW]`; rows summed first (x taps), then the rows (y taps), as torch
_BICUBIC = metal_kernel(
    name="torch_bicubic",
    input_names=["x", "iy", "wy", "ix", "wx", "mean", "inv"],
    output_names=["out"],
    source="""
        uint C = x_shape[1], H = x_shape[2], W = x_shape[3], OH = iy_shape[1], OW = ix_shape[1];
        uint pix = thread_position_in_grid.x;
        if (pix >= x_shape[0] * C * OH * OW) return;
        uint ox = pix % OW, oy = (pix / OW) % OH, nc = pix / (OW * OH);
        const device float* src = x + size_t(nc) * H * W;
        float v = 0.0f;
        for (uint j = 0; j < 4; ++j) {
            const device float* row = src + size_t(iy[j * OH + oy]) * W;
            float acc = row[ix[ox]] * wx[ox];
            for (uint k = 1; k < 4; ++k) acc = metal::precise::fma(row[ix[k * OW + ox]], wx[k * OW + ox], acc);
            float wj = wy[j * OH + oy];
            v = j == 0 ? acc * wj : metal::precise::fma(acc, wj, v);
        }
        if (NORM) v = metal::precise::divide(v - mean[nc % C], inv[nc % C]);
        out[pix] = v;
    """,
)

# one thread per output value. Torch's generic kernel: `w0 * x0 + (w1 * x1)` along x, then the same along y. Outputs
# with OH + OW <= 128 take its channels-last kernel (`_use_vectorized_kernel_cond_2d`): the four corner weights
# `wy * wx`, summed in vector order for the channels of whole 4-lanes, in scalar order (fused) for the rest.
_BILINEAR = metal_kernel(
    name="torch_bilinear_aligned",
    input_names=["x", "iy", "wy", "ix", "wx"],
    output_names=["out"],
    source="""
        uint C = x_shape[1], H = x_shape[2], W = x_shape[3], OH = iy_shape[1], OW = ix_shape[1];
        uint pix = thread_position_in_grid.x;
        if (pix >= x_shape[0] * C * OH * OW) return;
        uint ox = pix % OW, oy = (pix / OW) % OH, nc = pix / (OW * OH);
        const device float* src = x + size_t(nc) * H * W;
        uint x0 = ix[ox], x1 = ix[OW + ox];
        float a0 = wx[ox], a1 = wx[OW + ox], b0 = wy[oy], b1 = wy[OH + oy];
        const device float* r0 = src + size_t(iy[oy]) * W;
        const device float* r1 = src + size_t(iy[OH + oy]) * W;
        if (SMALL) {
            float w00 = b0 * a0, w01 = b0 * a1, w10 = b1 * a0, w11 = b1 * a1;
            float p00 = r0[x0] * w00, p01 = r0[x1] * w01, p10 = r1[x0] * w10, p11 = r1[x1] * w11;
            if (nc % C < C - C % 4) {
                out[pix] = p00 + (p01 + (p10 + p11));
            } else {
                float v = metal::precise::fma(r0[x0], w00, p01);
                v = metal::precise::fma(r1[x0], w10, v);
                out[pix] = metal::precise::fma(r1[x1], w11, v);
            }
            return;
        }
        float t0 = metal::precise::fma(r0[x0], a0, r0[x1] * a1), t1 = metal::precise::fma(r1[x0], a0, r1[x1] * a1);
        out[pix] = metal::precise::fma(t0, b0, t1 * b1);
    """,
)


def _run(kernel: object, x: mx.array, size: Sequence[int], taps: tuple, extra: list, template: list) -> mx.array:  # noqa: PLR0913, PLR0917 - the kernel's pieces
    (iy, wy), (ix, wx) = taps
    n, c = x.shape[:2]
    h, w = int(size[0]), int(size[1])
    inputs = [x.astype(mx.float32), mx.array(iy.astype(np.uint32)), mx.array(wy), mx.array(ix.astype(np.uint32))]
    inputs += [mx.array(wx), *extra]
    total = n * c * h * w
    tg = tuning().threadgroup
    return kernel(  # ty: ignore[call-non-callable] - `metal_kernel`'s callable
        inputs=inputs,
        template=template,
        grid=(-(-total // tg) * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(n, c, h, w)],
        output_dtypes=[mx.float32],
    )[0]


def bicubic(
    x: mx.array, size: Sequence[int], mean: Sequence[float] | None = None, std: Sequence[float] | None = None
) -> mx.array:
    """`F.interpolate(x, size, mode="bicubic")` of `[N, C, H, W]` (float32), then `(x - mean) / std` if given."""
    taps = (cubic_taps(x.shape[2], int(size[0])), cubic_taps(x.shape[3], int(size[1])))
    norm = mean is not None and std is not None
    m = mx.array(np.asarray(mean if norm else [0.0], dtype=np.float32))
    s = mx.array(np.asarray(std if norm else [1.0], dtype=np.float32))
    return _run(_BICUBIC, x, size, taps, [m, s], [("NORM", norm)])


def bilinear_aligned(x: mx.array, size: Sequence[int]) -> mx.array:
    """`F.interpolate(x, size, mode="bilinear", align_corners=True)` of `[N, C, H, W]` (float32)."""
    taps = (linear_taps_aligned(x.shape[2], int(size[0])), linear_taps_aligned(x.shape[3], int(size[1])))
    return _run(_BILINEAR, x, size, taps, [], [("SMALL", int(size[0]) + int(size[1]) <= _SMALL)])
