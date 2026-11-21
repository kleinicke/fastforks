"""`conv_mlx` against MLX references: the direct few-channel conv (random shapes, act) and the Winograd fp16 range."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from fastforks.kernels.conv_mlx import ops  # noqa: E402

pytestmark = pytest.mark.mlx


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize(
    ("n", "h", "w", "c", "co", "k", "s", "d", "act"),
    [
        (1, 37, 53, 3, 16, 3, 2, 1, 0),  # image stem, odd channels
        (2, 40, 31, 16, 16, 3, 1, 1, 1),
        (1, 33, 47, 28, 32, 3, 2, 1, 2),
        (1, 29, 30, 15, 12, 3, 1, 1, 0),  # odd C, CO % 8 != 0
        (1, 26, 34, 6, 8, 2, 2, 1, 1),  # CV 2, even kernel
        (1, 30, 30, 8, 16, 3, 1, 2, 2),  # dilation
        (1, 21, 19, 4, 4, 3, 1, 1, 0),
    ],
)
def test_direct_matches_conv2d(dtype, n, h, w, c, co, k, s, d, act):  # noqa: PLR0913
    rng = np.random.default_rng(h * w + c)
    dt = getattr(mx, dtype)
    x = mx.array(rng.standard_normal((n, h, w, c)).astype(np.float32))
    wt = mx.array((rng.standard_normal((co, k, k, c)) / np.sqrt(c * k * k)).astype(np.float32))
    b = mx.array(rng.standard_normal(co).astype(np.float32))
    slope = mx.array(rng.uniform(0.0, 0.3, co).astype(np.float32))
    p = d * (k - 1) // 2
    ref = mx.conv2d(x, wt, stride=s, padding=p, dilation=d) + b
    if act == 1:
        ref = mx.maximum(ref, 0)
    elif act == 2:
        ref = mx.where(ref > 0, ref, ref * slope)
    y = ops.direct(x.astype(dt), wt.astype(dt), b.astype(dt), (s, s), (p, p), (d, d), act=act, slope=slope)
    assert y.shape == ref.shape
    assert y.dtype == dt
    tol = 2e-2 if dtype == "float16" else 1e-4
    np.testing.assert_allclose(np.array(y.astype(mx.float32)), np.array(ref), atol=tol, rtol=tol)


def test_direct_supported():
    assert ops.direct_supported(3, 16, (3, 3), (2, 2))
    assert ops.direct_supported(16, 32, (3, 3), (1, 1))
    assert not ops.direct_supported(32, 32, (3, 3), (1, 1))  # ties mx.conv2d
    assert not ops.direct_supported(64, 64, (3, 3), (2, 2))
    assert not ops.direct_supported(3, 64, (7, 7), (2, 2))
    assert not ops.direct_supported(16, 16, (1, 1), (1, 1))
    assert not ops.direct_supported(16, 18, (3, 3), (1, 1))


@pytest.mark.parametrize("m", [2, 4])
def test_winograd_float16_range(m):
    """Large float16 activations (|x| ~ 2000) stay finite: the transforms store pre-scaled values."""
    rng = np.random.default_rng(m)
    x = mx.array((2000 * rng.standard_normal((1, 20, 24, 32))).astype(np.float32))
    w = mx.array((rng.standard_normal((32, 3, 3, 32)) / np.sqrt(9 * 32)).astype(np.float32))
    ref = np.array(mx.conv2d(x, w, padding=1))
    y = ops.conv3x3(x.astype(mx.float16), ops.weights(w, m, mx.float16), m=m)
    got = np.array(y.astype(mx.float32))
    assert np.isfinite(got).all()
    # F(4,3) in float16 is ~1.3 % of max off at any magnitude (unscaled it was NaN here), F(2,3) ~0.1 %
    np.testing.assert_allclose(got, ref, atol=(0.025 if m == 4 else 0.003) * np.abs(ref).max())


def test_winograd3d_float16_range():
    rng = np.random.default_rng(3)
    x = mx.array((2000 * rng.standard_normal((1, 6, 12, 16, 16))).astype(np.float32))
    w = mx.array((rng.standard_normal((16, 3, 3, 3, 16)) / np.sqrt(27 * 16)).astype(np.float32))
    ref = np.array(mx.conv3d(x, w, padding=1))
    got = np.array(ops.conv3d(x.astype(mx.float16), ops.weights3d(w, 4, mx.float16)).astype(mx.float32))
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, ref, atol=0.025 * np.abs(ref).max())


def test_depthwise_mixed_dtypes():
    """A float32 activation against float16 weights (FoundationStereo's float16 depthwise after a float32 norm)."""
    from fastforks.kernels.graph_mlx import ops as graph_ops  # noqa: PLC0415

    rng = np.random.default_rng(5)
    x = mx.array(rng.standard_normal((1, 12, 16, 8)).astype(np.float32))
    w = mx.array(rng.standard_normal((7, 7, 8)).astype(np.float16))
    b = mx.array(rng.standard_normal(8).astype(np.float16))
    ref = mx.conv2d(x, w.astype(mx.float32).transpose(2, 0, 1)[..., None], padding=3, groups=8) + b.astype(mx.float32)
    got = graph_ops.depthwise(x, w, b, (1, 1), (3, 3), (1, 1))
    assert got.dtype == mx.float32
    np.testing.assert_allclose(np.array(got), np.array(ref), atol=1e-4)
