"""`fastforks._core.decoder_ops`: the fused decoder steps against plain PyTorch, eagerly and on the MLX replay."""

import pytest
import torch
import torch.nn.functional as F

from fastforks._core import decoder_ops


def _maps(c=6, k=3, h=5, w=7):
    g = torch.Generator().manual_seed(c * h + w)
    x = torch.randn(2, c, h, w, generator=g)
    skip = torch.randn(2, c, 2 * h, 2 * w, generator=g)
    src = torch.randn(2, k, 2 * h, 2 * w, generator=g)
    weight, bias = torch.randn(c, k, generator=g), torch.randn(c, generator=g)
    return x, skip, src, weight, bias


def test_eager_forms():
    x, skip, src, weight, bias = _maps()
    up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    torch.testing.assert_close(decoder_ops.up2_skip(x, skip, relu=True), F.relu(up + skip))
    proj = up + F.conv2d(src, weight[:, :, None, None], bias)
    torch.testing.assert_close(decoder_ops.up2_proj(x, src, weight, bias), proj)
    torch.testing.assert_close(decoder_ops.up2_proj(x, src, weight, bias, relu=True), F.relu(proj))
    big = torch.randn(2, 5, 16, 24)
    torch.testing.assert_close(
        decoder_ops.area_pool(big, 4), F.interpolate(big, scale_factor=1 / 4, mode="area"), atol=1e-6, rtol=1e-5
    )


def test_matanyone_mlx_decoder_matches_upstream():
    """`_mlx_decode` (4-D, fused, the residual projection moved to half size) is upstream's decoder up to rounding."""
    from fastforks.matting.models.matanyone import model as mat  # noqa: PLC0415

    torch.manual_seed(0)
    dec = mat.MaskDecoder(mat._config(mat.CONFIG).model).eval()
    h, w = 32, 48
    dims = mat.CONFIG["model"]["pixel_encoder"]["ms_dims"]
    feats = [torch.randn(1, d, h >> s, w >> s) for d, s in zip(dims, (4, 3, 2, 1, 0), strict=True)]
    readout, sensory = torch.randn(1, 1, 256, h // 16, w // 16), torch.randn(1, 1, 256, h // 16, w // 16)
    last = torch.rand(1, 1, h, w)
    for kwargs in ({}, {"seg_pass": True}, {"last_mask": last, "sigmoid_residual": True}, {"update_sensory": False}):
        with torch.inference_mode():
            want = dec(feats, readout, sensory, **kwargs)
            got = mat._mlx_decode(dec, feats, readout, sensory, **kwargs)
        for a, b in zip(got, want, strict=True):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-4)


@pytest.mark.mlx
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_mlx_kernels(dtype):
    mx = pytest.importorskip("mlx.core")
    from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415
    from fastforks.shared import mlx_graph  # noqa: PLC0415

    dt = getattr(mx, dtype)
    tol = 2e-2 if dtype == "float16" else 1e-5
    ops = torch.ops.fastforks
    for c in (6, 40):  # below and above one thread group of channels
        x, skip, src, weight, bias = _maps(c)
        mx_x, mx_skip = to_mx(x).astype(dt), to_mx(skip).astype(dt)
        for relu in (False, True):
            y = mlx_graph.OPS[ops.up2_skip.default](mx_x, mx_skip, relu)
            assert y.dtype == dt
            torch.testing.assert_close(to_torch(y), decoder_ops.up2_skip(x, skip, relu=relu), atol=tol, rtol=tol)
            y = mlx_graph.OPS[ops.up2_proj.default](mx_x, to_mx(src), to_mx(weight), to_mx(bias), relu)
            assert y.dtype == dt
            want = decoder_ops.up2_proj(x, src, weight, bias, relu=relu)
            torch.testing.assert_close(to_torch(y), want, atol=tol, rtol=tol)
    big = torch.randn(2, 40, 16, 24)
    y = mlx_graph.CL_OPS[ops.area_pool.default](to_mx(big).astype(dt), 4)
    assert y.dtype == dt
    torch.testing.assert_close(to_torch(y), decoder_ops.area_pool(big, 4), atol=tol / 10, rtol=tol / 10)
