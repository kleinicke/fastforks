"""The `stereo_ops` kernels against their PyTorch definitions (GPU; odd sizes, batch 2)."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as F

from fastforks.stereo.models.las2 import kernels as las2_kernels
from fastforks.stereo.models.las2.net.aggregation_fasternet import AttentionModule
from fastforks.stereo.models.las2.net.submodule import build_correlation_volume

pytestmark = pytest.mark.gpu
CL = torch.channels_last


@pytest.fixture
def dev():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    return torch.device("cuda", torch.cuda.current_device())


@pytest.mark.parametrize(("c", "h", "w", "d"), [(40, 5, 77, 48), (8, 3, 31, 5), (70, 2, 40, 64)])
def test_corr_volume(dev, c, h, w, d):
    k = las2_kernels.stereo_ops.load(dev)
    fl, fr = (torch.randn(2, c, h, w, device=dev).half().contiguous(memory_format=CL) for _ in range(2))
    want = build_correlation_volume(fl.float(), fr.float(), d)
    got = las2_kernels.corr_volume(k, fl, fr, d)
    assert got.is_contiguous(memory_format=CL)
    torch.testing.assert_close(got.float(), want, atol=2e-3, rtol=2e-3)


def test_attention(dev):
    torch.manual_seed(0)
    m: Any = AttentionModule(24, 16).to(dev).eval()
    cost = torch.randn(2, 24, 9, 37, device=dev)
    x = torch.randn(2, 16, 9, 37, device=dev)
    with torch.no_grad():
        want = m(cost, x)
        las2_kernels.prepare(m, dev)
        m.half()
        m.conv0.forward = lambda t: F.conv2d(t, m.conv0.weight, m.conv0.bias).contiguous(memory_format=CL)
        m.conv3.forward = lambda t: F.conv2d(t, m.conv3.weight, m.conv3.bias)
        got = m(cost.half().contiguous(memory_format=CL), x.half().contiguous(memory_format=CL))
    torch.testing.assert_close(got.float(), want, atol=0.05, rtol=0.02)


def test_coex_cost_volume(dev):
    from fastforks.stereo.models.coex.model import _cost_volume_forward  # noqa: PLC0415
    from fastforks.stereo.models.coex.net.coex import CostVolume  # noqa: PLC0415

    cv: Any = CostVolume(47)
    x, y = (torch.randn(2, 16, 5, 61, device=dev).half().contiguous(memory_format=CL) for _ in range(2))
    want = cv(x.float(), y.float())
    cv._ops = las2_kernels.stereo_ops.load(dev)
    got = _cost_volume_forward(cv, x, y)
    assert got.shape == want.shape
    torch.testing.assert_close(got.float(), want, atol=1e-2, rtol=2e-3)


@pytest.mark.parametrize(("d", "topk"), [(48, 0), (5, 0), (48, 2), (7, 3), (3, 8)])
def test_soft_argmax(dev, d, topk):
    from fastforks.stereo.models.las2.net.submodule import disparity_regression  # noqa: PLC0415

    k = las2_kernels.stereo_ops.load(dev)
    ranks = torch.rand(2, 1, d, 7, 33, device=dev).argsort(2).float()  # distinct costs: no top-k ties
    cost = (ranks * 0.37 - 0.2 * d).half()[:, 0]  # a strided [N, D, H, W] view
    got = las2_kernels.stereo_ops.soft_argmax(k, cost, topk)
    c = cost.float()
    if topk:
        val, ind = c.topk(min(topk, d), 1)
        want = (F.softmax(val, 1) * ind).sum(1, keepdim=True)
    else:
        want = disparity_regression(F.softmax(c, 1), d)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize(("h", "w", "f"), [(5, 9, 4), (3, 7, 2), (1, 1, 4)])
def test_spx_upsample(dev, h, w, f):
    from fastforks.stereo.models.las2.net.submodule import context_upsample  # noqa: PLC0415

    k = las2_kernels.stereo_ops.load(dev)
    low = torch.randn(2, 1, h, w, device=dev) * 10
    logits = torch.randn(2, 16, h * f, w * f, device=dev).half().contiguous(memory_format=CL)[:, :9]  # a slice
    got = las2_kernels.stereo_ops.spx_upsample(k, low, logits, 4.0)
    if f == 4:
        want = context_upsample(low * 4.0, F.softmax(logits.float(), 1))[:, 0]
    else:
        unf = F.unfold(low, 3, 1, 1).reshape(2, 9, h, w)
        want = 4.0 * (F.interpolate(unf, (h * f, w * f), mode="nearest") * F.softmax(logits.float(), 1)).sum(1)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize(("c", "kk", "axis"), [(16, 7, 0), (16, 21, 1), (5, 11, 0), (5, 3, 1)])
def test_dw_line(dev, c, kk, axis):
    k = las2_kernels.stereo_ops.load(dev)
    torch.manual_seed(0)
    x = torch.randn(2, c, 9, 13, device=dev).half().contiguous(memory_format=CL)
    res = torch.randn_like(x).contiguous(memory_format=CL)
    wt, b = torch.randn(c, kk, device=dev), torch.randn(c, device=dev)
    shape = (c, 1, 1, kk) if axis == 0 else (c, 1, kk, 1)
    want = F.conv2d(x.float(), wt.view(shape), b, padding=(0, kk // 2) if axis == 0 else (kk // 2, 0), groups=c)
    want = want + res.float()
    got = las2_kernels.stereo_ops.dw_line(k, x, wt.t().contiguous(), b, axis, res=res, out=res)  # in place
    assert got.data_ptr() == res.data_ptr()
    torch.testing.assert_close(got.float(), want, atol=1e-2, rtol=2e-3)


@pytest.mark.parametrize("fill", ["replicate", "zero"])
@pytest.mark.parametrize(
    ("n", "h", "w", "pads"), [(1, 5, 9, (0, 0, 0, 0)), (2, 3, 7, (1, 2, 3, 0)), (1, 4, 4, (0, 5, 0, 6))]
)
def test_norm_pad_pair(dev, n, h, w, pads, fill):  # noqa: PLR0913, PLR0917 - parametrised
    k = las2_kernels.stereo_ops.load(dev)
    torch.manual_seed(0)
    left, right = (torch.rand(n, 3, h, w, device=dev) for _ in range(2))
    mean, std = torch.rand(3, device=dev), torch.rand(3, device=dev) + 0.5
    ms = torch.stack((mean, std))
    got = las2_kernels.stereo_ops.norm_pad_pair(k, left, right, ms, pads=pads, mul=255.0, fill=fill)
    x = (torch.cat((left, right)) * 255.0 - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
    want = F.pad(x, pads, mode="replicate") if fill == "replicate" else F.pad(x, pads)
    assert got.is_contiguous(memory_format=CL)
    torch.testing.assert_close(got.float(), want, atol=0.2, rtol=1e-3)


def test_corr_volume_batch_split(dev, monkeypatch):
    """`N * H` above the grid limit: one launch per image, same result."""
    k = las2_kernels.stereo_ops.load(dev)
    fl, fr = (torch.randn(3, 16, 5, 33, device=dev).half().contiguous(memory_format=CL) for _ in range(2))
    want = las2_kernels.corr_volume(k, fl, fr, 12)
    monkeypatch.setattr(las2_kernels.stereo_ops, "_MAX_ROWS", 6)
    torch.testing.assert_close(las2_kernels.corr_volume(k, fl, fr, 12), want)


def test_soft_argmax_needs_disparities(dev):
    k = las2_kernels.stereo_ops.load(dev)
    with pytest.raises(RuntimeError, match="D >= 1"):
        las2_kernels.stereo_ops.soft_argmax(k, torch.zeros(1, 0, 2, 2, device=dev, dtype=torch.float16))
