"""Numerical and memory-layout contracts of the fused PointDiT attention input."""

import pytest
import torch

from fastforks._core.kernels import linear
from fastforks.depth.models.pointdit.net import rms
from fastforks.kernels import fastbb
from fastforks.kernels.pointdit_norm import normalize_rope


@pytest.mark.gpu
@pytest.mark.parametrize(("b", "n", "h"), [(2, 19, 12), (1, 1024, 16)])
def test_qk_norm_rope(b, n, h):
    """Match the previous fp32 RMS, intermediate fp16 rounding and packed rotation; V is untouched."""
    gen = torch.Generator(device="cuda").manual_seed(161)
    qkv = torch.randn(b, n, 3, h, 64, device="cuda", dtype=torch.float16, generator=gen)
    weight = torch.randn(2, 1, 64, device="cuda", generator=gen)
    angles = torch.randn(n, 32, device="cuda", generator=gen)
    table = torch.stack((angles.cos(), angles.sin()), 1)
    expected = qkv.clone()
    qk = expected[:, :, :2]
    qk.copy_(rms(qk, weight))
    total = b * n * 2 * h * 16
    fastbb.load(qkv.device).launch("rope_qk_h", *linear(total), expected, table, n, 0, h, total, 0.125)
    normalize_rope(qkv, weight, table)
    torch.testing.assert_close(qkv, expected, atol=0, rtol=0)
