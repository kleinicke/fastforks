"""Compare the super-resolution fused operations with their PyTorch fp16 evaluation order."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fastforks.kernels import sr_ops

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("learned", [False, True])
def test_channel_slice_activation(*, learned: bool) -> None:
    g = torch.Generator(device="cuda").manual_seed(2201)
    storage = torch.randn(2, 48, 11, 13, device="cuda", dtype=torch.float16, generator=g).to(
        memory_format=torch.channels_last
    )
    original = storage.clone()
    x = storage[:, 16:35]
    bias = torch.randn(19, device="cuda", dtype=x.dtype, generator=g)
    slope = torch.randn(19, device="cuda", dtype=x.dtype, generator=g) if learned else None
    value = x + bias[None, :, None, None]
    expected = F.prelu(value, slope) if slope is not None else F.leaky_relu(value, 0.2)
    got = sr_ops.bias_activation(x, bias, slope)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    torch.testing.assert_close(storage[:, :16], original[:, :16], rtol=0, atol=0)
    torch.testing.assert_close(storage[:, 35:], original[:, 35:], rtol=0, atol=0)


@pytest.mark.parametrize("attention", [False, True])
def test_residual(*, attention: bool) -> None:
    g = torch.Generator(device="cuda").manual_seed(40)
    branch = torch.randn(2, 19, 7, 11, device="cuda", dtype=torch.float16, generator=g).to(
        memory_format=torch.channels_last
    )
    skip = torch.randn_like(branch)
    expected = (branch + skip) * (branch.sigmoid() - 0.5) if attention else branch * 0.2 + skip
    torch.testing.assert_close(sr_ops.residual(branch, skip, attention=attention), expected, rtol=0, atol=0)


@pytest.mark.parametrize("scale", [2, 4])
def test_packed_shuffle_residual(scale: int) -> None:
    g = torch.Generator(device="cuda").manual_seed(23)
    low = torch.randn(2, 3 * scale**2, 9, 7, device="cuda", dtype=torch.float16, generator=g).to(
        memory_format=torch.channels_last
    )
    skip = torch.randn(2, 3, 9, 7, device="cuda", dtype=low.dtype, generator=g).to(memory_format=torch.channels_last)
    expected = F.pixel_shuffle(low, scale) + F.interpolate(skip, scale_factor=scale, mode="nearest")
    torch.testing.assert_close(sr_ops.shuffle_residual(low, skip, scale), expected, rtol=0, atol=0)
