"""`native_depthwise`: the depthwise layers only, same output as the cuDNN / reference convolution."""

from __future__ import annotations

import pytest
import torch

from fastforks.shared.native_depthwise import native_depthwise


def test_only_depthwise_layers_are_swapped() -> None:
    net = torch.nn.Sequential(
        torch.nn.Conv2d(8, 8, 3, padding=1, groups=8),
        torch.nn.Conv2d(8, 16, 1),
        torch.nn.Conv2d(16, 16, 3, padding="same", groups=16),
        torch.nn.Conv2d(16, 16, 3, padding=1, groups=4),
    )
    assert native_depthwise(net) == 1
    assert type(net[0]) is not torch.nn.Conv2d  # exact type: overridden
    assert all(type(m) is torch.nn.Conv2d for m in net[1:])  # exact type: untouched


@pytest.mark.gpu
@pytest.mark.parametrize("stride", [1, 2, (2, 1)])
def test_output_matches_the_convolution(stride: int | tuple[int, int]) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA")
    conv = torch.nn.Conv2d(32, 32, 5, stride, 2, groups=32).cuda().half()
    x = torch.randn(2, 32, 40, 56, device="cuda", dtype=torch.float16)
    expected = torch.nn.functional.conv2d(x.float(), conv.weight.float(), conv.bias.float(), stride, 2, 1, 32)
    native_depthwise(conv)
    torch.testing.assert_close(conv(x).float(), expected, atol=2e-2, rtol=1e-2)
