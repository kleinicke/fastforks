"""`cnn_kernels.reflect_pad`: torch's reflection pad as channels-last, and the patched `ReflectionPad2d` (CPU)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from fastforks.shared import cnn_kernels


@pytest.mark.parametrize("pad", [(1, 1, 1, 1), (3, 3, 3, 3), (0, 2, 1, 0), (2, 0, 0, 3)])
def test_reflect_pad_matches_torch(pad: tuple[int, int, int, int]) -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 16, 9, 11, generator=g).contiguous(memory_format=torch.channels_last)
    y = cnn_kernels.reflect_pad(x, pad)
    assert y.is_contiguous(memory_format=torch.channels_last)
    assert torch.equal(y, F.pad(x, pad, mode="reflect"))


def test_patched_reflection_pad() -> None:
    net = nn.Sequential(nn.ReflectionPad2d(2), nn.ReflectionPad2d((1, 0, 2, 1)))
    ref = nn.Sequential(nn.ReflectionPad2d(2), nn.ReflectionPad2d((1, 0, 2, 1)))
    cnn_kernels.patch_reflection_pads(net)
    g = torch.Generator().manual_seed(1)
    wide = torch.randn(1, 8, 7, 6, generator=g).contiguous(memory_format=torch.channels_last)
    narrow = torch.randn(1, 3, 7, 6, generator=g).contiguous(memory_format=torch.channels_last)  # torch's kernel
    for x in (wide, narrow, wide.contiguous()):
        assert torch.equal(net(x), ref(x))
    assert net(wide).is_contiguous(memory_format=torch.channels_last)
