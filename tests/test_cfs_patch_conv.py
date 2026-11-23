"""C-FS-S kernels: the slab-wise patch conv (`_patch_conv`) gives cuDNN's channels-last result bit for bit."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from fastforks.stereo.models.c_foundationstereo.kernels import inference

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    ("n", "d", "h", "w", "padding"),
    [
        (1, 40, 32, 48, (1, 1, 0)),  # C-FS-S's `conv_patch`: padded disparities and rows
        (2, 36, 20, 28, (0, 0, 0)),  # batch 2, disparities not a multiple of the slab
        (1, 13, 17, 22, (0, 1, 1)),  # sizes not a multiple of the kernel
    ],
)
def test_patch_conv_matches_cudnn(n: int, d: int, h: int, w: int, padding: tuple[int, int, int]) -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    torch.manual_seed(0)
    c = 28
    m = nn.Conv3d(c, c, 4, stride=4, padding=padding, groups=c).cuda().half()
    buf = torch.randn(n, 32, d, h, w, device="cuda").half().contiguous(memory_format=torch.channels_last_3d)
    x = buf[:, :c]  # 28 channels of a 32-channel buffer, as the network hands it over
    y = inference._patch_conv(m, x)
    assert y is not None
    ref = F.conv3d(x, m.weight, None, m.stride, m.padding, m.dilation, m.groups)
    assert y.shape == ref.shape
    assert y.is_contiguous(memory_format=torch.channels_last_3d)
    assert torch.equal(y, ref)


def test_patch_conv_declines_other_convs() -> None:
    m = nn.Conv3d(8, 8, 3, stride=1, padding=1, groups=8).half()
    assert inference._patch_conv(m, torch.zeros(1, 8, 4, 4, 4, dtype=torch.float16)) is None
    m = nn.Conv3d(8, 8, 4, stride=4, groups=8).half()
    assert inference._patch_conv(m, torch.zeros(1, 8, 8, 8, 8, dtype=torch.float16)) is None  # channels-first
