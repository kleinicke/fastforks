"""Apple MPS specifics: the transposed convolutions `adapt` replaces, a clean interpreter exit."""

from __future__ import annotations

import copy
import subprocess
import sys

import pytest
import torch

from fastforks._core import mps

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS"),
]

# a NumPy pair runs last, so its MPS work is still queued when the script ends
_EXIT = """
import numpy as np, torch
from fastforks import stereo
m = stereo.load("hitnet-eth3d", device="mps")
x = torch.rand(1, 3, 480, 640)
m(x, x)
y = np.zeros((480, 640, 3), np.uint8)
m(y, y)
"""


@pytest.mark.parametrize(
    ("kernel", "stride", "padding", "output_padding"),
    [(4, 2, 1, 0), (3, 2, 1, 1), (3, 2, 1, 0), (2, 2, 0, 0), (5, 2, 2, 1), (3, 1, 1, 0)],  # first two: the families
)
def test_float16_transposed_conv3d(kernel: int, stride: int, padding: int, output_padding: int) -> None:
    torch.manual_seed(0)
    conv = torch.nn.ConvTranspose3d(8, 4, kernel, stride, padding, output_padding).to("mps")
    x = torch.randn(2, 8, 5, 1, 7, device="mps")
    want = conv(x)
    mps.adapt(conv)
    got = conv.half()(x.half())
    assert got.dtype == torch.float16
    torch.testing.assert_close(got.float(), want, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize(
    ("kernel", "stride", "padding", "output_padding"),
    [(4, 2, 1, 0), (3, 2, 1, 1), (1, 2, 0, 0), (2, 2, 0, 0), (4, 4, 0, 0), (3, 1, 1, 0)],  # polyphase, pixel shuffle
)
def test_transposed_conv2d(kernel: int, stride: int, padding: int, output_padding: int) -> None:
    torch.manual_seed(0)
    conv = torch.nn.ConvTranspose2d(8, 4, kernel, stride, padding, output_padding).to("mps")
    x = torch.randn(2, 8, 5, 7, device="mps")
    want = copy.deepcopy(conv).cpu().double()(x.cpu().double())
    mps.adapt(conv)
    torch.testing.assert_close(conv(x).cpu().double(), want, atol=1e-4, rtol=1e-4)
    # an odd `output_size`: per-axis output padding
    if stride == 2 and kernel > 1:
        size = [4 * 2 - 2 * padding + kernel, 6 * 2 - 2 * padding + kernel + 1]  # output padding 0, 1
        want = torch.nn.functional.conv_transpose2d(
            x.cpu().double(), conv.weight.cpu().double(), conv.bias.cpu().double(), 2, padding, [0, 1]
        )
        torch.testing.assert_close(conv(x, size).cpu().double(), want, atol=1e-4, rtol=1e-4)


@pytest.mark.model
def test_exit_with_queued_work():
    """Queued MPS work holding NumPy memory is finished before shutdown (without: abort 134 or a hang, torch 2.14)."""
    from fastforks import stereo  # noqa: PLC0415

    stereo.load("hitnet-eth3d", device="cpu")  # the download (a slow host) outside the timed exits
    for _ in range(3):
        p = subprocess.run([sys.executable, "-c", _EXIT], capture_output=True, text=True, timeout=120, check=False)
        assert p.returncode == 0, p.stderr[-2000:]
