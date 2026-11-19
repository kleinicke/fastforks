"""DRCT shifted-window bias sharing and strip-wise reconstruction."""

from __future__ import annotations

import warnings

import pytest
import torch

from fastforks.restoration.models.drct import model as drct
from fastforks.restoration.models.drct.net import DRCT
from fastforks.restoration.models.drct.net.drct_arch import WindowAttention
from fastforks.shared.swin_sr import relative_bias, shift_mask


@pytest.mark.parametrize("size", [(4, 4), (12, 4), (4, 16), (12, 16)])
def test_boundary_attention(size: tuple[int, int], monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(923)
    win, shift, batch, c = 4, 2, 2, 24
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "torch.meshgrid", UserWarning)
        attn = WindowAttention(c, (win, win), 3).eval()
    attn.register_buffer("bias", relative_bias(attn.relative_position_bias_table, attn.relative_position_index))
    windows = torch.randn(batch * size[0] * size[1] // win**2, win**2, c)
    full = shift_mask(*size, win, shift, windows.device)
    expected = drct._attention(attn, windows, full)
    torch.testing.assert_close(drct._boundary_attention(attn, windows, size, win, shift), expected)
    monkeypatch.setattr(drct, "_WORKSPACE_BYTES", 5 * 1024)
    geometry = drct._BoundaryMask(size, win, shift)
    torch.testing.assert_close(drct._attention(attn, windows, geometry), expected)


@torch.inference_mode()
def test_reconstruction_strip_boundaries() -> None:
    torch.manual_seed(923)
    net = DRCT(img_size=8, embed_dim=12, depths=[], window_size=4, upsampler="pixelshuffle", upscale=4).eval()
    feature = torch.randn(2, net.conv_last.in_channels, 13, 19)
    expected = net.conv_last(net.upsample(feature))
    actual = drct._strip_reconstruction(net, feature, rows=4)
    torch.testing.assert_close(actual, expected)
