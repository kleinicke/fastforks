"""DRCT shifted-window bias sharing at rectangular, singleton and batched window grids."""

from __future__ import annotations

import warnings

import pytest
import torch

from fastforks.restoration.models.drct.model import _attention, _boundary_attention
from fastforks.restoration.models.drct.net.drct_arch import WindowAttention
from fastforks.shared.swin_sr import relative_bias, shift_mask


@pytest.mark.parametrize("size", [(4, 4), (12, 4), (4, 16), (12, 16)])
def test_boundary_attention(size: tuple[int, int]) -> None:
    torch.manual_seed(923)
    win, shift, batch, c = 4, 2, 2, 24
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "torch.meshgrid", UserWarning)
        attn = WindowAttention(c, (win, win), 3).eval()
    attn.register_buffer("bias", relative_bias(attn.relative_position_bias_table, attn.relative_position_index))
    windows = torch.randn(batch * size[0] * size[1] // win**2, win**2, c)
    full = shift_mask(*size, win, shift, windows.device)
    torch.testing.assert_close(_boundary_attention(attn, windows, size, win, shift), _attention(attn, windows, full))
