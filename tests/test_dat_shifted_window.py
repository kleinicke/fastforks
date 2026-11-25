"""Compact DAT mask parity on batches and grids with a single row or column."""

from __future__ import annotations

import pytest
import torch

from fastforks.restoration.models.dat.model import _masks, _shifted_window, _spatial
from fastforks.restoration.models.dat.net.dat_arch import Spatial_Attention


@pytest.mark.parametrize(("h", "w"), [(4, 4), (4, 12), (12, 4), (12, 12)])
@pytest.mark.parametrize("branch", [0, 1])
def test_dat_shared_mask_matches_full_grid(h: int, w: int, branch: int) -> None:
    """Each window attends only over the same tokens as the original dense mask; the rolls are folded in."""
    generator = torch.Generator().manual_seed(0)
    split = (2, 4)
    shift = (1, 2)
    attn = Spatial_Attention(dim=16, idx=branch, split_size=split, num_heads=2, position_bias=False)
    n = split[0] * split[1]
    attn.register_buffer("bias", torch.randn(2, n, n, generator=generator))
    qkv = torch.randn(3, 2, h * w, 16, generator=generator)
    masks = _masks(split, shift, h, w, torch.device("cpu"))
    want = _spatial(attn, qkv, h, w, masks[branch])
    sh = shift if branch == 0 else (shift[1], shift[0])
    image = torch.roll(qkv.view(3, 2, h, w, 16), sh, (2, 3))  # the unrolled input: its roll by -sh is qkv
    got = _shifted_window(attn, image, h, w, sh, into=torch.empty(2, h, w, 16))
    torch.testing.assert_close(got, torch.roll(want, sh, (1, 2)))
