"""Selective RT-MonSter++ stereo warps, including the upstream grid convention and zero border."""

import pytest
import torch

from fastforks.kernels import monster_gwc
from fastforks.stereo.models.rt_monster_plusplus.net.submodule import build_gwc_volume_selective


@pytest.mark.gpu
@pytest.mark.parametrize(("batch", "channels", "groups"), [(1, 32, 8), (2, 96, 8)])
def test_selective_groupwise_volume(batch: int, channels: int, groups: int) -> None:
    depth, h, w = 9, 13, 29
    left = torch.randn(batch, channels, h, w, device="cuda", dtype=torch.float16).contiguous(
        memory_format=torch.channels_last
    )
    right = torch.randn_like(left)
    samples = torch.randn(batch, depth, h, w, device="cuda") * 5
    samples[:, 0] = 0
    samples[:, 1] = w
    actual = monster_gwc.volume(left, right, samples, depth, groups)
    expected = build_gwc_volume_selective(left.float(), right.float(), samples, depth, groups).half()
    torch.testing.assert_close(actual, expected)
    assert actual[:, :, 1].count_nonzero().item() == 0
