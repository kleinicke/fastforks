"""C-FS-S kernels: the batched-matmul cost volume (`_cost_volume`) matches TAO's per-disparity loops."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from fastforks.stereo.models.c_foundationstereo.kernels import inference


def _loops(fl: torch.Tensor, fr: torch.Tensor, d: int, groups: int, *, normalize: bool) -> torch.Tensor:
    """TAO: group-wise correlation and the concatenated features per disparity, zero where x < disparity."""
    b, c, h, w = fl.shape
    a = fl.reshape(b, groups, c // groups, h, w)
    z = fr.reshape(b, groups, c // groups, h, w)
    if normalize:
        a, z = F.normalize(a, dim=2), F.normalize(z, dim=2)
    gwc = torch.zeros(b, groups, d, h, w)
    cat = torch.zeros(b, 2 * c, d, h, w)
    for i in range(min(d, w)):
        gwc[:, :, i, :, i:] = (a[..., i:] * z[..., : w - i]).sum(2)
        cat[:, :c, i] = fl
        cat[:, c:, i, :, i:] = fr[..., : w - i]
    for i in range(min(d, w), d):
        cat[:, :c, i] = fl
    return torch.cat([gwc, cat], 1)


@pytest.mark.parametrize(
    ("shape", "d", "groups", "normalize"),
    [
        ((1, 16, 21, 30), 12, 4, True),  # H not a multiple of the row block
        ((2, 8, 5, 9), 12, 2, False),  # batch 2, more disparities than columns
    ],
)
def test_cost_volume_matches_loops(shape: tuple[int, int, int, int], d: int, groups: int, normalize: bool) -> None:  # noqa: FBT001
    torch.manual_seed(0)
    fl, fr = torch.randn(shape), torch.randn(shape)
    got = inference._cost_volume(fl, fr, nn.Identity(), d, groups, normalize=normalize)
    want = _loops(fl, fr, d, groups, normalize=normalize)
    assert got.shape == want.shape
    assert got.dtype == torch.float16
    torch.testing.assert_close(got.float(), want, atol=2e-3, rtol=1e-3)
