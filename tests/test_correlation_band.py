"""`correlation_band` against the slab loop it replaces (`gwc_volume`)."""

import pytest
import torch

from fastforks._core.volume_ops import gwc_volume
from fastforks.stereo import _volume


@pytest.mark.parametrize(("shape", "groups", "depth"), [((1, 16, 5, 9), 1, 5), ((2, 12, 5, 7), 3, 4)])
@pytest.mark.parametrize("rows_per_gemm", [None, 2])  # 2: the chunk loop and its offsets
def test_correlation_band_matches_the_slab_loop(
    shape: tuple[int, int, int, int],
    groups: int,
    depth: int,
    rows_per_gemm: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if rows_per_gemm is not None:
        monkeypatch.setattr(_volume, "_BAND_BYTES", rows_per_gemm * (depth + shape[3]) * shape[3] * 4)
    g = torch.Generator().manual_seed(0)
    left = torch.randn(shape, generator=g, dtype=torch.float64).float()
    right = torch.randn(shape, generator=g, dtype=torch.float64).float()
    got = _volume.correlation_band(left, right, depth, groups)
    torch.testing.assert_close(got, gwc_volume(left, right, depth, groups), rtol=1e-5, atol=1e-6)


def test_correlation_band_disparities_beyond_the_width_are_zero() -> None:
    left, right = torch.randn(1, 4, 3, 3), torch.randn(1, 4, 3, 3)
    got = _volume.correlation_band(left, right, 5)
    assert got[:, :, 3:].abs().max() == 0
    torch.testing.assert_close(got, gwc_volume(left, right, 5, 1))
