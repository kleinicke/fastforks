"""MVSplat camera geometry."""

import pytest
import torch

from fastforks.multiview._runner import project_harmonics
from fastforks.multiview.models.mvsplat.model import _poses


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_relative_poses_match_matrix_inverse(dtype: torch.dtype) -> None:
    c2w = torch.eye(4, dtype=dtype).repeat(3, 1, 1)
    c2w[:, :3, :3] = torch.tensor([[[1.0, 0.1, 0.0], [0.0, 0.9, 0.2], [0.0, 0.0, 1.1]]] * 3, dtype=dtype)
    c2w[:, :3, 3] = torch.tensor([[0, 0, 0], [1, 2, 3], [-2, 4, 1]], dtype=dtype)
    actual = _poses(c2w)
    assert len(actual) == 2
    for offset, pose in enumerate(actual, 1):
        expected = torch.linalg.inv(c2w).roll(-offset, 0) @ c2w
        torch.testing.assert_close(pose, expected)


@pytest.mark.parametrize("channels", [3, 5, 7, 9])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_harmonic_projection_matches_matrix_product(channels: int, dtype: torch.dtype) -> None:
    gen = torch.Generator().manual_seed(0)
    harmonics = torch.randn(2, 64, channels, generator=gen, dtype=dtype)
    inverse = torch.randn(64, channels, generator=gen, dtype=dtype)
    expected = harmonics.transpose(1, 2) @ inverse
    torch.testing.assert_close(project_harmonics(harmonics, inverse), expected)
