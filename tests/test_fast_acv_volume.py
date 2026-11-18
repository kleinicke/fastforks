"""Fast-ACVNet's sampled cost volume in native convolution layout."""

import pytest
import torch

from fastforks.kernels import acv_volume
from fastforks.stereo.models.fast_acvnet.model import warp


@pytest.mark.gpu
@pytest.mark.parametrize(("h", "w"), [(13, 23), (32, 40), (33, 129)])
def test_weighted_volume_matches_grid_sampling(h: int, w: int) -> None:
    rng = torch.Generator(device="cuda").manual_seed(53)
    b, c, d = 2, 5, 7
    features = torch.randn(2 * b, c, h, w, device="cuda", dtype=torch.float16, generator=rng)
    indices = torch.randint(0, w + 2, (b, 1, d, h, w), device="cuda", generator=rng)
    attention = torch.randn(b, 1, d, h, w, device="cuda", generator=rng).softmax(2)
    left = features[:b].float().unsqueeze(2) * attention
    right = warp(features[b:], indices[:, 0].float()) * attention
    actual = acv_volume.weighted(features, indices, attention)
    assert actual.is_contiguous(memory_format=torch.channels_last_3d)
    torch.testing.assert_close(actual, torch.cat((left, right), 1).half(), rtol=1e-3, atol=5e-5)
