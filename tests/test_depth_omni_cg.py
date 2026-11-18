"""Omni-DC solver operator agreement at borders and at both pooled resolutions."""

import pytest
import torch

from fastforks.depth.models.omni_dc.net import normal_op
from fastforks.kernels import omni_cg


@pytest.mark.gpu
@pytest.mark.parametrize(("batch", "height", "width"), [(1, 8, 12), (2, 36, 60), (1, 120, 208)])
def test_fused_normal_operator(batch: int, height: int, width: int) -> None:
    direction = torch.randn((batch, 1, height, 2 * width), device="cuda")[..., ::2]
    diagonal = torch.rand_like(direction)
    actual = omni_cg.apply(direction, diagonal)
    expected = normal_op(direction, diagonal)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
