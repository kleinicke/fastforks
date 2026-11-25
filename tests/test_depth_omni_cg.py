"""Omni-DC solver operator agreement at borders and at both pooled resolutions."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from fastforks.depth.models.omni_dc.net import OmniDCNet, normal_op
from fastforks.kernels import omni_cg


@pytest.mark.gpu
@pytest.mark.parametrize(("batch", "height", "width"), [(1, 8, 12), (2, 36, 60), (1, 120, 208)])
def test_fused_normal_operator(batch: int, height: int, width: int) -> None:
    direction = torch.randn((batch, 1, height, 2 * width), device="cuda")[..., ::2]
    diagonal = torch.rand_like(direction)
    actual = omni_cg.apply(direction, diagonal)
    expected = normal_op(direction, diagonal)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("batch", "height", "width", "layout"),
    [
        (1, 7, 13, torch.contiguous_format),
        (2, 32, 77, torch.channels_last),
    ],
)
def test_fused_depth_propagation(batch: int, height: int, width: int, layout: torch.memory_format) -> None:
    """The six deformable updates use strided affinities, zero borders and sparse depth reinjection."""
    gen = torch.Generator("cuda").manual_seed(11)
    shape = (batch, 1, height, width)
    depth = torch.rand(shape, device="cuda", generator=gen).mul_(8)
    sparse = torch.rand(shape, device="cuda", generator=gen).mul_(4)
    sparse[..., ::2, ::2] = 0
    conf = torch.randn(shape, device="cuda", generator=gen)
    offsets = torch.randn((batch, 90, height, width), device="cuda", generator=gen)
    offsets[:, :60].mul_(12)
    offsets = offsets.contiguous(memory_format=layout)
    net = cast("OmniDCNet", SimpleNamespace(prop_layer=SimpleNamespace(conv_offset_aff=torch.nn.Identity())))
    expected = OmniDCNet.spn(net, depth, offsets, sparse, conf)
    carry = conf.sigmoid() * sparse.sign()
    actual = omni_cg.propagate(depth, offsets, 1 - carry, carry * sparse)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-5)
