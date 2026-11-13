"""LAS2 portable: the batched correlation volumes and the BatchNorm folding match upstream's layers."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from fastforks.stereo.models.las2 import model as las2
from fastforks.stereo.models.las2.net.submodule import (
    BasicConv,
    build_correlation_volume,
    build_gwc_volume_fast,
)


@pytest.mark.parametrize(("depth", "groups", "width"), [(1, 1, 5), (6, 1, 13), (6, 4, 13), (12, 8, 7)])
def test_volumes_match_upstream(depth: int, groups: int, width: int) -> None:
    g = torch.Generator().manual_seed(depth * groups + width)
    left, right = (torch.randn(2, 16, 3, width, generator=g) for _ in range(2))
    torch.testing.assert_close(
        las2.gwc_volume_einsum(left, right, depth, groups), build_gwc_volume_fast(left, right, depth, groups)
    )
    if groups == 1:
        torch.testing.assert_close(
            las2.corr_volume_einsum(left, right, depth), build_correlation_volume(left, right, depth)
        )


@pytest.mark.parametrize("deconv", [False, True])
@pytest.mark.parametrize("is_3d", [False, True])
def test_fold_batchnorms_keeps_output(*, deconv: bool, is_3d: bool) -> None:
    torch.manual_seed(0)
    conv = (nn.ConvTranspose3d if deconv else nn.Conv3d) if is_3d else (nn.ConvTranspose2d if deconv else nn.Conv2d)
    norm = nn.BatchNorm3d if is_3d else nn.BatchNorm2d
    net = nn.Sequential(
        BasicConv(4, 6, deconv=deconv, is_3d=is_3d, kernel_size=3, padding=1),
        nn.Sequential(conv(6, 5, 3, padding=1), norm(5), nn.ReLU()),
    )
    for m in net.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.running_mean.uniform_(-1, 1)
            m.running_var.uniform_(0.5, 2)
            m.weight.data.uniform_(0.5, 2)
            m.bias.data.uniform_(-1, 1)
    net.eval()
    x = torch.randn(1, 4, *([3] if is_3d else []), 5, 6)
    with torch.no_grad():
        want = net(x)
        assert las2.fold_batchnorms(net) == 2
        got = net(x)
    assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in net.modules())
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
