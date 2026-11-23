"""`cnn_kernels.fuse_sequentials`: which conv -> BatchNorm -> activation runs it fuses (CPU, structure only)."""

from __future__ import annotations

from torch import nn

from fastforks.shared import cnn_kernels
from fastforks.shared.cnn_kernels import ACT, FusedConvBN


def _convbn() -> nn.Sequential:
    return nn.Sequential(nn.Conv3d(4, 4, 3, padding=1, bias=False), nn.BatchNorm3d(4))


def test_nested_relu_joins_the_epilogue() -> None:
    net = nn.Sequential(_convbn(), nn.ReLU(inplace=True), _convbn(), nn.Sigmoid(), _convbn())
    assert cnn_kernels.fuse_sequentials(net) == 3
    first, second, third = net[0][0], net[2][0], net[4][0]
    assert isinstance(first, FusedConvBN)
    assert isinstance(second, FusedConvBN)
    assert isinstance(third, FusedConvBN)
    assert first.act == ACT["relu"]
    assert isinstance(net[1], nn.Identity)
    assert second.act == 0  # sigmoid in fp32 would round differently: stays a module
    assert isinstance(net[3], nn.Sigmoid)
    assert third.act == 0


def test_flat_run_keeps_its_activation() -> None:
    net = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU6(), nn.ReLU())
    assert cnn_kernels.fuse_sequentials(net) == 1
    assert isinstance(net[0], FusedConvBN)
    assert net[0].act == ACT["relu6"]
    assert isinstance(net[3], nn.ReLU)  # a second activation is not a nested run's
