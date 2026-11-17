"""Exact-maths folds of the portable runners: LayerScale into the linear before it, BatchNorm into its conv (CPU)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from fastforks.segmentation.models.mmseg import net as mm
from fastforks.segmentation.models.mmseg.model import fold_batchnorms
from fastforks.shared.depth_anything_v2 import fold_layer_scale
from fastforks.shared.depth_anything_v2.net.dinov2 import DinoVisionTransformer


def _randomize_bn(net: nn.Module) -> int:
    g = torch.Generator().manual_seed(0)
    n = 0
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            c = m.num_features
            m.running_mean.copy_(torch.randn(c, generator=g))
            m.running_var.copy_(torch.rand(c, generator=g) + 0.5)
            m.weight.data.copy_(torch.rand(c, generator=g) + 0.5)
            m.bias.data.copy_(torch.randn(c, generator=g))
            n += 1
    return n


@pytest.mark.parametrize("arch", ["fcn-hr48", "sepaspp-r101-d8"])
def test_fold_batchnorms(arch: str) -> None:
    torch.manual_seed(0)
    net = mm.build(arch).eval()
    count = _randomize_bn(net)
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        ref = net(x)
        assert fold_batchnorms(net) == count  # every BatchNorm follows its conv
        out = net(x)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in net.modules())
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4 * ref.abs().max().item())


@pytest.mark.parametrize("ffn", ["mlp", "swiglufused"])
def test_fold_layer_scale(ffn: str) -> None:
    torch.manual_seed(0)
    net = DinoVisionTransformer(
        img_size=28, patch_size=14, embed_dim=64, depth=2, num_heads=2, init_values=0.5, ffn_layer=ffn
    ).eval()
    for name, p in net.named_parameters():
        if name.endswith("gamma"):
            p.data.uniform_(0.1, 2.0)
    x = torch.rand(1, 3, 28, 28)
    with torch.no_grad():
        ref = net.forward_features(x)["x_norm_patchtokens"]
        assert fold_layer_scale(net) == 4  # two blocks, two LayerScales each
        out = net.forward_features(x)["x_norm_patchtokens"]
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(("groups", "bias", "dilation"), [(1, True, 1), (1, False, 2), (4, True, 1)])
def test_fold_bn(groups: int, bias: bool, dilation: int) -> None:  # noqa: FBT001 - parametrized
    """`cnn_kernels.fold_bn`: conv -> BatchNorm unchanged, the BatchNorm left as an exact identity."""
    from fastforks.shared.cnn_kernels import fold_bn  # noqa: PLC0415 - kernels package import, CPU only here

    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(8, 12, 3, padding=dilation, dilation=dilation, groups=groups, bias=bias),
                        nn.BatchNorm2d(12)).eval()  # fmt: skip
    _randomize_bn(net)
    x = torch.randn(2, 8, 9, 11)
    want = net(x)
    fold_bn(net[0], net[1])
    torch.testing.assert_close(net(x), want, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(net[1](want), want, rtol=0, atol=0)
