"""The HITNet feature-head kernels (`img_conv3`, `deconv2x2`, `cat_cl`) against PyTorch (GPU; odd sizes, batch 2)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fastforks.stereo.models.hitnet import kernels as hitnet_kernels
from fastforks.stereo.models.hitnet.net.hitnet import HITNet

pytestmark = pytest.mark.gpu


@pytest.fixture
def dev():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    return torch.device("cuda", torch.cuda.current_device())


@pytest.mark.parametrize(("in_ch", "pads"), [(1, (127.5, 127.5)), (3, (0.0, 255.0))])
def test_feature_head(dev, in_ch, pads):
    """`img_conv3` (padding, gray, scaling, first conv), `deconv2x2` (into the skip concatenation), fused convs."""
    torch.manual_seed(0)
    net = HITNet(in_ch, [16, 16, 24], [8, 16, 32], []).to(dev).eval()
    m = hitnet_kernels.KernelsHITNet(net, dev)
    left, right = (torch.rand(2, 3, 37, 61, device=dev) * 255 for _ in range(2))
    size = (64, 64)
    with torch.no_grad():
        lp = F.pad(left, (0, 3, 0, 27), value=pads[0])
        rp = F.pad(right, (0, 3, 0, 27), value=pads[1])
        want = net.fe(net.prepare(lp, rp))
        got = m.features(left, right, size, pads)
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        assert g.shape == w.shape
        torch.testing.assert_close(g.float(), w, atol=0.01 * float(w.abs().max()), rtol=0.01)


def test_cat_cl(dev):
    """Up to four fp16 / fp32 parts in any layout -> one fp16 channels-last tensor."""
    parts = [
        torch.randn(2, 8, 5, 7, device=dev).half().contiguous(memory_format=torch.channels_last),
        torch.randn(2, 2, 5, 7, device=dev),
        torch.randn(2, 7, 5, 9, device=dev).half()[..., 1:8],
        torch.randn(2, 1, 5, 7, device=dev),
    ]
    got = hitnet_kernels._cat_cl(hitnet_kernels.hitnet.load(dev), parts)
    assert got.is_contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(got, torch.cat([p.half() for p in parts], 1))
