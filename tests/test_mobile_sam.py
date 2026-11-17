"""MobileSAM: the traceable TinyViT forward (`fast.Trunk`) equals upstream `TinyViT.forward` on the same modules."""

from __future__ import annotations

import pytest
import torch

from fastforks.promptable_segmentation.models.mobile_sam.fast import Trunk
from fastforks.promptable_segmentation.models.mobile_sam.model import Params, Sam


@pytest.mark.slow
def test_trunk_matches_upstream_forward() -> None:
    size = 1024  # upstream's forward reshapes to 64 x 64 tokens: the only size it runs at
    torch.manual_seed(0)
    enc = Sam(Params(image_size=size)).image_encoder.eval()
    for m in enc.modules():  # upstream zero-initialises the residual branches' last BatchNorm: make every path count
        if isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.uniform_(m.weight, 0.5, 1.5)
            torch.nn.init.uniform_(m.running_var, 0.5, 1.5)
            torch.nn.init.normal_(m.running_mean, std=0.2)
    for name, p in enc.named_parameters():
        if name.endswith("attention_biases"):
            torch.nn.init.normal_(p)
    enc.train().eval()  # upstream `Attention.train(False)` caches the bias table `ab` from `attention_biases`
    x = torch.randn(1, 3, size, size)
    with torch.no_grad():
        want = enc(x)
        got = enc.neck[3](Trunk(enc)(x))
    assert want.shape == (1, 256, 64, 64)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
