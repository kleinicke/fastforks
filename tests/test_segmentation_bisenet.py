"""BiSeNet face parsing: the card's variant, its label set and the main-head forward against the vendored network."""

from __future__ import annotations

import torch

from fastforks._core import registry
from fastforks._core.labels import label_set
from fastforks.segmentation.models.bisenet import model
from fastforks.segmentation.models.bisenet.net import BiSeNet


def test_card_variant() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("segmentation").values() if c.family == "bisenet"}
    assert sorted(cards) == ["bisenet-face-parsing"]
    names = label_set("celebamask-hq-19-bisenet")
    assert len(names) == 19
    assert (names[0], names[1], names[10], names[17]) == ("background", "skin", "nose", "hair")  # face-parsing.PyTorch


def test_main_head_folded_matches_network() -> None:
    torch.manual_seed(0)
    net = BiSeNet(num_class=19).eval()
    for m in net.modules():  # non-trivial BatchNorm statistics
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.5, 0.5)
    x = torch.randn(1, 3, 64, 96)
    with torch.inference_mode():
        want = net(x)[0]
        model._fold(net)
        got = model._main_head(net, x)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
