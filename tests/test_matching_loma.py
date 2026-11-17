"""LoMa: the card, DaD's sub-pixel keypoint sampling, the match filter, the checkpoint split and the net's shapes."""

from __future__ import annotations

import torch

from fastforks import matching
from fastforks._core import registry
from fastforks.matching.models.loma.model import PREFIXES, split_checkpoint
from fastforks.matching.models.loma.net.dad import sample_keypoints
from fastforks.matching.models.loma.net.loma import LoMaNet, filter_matches


def test_card() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("matching").values() if c.family == "loma"}
    assert sorted(cards) == ["loma-b"]
    card, variant = cards["loma-b"]
    assert dict(card.options) == {"max_keypoints": 2048, "long_side": 1024, "size": 784}  # upstream Cfg, DaD
    assert dict(card.params) == {"threshold": 0.1}
    assert dict(variant.params) == {"dim": 256, "heads": 4, "layers": 9}
    assert set(card.outputs) == {"keypoints1", "keypoints2", "pairs", "scores"}
    assert [w.name for w in variant.weights] == ["loma_B.pt"]
    assert "loma-b" in matching.list_models()


def test_sample_keypoints() -> None:
    """The best pixels after 3 x 3 NMS, each pulled towards the higher logits in its patch."""
    logits = torch.zeros(1, 1, 8, 8)
    logits[0, 0, 2, 3] = 4.0
    logits[0, 0, 2, 4] = 3.0  # suppressed by the (2, 3) peak, which it pulls to the right
    logits[0, 0, 5, 5] = 4.0
    logits[0, 0, 5, 6] = 4.0  # equal neighbour: NMS keeps both (ties), the offsets pull each towards the other
    kpts = sample_keypoints(logits, 3, nms_size=3, temp=0.5)
    px = (kpts + 1) / 2 * 8 - 0.5  # (x, y) pixel centres
    rows = px[:, 1].round()
    assert rows.tolist().count(2.0) == 1
    assert rows.tolist().count(5.0) == 2
    peak = px[rows == 2.0][0]
    assert torch.allclose(peak[1], torch.tensor(2.0), atol=1e-3)
    assert 3.0 < peak[0] < 4.0
    others = px[rows == 5.0]
    assert torch.allclose(others[:, 1], torch.full((2,), 5.0), atol=1e-3)
    assert (others[:, 0] > 5.0).all()
    assert (others[:, 0] < 6.0).all()
    assert torch.allclose(others[:, 0].sum(), torch.tensor(11.0), atol=1e-3)  # symmetric pull


def test_filter_matches() -> None:
    """Mutual maxima above the threshold; one-sided maxima and weak mutual ones are dropped."""
    s = torch.zeros(1, 3, 3)
    s[0, 0, 1] = 0.9  # mutual
    s[0, 1, 0] = 0.05  # mutual but weak
    s[0, 2, 1] = 0.5  # row 2 -> col 1, but col 1's maximum is row 0
    m0, scores = filter_matches(s, 0.1)
    assert m0.tolist() == [1, -1, -1]
    assert torch.allclose(scores, torch.tensor([0.9, 0.05, 0.0]))


def test_split_checkpoint_and_shapes() -> None:
    """A random checkpoint in upstream's layout loads strictly into the parts; the matcher gives `[1, M, N]`."""
    net = LoMaNet(32, 32, 4, 2)
    sd = {f"transformers.{k}" if k.startswith(("0.", "1.")) else k: v for k, v in net.state_dict().items()}
    sd["_detector.encoder.layers.0.weight"] = torch.zeros(1)
    sd["_descriptor.encoder.vgg.layers.0.weight"] = torch.zeros(1)
    sd["_descriptor.encoder.frozen_dinov2.dinov2_vitl14.cls_token"] = torch.zeros(1)
    sd["_descriptor.decoder.layers.8.out_conv.weight"] = torch.zeros(1)
    parts = split_checkpoint(sd)
    assert set(parts) == {*PREFIXES.values(), "matcher"}
    assert list(parts["dinov2"]) == ["cls_token"]
    assert list(parts["descriptor.decoder"]) == ["layers.8.out_conv.weight"]
    assert set(parts["matcher"]) == set(net.state_dict())
    net.load_state_dict(parts["matcher"], strict=True)
    g = torch.Generator().manual_seed(0)
    k0, k1 = torch.rand(1, 5, 2, generator=g) * 2 - 1, torch.rand(1, 7, 2, generator=g) * 2 - 1
    scores = net(k0, k1, torch.randn(1, 5, 32, generator=g), torch.randn(1, 7, 32, generator=g))
    assert scores.shape == (1, 5, 7)
    assert (scores >= 0).all()
    assert (scores <= 1).all()
