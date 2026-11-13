"""LightGlue sparse matchers: the cards, the sparse result's matched points, the long-side resize."""

from __future__ import annotations

import torch

from fastforks import matching
from fastforks._core import registry
from fastforks.matching import SparseMatchingResult
from fastforks.shared.lightglue import resize_long


def test_cards() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("matching").values() if c.family == "lightglue"}
    assert sorted(cards) == ["aliked-lightglue", "disk-lightglue"]
    for mid, (card, variant) in cards.items():
        assert dict(card.options) == {"max_keypoints": 2048, "long_side": 1024}  # upstream's match_pair setting
        assert set(variant.outputs or card.outputs) == {"keypoints1", "keypoints2", "pairs", "scores"}
        assert mid in matching.list_models()


def test_matches_index_the_keypoints() -> None:
    k1, k2 = torch.arange(10.0).reshape(5, 2), -torch.arange(6.0).reshape(3, 2)
    r = SparseMatchingResult(k1, k2, torch.tensor([[4, 0], [1, 2]]), torch.tensor([0.9, 0.2]), "m", None)  # type: ignore[arg-type]
    p1, p2 = r.matches()
    assert torch.equal(p1, k1[[4, 1]])
    assert torch.equal(p2, k2[[0, 2]])
    p1, p2 = r.matches(threshold=0.5)
    assert torch.equal(p1, k1[[4]])
    assert torch.equal(p2, k2[[0]])


def test_resize_long_keeps_the_aspect() -> None:
    x = torch.rand(1, 3, 48, 64)
    assert resize_long(x, 32).shape[-2:] == (24, 32)
    assert resize_long(x.transpose(-1, -2), 32).shape[-2:] == (32, 24)
