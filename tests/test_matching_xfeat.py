"""XFeat sparse matchers: the cards."""

from __future__ import annotations

from fastforks import matching
from fastforks._core import registry


def test_cards() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("matching").values() if c.family == "xfeat"}
    assert sorted(cards) == ["xfeat", "xfeat-lighterglue"]
    for mid, (card, variant) in cards.items():
        assert dict(card.options) == {"max_keypoints": 4096, "long_side": 0}  # upstream's top_k, no resize
        assert set(variant.outputs or card.outputs) == {"keypoints1", "keypoints2", "pairs", "scores"}
        assert mid in matching.list_models()
