"""XFeat sparse matchers: card metadata and empty descriptor sets."""

from __future__ import annotations

import torch

from fastforks import matching
from fastforks._core import registry
from fastforks.matching.models.xfeat.net.xfeat import XFeat


def test_cards() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("matching").values() if c.family == "xfeat"}
    assert sorted(cards) == ["xfeat", "xfeat-lighterglue"]
    for mid, (card, variant) in cards.items():
        assert dict(card.options) == {"max_keypoints": 4096, "long_side": 0}  # upstream's top_k, no resize
        assert set(variant.outputs or card.outputs) == {"keypoints1", "keypoints2", "pairs", "scores"}
        assert mid in matching.list_models()


def test_match_without_keypoints() -> None:
    model = XFeat()
    empty = torch.empty(0, 64)
    populated = torch.ones(2, 64)
    for left, right in ((empty, empty), (empty, populated), (populated, empty)):
        idx0, idx1 = model.match(left, right)
        for indices in (idx0, idx1):
            assert indices.shape == (0,)
            assert indices.dtype == torch.long
            assert indices.device == left.device
