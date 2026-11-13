"""LoFTR: the cards, the matching size and the position encoding of both variants."""

from __future__ import annotations

import pytest
import torch

from fastforks import matching
from fastforks._core import registry
from fastforks.matching.models.loftr.model import Matcher, Params, resized_size
from fastforks.matching.models.loftr.net.loftr import sine_encoding


def test_cards() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("matching").values() if c.family == "loftr"}
    assert sorted(cards) == ["loftr-indoor", "loftr-outdoor"]
    for mid, (card, variant) in cards.items():
        assert dict(card.options) == {"long_side": 840}  # upstream's MegaDepth test size
        assert set(variant.outputs or card.outputs) == {"keypoints1", "keypoints2", "pairs", "scores"}
        assert mid in matching.list_models()
    assert cards["loftr-outdoor"][1].params == {"temp_bug_fix": False, "border_rm": 2}
    assert cards["loftr-indoor"][1].params == {"temp_bug_fix": True, "border_rm": 0}


def test_resized_size() -> None:
    assert resized_size(384, 512, 840) == (624, 840)  # 630 cut down to a multiple of 8
    assert resized_size(480, 640, 0) == (480, 640)
    assert resized_size(483, 645, 0) == (480, 640)


def test_too_small() -> None:
    m = Matcher(Params(temp_bug_fix=True, border_rm=0, threshold=0.2, long_side=0))
    with pytest.raises(ValueError, match="too small for LoFTR"):
        m.prepare(torch.zeros(1, 3, 12, 64))


def test_sine_encoding_bug_fix() -> None:
    """The fixed encoding divides the frequencies by d_model / 2; the original one (outdoor weights) does not."""
    fixed = sine_encoding(8, 2, 3, temp_bug_fix=True, device=torch.device("cpu"))
    buggy = sine_encoding(8, 2, 3, temp_bug_fix=False, device=torch.device("cpu"))
    assert fixed.shape == buggy.shape == (1, 8, 2, 3)
    torch.testing.assert_close(fixed[0, 0, 0], torch.arange(1, 4.0).sin())  # x + 1, frequency 1
    assert not torch.allclose(fixed[0, 4], buggy[0, 4])
