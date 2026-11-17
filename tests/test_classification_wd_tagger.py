"""WD Tagger v3: multi-label probabilities, the white square canvas, the card's `[output]` / `[io]` rules."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from fastforks._core import cards
from fastforks._core.errors import CardError
from fastforks._core.labels import label_set
from fastforks._core.preprocess import resize_crop
from fastforks.classification import ClassificationResult, model_card


def test_card_multi_label():
    card = model_card("wd-vit-tagger-v3")
    names = label_set(card.output_info["label_set"])
    assert card.output_info["probs"] == "sigmoid"
    assert len(names) == 10861
    assert names[:4] == ("general", "sensitive", "questionable", "explicit")
    assert card.io.pad_square == 1.0


def test_probs_sigmoid_vs_softmax():
    logits = torch.tensor([[0.0, 2.0, -1.0]])
    single = ClassificationResult(logits, ("a", "b", "c"), "m", None)
    multi = dataclasses.replace(single, multi_label=True)
    torch.testing.assert_close(single.probs.sum(-1), torch.ones(1))
    torch.testing.assert_close(multi.probs, logits.sigmoid())
    assert multi.top(1)[0][0][0] == "b"


@pytest.mark.parametrize(("h", "w"), [(5, 9), (8, 3), (4, 4)])
def test_pad_square_pastes_centred(h, w):
    x = torch.zeros(1, 3, h, w)
    io = cards.Io(pad_square=1.0)
    out = resize_crop(x, io)
    side = max(h, w)
    left, top = (side - w) // 2, (side - h) // 2  # PIL paste offset of wdv3_timm.py
    assert out.shape[-2:] == (side, side)
    assert out[..., top : top + h, left : left + w].eq(0).all()
    assert out.sum() == 3 * (side * side - h * w)


def test_output_values_checked():
    sp_key = {"label_set": "imagenet-1k", "probs": "tanh"}
    with pytest.raises(CardError, match="probs"):
        cards._output("classification", ("logits",), sp_key, "card")
    with pytest.raises(CardError, match="pad_square"):
        cards._io({"pad_square": 2}, "card", "classification")
