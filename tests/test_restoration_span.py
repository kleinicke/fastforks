"""SPAN: the card's variants, the once-fused network against upstream's per-call fusion, export."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import restoration
from fastforks._core import registry
from fastforks._core.bench import synthetic_image
from fastforks.restoration.models.span.net import SPAN

if TYPE_CHECKING:
    from pathlib import Path

VARIANTS = {"span-x4": 4, "span-x2": 2}


def test_card_variants() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("restoration").values() if c.family == "span"}
    assert sorted(cards) == sorted(VARIANTS)
    for vid, scale in VARIANTS.items():
        card, variant = cards[vid]
        assert {**card.params, **variant.params}["scale"] == scale, vid


@pytest.mark.parametrize("ref", sorted(VARIANTS))
def test_fused_once_matches_upstream_eval(tmp_path: Path, ref: str) -> None:
    torch.manual_seed(0)
    net = SPAN(3, 3, feature_channels=48, upscale=VARIANTS[ref]).eval()
    torch.save({"params_ema": net.state_dict()}, tmp_path / "w.pth")
    m = restoration.load(ref, weights=tmp_path / "w.pth", device="cpu", dtype="float32")
    img = synthetic_image(17, 23, torch.device("cpu"))
    with torch.inference_mode():
        want = net(img)  # upstream: re-fuses every Conv3XC on each eval call
        raw = m._runner(img)["image"]  # unclamped: random weights saturate the clamp
        exported = m.module()(img)
    torch.testing.assert_close(raw, want, rtol=1e-5, atol=1e-4 * float(want.abs().max()))
    got = m(img).image
    assert got.shape == (1, 3, 17 * VARIANTS[ref], 23 * VARIANTS[ref])
    torch.testing.assert_close(exported, got, rtol=0, atol=1e-6)
