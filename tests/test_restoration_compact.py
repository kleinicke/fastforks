"""Real-ESRGAN Compact: the card's variant, the network against upstream's forward, `load_file` detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import restoration
from fastforks._core import registry
from fastforks._core.bench import synthetic_image
from fastforks._core.errors import WeightsError
from fastforks.restoration.models.compact import model
from fastforks.restoration.models.compact.net import SRVGGNetCompact

if TYPE_CHECKING:
    from pathlib import Path


def test_card_variant() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("restoration").values() if c.family == "compact"}
    assert sorted(cards) == ["realesr-general-x4v3"]
    card, variant = cards["realesr-general-x4v3"]
    assert {**card.params, **variant.params} == {"scale": 4, "num_feat": 64, "num_conv": 32}


@pytest.mark.parametrize(("scale", "feat", "convs"), [(4, 64, 32), (2, 16, 3)])
def test_load_file_matches_upstream(tmp_path: Path, scale: int, feat: int, convs: int) -> None:
    torch.manual_seed(0)
    net = SRVGGNetCompact(3, 3, num_feat=feat, num_conv=convs, upscale=scale, act_type="prelu").eval()
    torch.save({"params": net.state_dict()}, tmp_path / "w.pth")
    m = restoration.load_file(tmp_path / "w.pth", device="cpu", dtype="float32")
    assert m.variant.id == "realesr-general-x4v3"
    img = synthetic_image(17, 23, torch.device("cpu"))
    with torch.inference_mode():
        want = net(img).clamp(0, 1)
        got = m(img).image
        exported = m.module()(img).clamp(0, 1)
    assert got.shape == (1, 3, 17 * scale, 23 * scale)
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)
    torch.testing.assert_close(exported, got, rtol=0, atol=1e-6)


def test_from_checkpoint_rejects_other_configs() -> None:
    sd = SRVGGNetCompact(3, 3, num_feat=8, num_conv=2, upscale=4, act_type="relu").state_dict()
    with pytest.raises(WeightsError, match="PReLU"):
        model.from_checkpoint(sd)
    assert model.from_checkpoint({"conv_first.weight": torch.zeros(64, 3, 3, 3)}) is None
