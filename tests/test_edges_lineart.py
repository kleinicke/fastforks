"""Line art (informative-drawings): the card's variants and the runner's pre- and post-processing around the network."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import edges
from fastforks._core import registry
from fastforks._core.bench import synthetic_image
from fastforks.edges.models.lineart.net import Generator

if TYPE_CHECKING:
    from pathlib import Path


def test_card() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("edges").values() if c.family == "lineart"}
    assert sorted(cards) == ["lineart", "lineart-coarse"]
    card = cards["lineart"][0]
    assert card.licence.code == "MIT"
    assert card.options["resolution"] == 512


@pytest.mark.slow
def test_matches_lineart_detector(tmp_path: Path) -> None:
    """`LineartDetector.__call__` in float at a detect size equal to the image size: `1 - net(x / 255)`."""
    torch.manual_seed(0)
    net = Generator(3, 1, 3).eval()
    torch.save(net.state_dict(), tmp_path / "w.pth")
    m = edges.load("lineart", weights=tmp_path / "w.pth", device="cpu", dtype="float32", options={"resolution": 64})
    img = synthetic_image(64, 128, torch.device("cpu"))
    with torch.inference_mode():
        want = 1 - net((img * 255).round() / 255)[0, 0]
        got = m(img).edges[0]
        exported = m.module()(img)[0]
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)
    torch.testing.assert_close(exported, got, rtol=0, atol=1e-6)
