"""TEED: the card and the runner's post-processing of the four outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn.functional as F

from fastforks import edges
from fastforks._core import registry
from fastforks._core.bench import synthetic_image
from fastforks.shared.teed import TED

if TYPE_CHECKING:
    from pathlib import Path


def test_card() -> None:
    cards = {v.id: c for c, v in registry.cards("edges").values() if c.family == "teed"}
    assert sorted(cards) == ["teed"]
    assert cards["teed"].licence.code == "MIT"
    assert cards["teed"].options["safe_steps"] == 2


@pytest.mark.slow
@pytest.mark.parametrize("steps", [0, 2])
def test_matches_teed_detector(tmp_path: Path, steps: int) -> None:
    """`TEEDdetector.__call__` in float: outputs bilinear to the detect size, sigmoid of the mean, `safe_step`."""
    torch.manual_seed(0)
    net = TED().eval()
    torch.save(net.state_dict(), tmp_path / "w.pth")
    opts = {"resolution": 64, "safe_steps": steps}
    m = edges.load("teed", weights=tmp_path / "w.pth", device="cpu", dtype="float32", options=opts)
    img = synthetic_image(64, 128, torch.device("cpu"))
    with torch.inference_mode():
        outs = [F.interpolate(e, (64, 128), mode="bilinear", align_corners=False) for e in net((img * 255).round())]
        want = torch.sigmoid(torch.cat(outs, 1).mean(1))[0]
        if steps:
            want = ((want * (steps + 1)).floor() / steps).clamp(max=1)
        got = m(img).edges[0]
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)
