"""HED (ControlNet's Apache-2.0 re-implementation): the card and the runner's post-processing of the side outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn.functional as F

from fastforks import edges
from fastforks._core import registry
from fastforks._core.bench import synthetic_image
from fastforks.edges.models.hed.net import ControlNetHED_Apache2

if TYPE_CHECKING:
    from pathlib import Path


def test_card() -> None:
    cards = {v.id: c for c, v in registry.cards("edges").values() if c.family == "hed"}
    assert sorted(cards) == ["hed"]
    assert cards["hed"].licence.code == "Apache-2.0"
    assert cards["hed"].options["safe"] is False


@pytest.mark.slow
@pytest.mark.parametrize("safe", [False, True])
def test_matches_hed_detector(tmp_path: Path, safe: bool) -> None:  # noqa: FBT001 - parametrized
    """`HEDdetector.__call__` in float: side outputs bilinear to the detect size, sigmoid of the mean, `safe_step`."""
    torch.manual_seed(0)
    net = ControlNetHED_Apache2().eval()
    torch.save(net.state_dict(), tmp_path / "w.pth")
    opts = {"resolution": 64, "safe": safe}
    m = edges.load("hed", weights=tmp_path / "w.pth", device="cpu", dtype="float32", options=opts)
    img = synthetic_image(64, 128, torch.device("cpu"))
    with torch.inference_mode():
        sides = [F.interpolate(e, (64, 128), mode="bilinear", align_corners=False) for e in net((img * 255).round())]
        want = torch.sigmoid(torch.cat(sides, 1).mean(1))[0]
        if safe:
            want = (want * 3).floor() / 2
        got = m(img).edges[0]
        exported = m.module()(img)[0]
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)
    torch.testing.assert_close(exported, got, rtol=0, atol=1e-6)
