"""PiDiNet: the pixel-difference convolutions folded into plain kernels, and the runner around the network."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import edges
from fastforks._core import registry
from fastforks._core.bench import synthetic_image
from fastforks.edges.models.pidinet.model import CONFIG, convert
from fastforks.edges.models.pidinet.net import PiDiNet, config_model, nets

if TYPE_CHECKING:
    from pathlib import Path


def _pdc(*, converted: bool) -> PiDiNet:
    pdcs = [nets[CONFIG][f"layer{i}"] for i in range(16)] if converted else config_model(CONFIG)
    return PiDiNet(60, pdcs, dil=24, sa=True, convert=converted).eval()


def test_card() -> None:
    cards = {v.id: c for c, v in registry.cards("edges").values() if c.family == "pidinet"}
    assert sorted(cards) == ["pidinet"]
    assert cards["pidinet"].licence.code == "LicenseRef-research-only"


def test_convert_is_the_same_function() -> None:
    """Every op of the config (cv, cd, ad, rd) folded into a plain kernel gives upstream's output."""
    torch.manual_seed(0)
    pdc = _pdc(converted=False)
    for p in pdc.parameters():  # non-trivial kernels for every op
        p.data.normal_(0, 0.1)
    plain = _pdc(converted=True)
    plain.load_state_dict(convert(pdc.state_dict()))
    x = torch.rand(1, 3, 32, 48)
    with torch.inference_mode():
        for a, b in zip(plain(x), pdc(x), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-5)


@pytest.mark.slow
def test_matches_pidinet_detector(tmp_path: Path) -> None:
    """`PidiNetDetector.__call__` in float at a detect size equal to the image size: the fused map of BGR / 255."""
    torch.manual_seed(0)
    pdc = _pdc(converted=False)
    torch.save({"state_dict": {f"module.{k}": v for k, v in pdc.state_dict().items()}}, tmp_path / "w.pth")
    m = edges.load("pidinet", weights=tmp_path / "w.pth", device="cpu", dtype="float32", options={"resolution": 64})
    img = synthetic_image(64, 128, torch.device("cpu"))
    with torch.inference_mode():
        want = pdc((img * 255).round().flip(1) / 255)[-1][0, 0]
        got = m(img).edges[0]
        exported = m.module()(img)[0]
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)
    torch.testing.assert_close(exported, got, rtol=0, atol=1e-6)
