"""Anime line art (pix2pix U-Net): the card and the runner's pre- and post-processing around the network."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import edges
from fastforks._core import registry
from fastforks._core.bench import synthetic_image
from fastforks.edges.models.lineart_anime.net import UnetGenerator

if TYPE_CHECKING:
    from pathlib import Path


def test_card() -> None:
    cards = {v.id: c for c, v in registry.cards("edges").values() if c.family == "lineart_anime"}
    assert sorted(cards) == ["lineart-anime"]
    assert cards["lineart-anime"].licence.code == "MIT"


@pytest.mark.slow
def test_matches_lineart_anime_detector(tmp_path: Path) -> None:
    """`LineartAnimeDetector.__call__` at a 256 x 256 detect size (no resize): `1 - clip(net(x) * 127.5 + 127.5)`."""
    torch.manual_seed(0)
    norm = functools.partial(torch.nn.InstanceNorm2d, affine=False, track_running_stats=False)
    net = UnetGenerator(3, 1, 8, 64, norm_layer=norm, use_dropout=False).eval()
    torch.save({f"module.{k}": v for k, v in net.state_dict().items()}, tmp_path / "w.pth")  # as the DataParallel file
    m = edges.load(
        "lineart-anime", weights=tmp_path / "w.pth", device="cpu", dtype="float32", options={"resolution": 256}
    )
    img = synthetic_image(256, 256, torch.device("cpu"))
    with torch.inference_mode():
        y = net((img * 255).round() / 127.5 - 1)[0, 0] * 127.5 + 127.5
        want = 1 - y.clamp(0, 255) / 255
        got = m(img).edges[0]
        exported = m.module()(img)[0]
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)
    torch.testing.assert_close(exported, got, rtol=0, atol=1e-6)
