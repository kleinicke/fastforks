"""FILM: the card's variant, the forward against vendored upstream on random weights, padding, export."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import interpolation
from fastforks._core import registry
from fastforks._core.bench import synthetic_pair
from fastforks.interpolation.models.film.model import _interpolate, _taps, _warp
from fastforks.interpolation.models.film.net import Interpolator

if TYPE_CHECKING:
    from pathlib import Path

# the card's level structure, narrow layers (fast on the CPU)
SMALL = {"filters": 4, "flow_convs": (3, 3, 3, 3), "flow_filters": (4, 4, 8, 8)}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_export_warp_matches_eager(monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype) -> None:
    """Export's gather implementation agrees with the eager four-tap warp, including borders."""
    torch.manual_seed(3)
    image = torch.randn(2, 7, 5, 9, dtype=dtype)
    flow = 4 * torch.randn(2, 2, 5, 9)
    taps = _taps(flow, dtype, other=True)
    want = _warp(image, taps)
    monkeypatch.setattr(torch.compiler, "is_exporting", lambda: True)
    got = _warp(image, taps)
    torch.testing.assert_close(got, want, rtol=0, atol=0.002 if dtype == torch.float16 else 1e-6)


def test_card_variants() -> None:
    cards = {v.id for c, v in registry.cards("interpolation").values() if c.family == "film"}
    assert cards == {"film"}


@pytest.fixture(scope="module")
def net() -> Interpolator:
    torch.manual_seed(0)
    return Interpolator(**SMALL).eval()


@pytest.mark.parametrize("t", [0.5, 0.3])
def test_forward_matches_upstream(net: Interpolator, t: float) -> None:
    """`_interpolate` (both frames in one batch, float32 flows and warps) is the vendored upstream forward."""
    a, b = synthetic_pair(128, 192, torch.device("cpu"), shift=3)
    with torch.inference_mode():
        want = net(a, b, torch.full((1, 1), t))
        got = _interpolate(net, a, b, torch.full((1, 1, 1, 1), t))
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)


def test_odd_size_is_padded_and_exports(tmp_path: Path) -> None:
    """The full-width card network with random weights (saved as the release's TorchScript archive)."""
    torch.manual_seed(0)
    path = tmp_path / "film-net-style-fp32.pt"
    torch.save(Interpolator().state_dict(), path)
    model = interpolation.load("film", weights=path, device="cpu")
    a, b = synthetic_pair(71, 101, torch.device("cpu"), shift=3)  # zero padded to 128 x 128 (centred), cropped
    out = model(a, b).frame
    assert out.shape == (1, 3, 71, 101)
    assert torch.isfinite(out).all()
    assert out.min() >= 0
    assert out.max() <= 1
    with torch.inference_mode():
        exported = model.module()(a, b)
    torch.testing.assert_close(exported, out, rtol=0, atol=1e-5)
