"""RIFE: the card's variants, the network on random weights, padding of odd sizes, timestep checks, export."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import interpolation
from fastforks._core import registry
from fastforks._core.bench import synthetic_pair
from fastforks._core.errors import InputError
from fastforks.interpolation.models.rife.net import IFNet

if TYPE_CHECKING:
    from pathlib import Path

VARIANTS = {"rife-4.25": 32, "rife-4.26": 32, "rife-4.25-lite": 24}  # id -> channels of the last block


def test_card_variants() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("interpolation").values() if c.family == "rife"}
    assert sorted(cards) == sorted(VARIANTS)
    for vid, c4 in VARIANTS.items():
        card, variant = cards[vid]
        assert {**card.params, **variant.params}["c4"] == c4, vid
    assert interpolation.list_models()[0] == "rife-4.25"  # the default: upstream's recommended release


@pytest.fixture(scope="module")
def weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A checkpoint with random weights in the releases' layout (`module.` prefix, unused teacher keys)."""
    torch.manual_seed(0)
    sd = {f"module.{k}": v for k, v in IFNet().state_dict().items()}
    sd["module.teacher.conv0.0.weight"] = torch.zeros(1)  # training-only keys the releases carry
    path = tmp_path_factory.mktemp("rife") / "flownet.pkl"
    torch.save(sd, path)
    return path


def test_odd_size_is_padded_and_exports(weights: Path) -> None:
    model = interpolation.load("rife-4.25", weights=weights, device="cpu")
    a, b = synthetic_pair(67, 93, torch.device("cpu"), shift=3)  # zero padded to 128 x 128, cropped again
    out = model(a, b).frame
    assert out.shape == (1, 3, 67, 93)
    assert torch.isfinite(out).all()
    with torch.inference_mode():
        exported = model.module()(a, b)  # export does what the call does (time 0.5)
    torch.testing.assert_close(exported, out, rtol=0, atol=1e-5)


@pytest.mark.parametrize("timestep", [-0.1, 1.5, float("nan"), "0.5"])
def test_bad_timestep_is_rejected(weights: Path, timestep: object) -> None:
    model = interpolation.load("rife-4.25", weights=weights, device="cpu")
    a, b = synthetic_pair(32, 32, torch.device("cpu"))
    with pytest.raises(InputError, match="timestep"):
        model(a, b, timestep=timestep)
