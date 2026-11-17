"""Matting runners stay finite on constant images (the `/ max` and min-max normalisations once divided 0 by 0)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import matting
from fastforks.matting.models.isnet.net import ISNetDIS
from fastforks.matting.models.u2net.net import U2NETP

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.slow
@pytest.mark.parametrize(("ref", "cls"), [("u2netp", U2NETP), ("isnet", ISNetDIS)])
def test_alpha_finite_on_constant_images(tmp_path: Path, ref: str, cls: type) -> None:
    """Black, white and grey images give a finite alpha (random weights: the maps are constant, hi == lo)."""
    torch.manual_seed(0)
    torch.save(cls(3, 1).eval().state_dict(), tmp_path / "w.pth")
    m = matting.load(ref, weights=tmp_path / "w.pth", device="cpu", dtype="float32")
    for name, value in {"black": 0.0, "white": 1.0, "grey": 0.5}.items():
        alpha = m(torch.full((1, 3, 45, 67), value)).alpha
        assert torch.isfinite(alpha).all(), f"{ref} {name}: NaN / inf"
