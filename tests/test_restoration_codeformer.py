"""CodeFormer: the card, the inference forward against the vendored upstream network, the input size check, export."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import restoration
from fastforks._core import registry
from fastforks._core.errors import InputError
from fastforks.restoration.models.codeformer.net import CodeFormer

if TYPE_CHECKING:
    from pathlib import Path


def _card() -> tuple:
    (cv,) = [cv for cv in registry.cards("restoration").values() if cv[0].family == "codeformer"]
    return cv


def test_card_variants() -> None:
    _, variant = _card()
    assert variant.id == "codeformer"


@pytest.mark.slow
def test_matches_vendored_upstream(tmp_path: Path) -> None:
    card, _ = _card()
    torch.manual_seed(0)
    net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=["32", "64", "128", "256"])
    net.eval()
    torch.save({card.params["key"]: net.state_dict()}, tmp_path / "w.pth")
    m = restoration.load("codeformer", weights=tmp_path / "w.pth", device="cpu", dtype="float32")
    img = torch.rand(1, 3, 512, 512)
    with torch.inference_mode():
        want = ((net(img * 2 - 1, w=card.params["fidelity"], adain=True)[0] + 1) / 2).clamp(0, 1)
        got = m(img).image
        exported = m.module()(img)
    torch.testing.assert_close(got, want, rtol=0, atol=1e-4)
    torch.testing.assert_close(exported, got, rtol=0, atol=1e-6)
    with pytest.raises(InputError, match="512 x 512"):
        m(torch.rand(1, 3, 512, 640))
