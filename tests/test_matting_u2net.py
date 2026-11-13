"""U²-Net: the card's variants and the runner's pre- and post-processing around the vendored network."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn.functional as F

from fastforks import matting
from fastforks._core import registry
from fastforks._core.bench import synthetic_image
from fastforks.matting.models.u2net.net import U2NET, U2NETP

if TYPE_CHECKING:
    from pathlib import Path


def test_card() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("matting").values() if c.family == "u2net"}
    assert sorted(cards) == ["u2net", "u2net-human-seg", "u2netp"]
    assert {v: {**c.params, **x.params}["arch"] for v, (c, x) in cards.items()}["u2netp"] == "u2netp"
    card = cards["u2net"][0]
    assert card.licence.code == card.licence.weights == "Apache-2.0"


@pytest.mark.slow
@pytest.mark.parametrize(("ref", "cls"), [("u2netp", U2NETP), ("u2net", U2NET)])
def test_matches_u2net_test(tmp_path: Path, ref: str, cls: type) -> None:
    """`u2net_test.py` on float tensors: 320 x 320, / max, ImageNet, d0, normPRED, bilinear back."""
    torch.manual_seed(0)
    net = cls(3, 1).eval()
    torch.save(net.state_dict(), tmp_path / "w.pth")
    m = matting.load(ref, weights=tmp_path / "w.pth", device="cpu", dtype="float32")
    img = synthetic_image(45, 67, torch.device("cpu"))
    x = F.interpolate(img, (320, 320), mode="bilinear", align_corners=False, antialias=True)
    x = x / x.max()
    mean, std = torch.tensor([0.485, 0.456, 0.406])[:, None, None], torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    with torch.inference_mode():
        d0 = net((x - mean) / std)[0]
        want = F.interpolate((d0 - d0.min()) / (d0.max() - d0.min()), (45, 67), mode="bilinear")[0, 0]
        got = m(img).alpha[0]
        exported = m.module()(img)[0]
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)
    torch.testing.assert_close(exported, got, rtol=0, atol=1e-6)
