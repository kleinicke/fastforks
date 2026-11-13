"""IS-Net: the card's variants and the runner's pre- and post-processing around the vendored network."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn.functional as F

from fastforks import matting
from fastforks._core import registry
from fastforks._core.bench import synthetic_image
from fastforks.matting.models.isnet.net import ISNetDIS

if TYPE_CHECKING:
    from pathlib import Path


def test_card() -> None:
    cards = {v.id: c for c, v in registry.cards("matting").values() if c.family == "isnet"}
    assert sorted(cards) == ["isnet", "isnet-general-use"]
    card = cards["isnet"]
    assert card.licence.code == card.licence.weights == "Apache-2.0"
    assert card.params["size"] == 1024


@pytest.mark.slow
def test_matches_inference_py(tmp_path: Path) -> None:
    """`IS-Net/Inference.py` on float tensors: bilinear to 1024, mean 0.5 / std 1, d1, bilinear back, min-max."""
    torch.manual_seed(0)
    net = ISNetDIS(3, 1).eval()
    torch.save(net.state_dict(), tmp_path / "w.pth")
    m = matting.load("isnet", weights=tmp_path / "w.pth", device="cpu", dtype="float32")
    img = synthetic_image(45, 67, torch.device("cpu"))
    x = F.interpolate(img, (1024, 1024), mode="bilinear", align_corners=False)
    with torch.inference_mode():
        d1 = net(x - 0.5)[0][0]
        y = F.interpolate(d1, (45, 67), mode="bilinear")[0, 0]
        want = (y - y.min()) / (y.max() - y.min())
        got = m(img).alpha[0]
        exported = m.module()(img)[0]
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)
    torch.testing.assert_close(exported, got, rtol=0, atol=1e-6)
