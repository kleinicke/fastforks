"""NeuFlow v2: the card's variants, the network on random weights (batch, non-square), padding of odd sizes, export."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import flow
from fastforks._core import registry
from fastforks._core.bench import synthetic_pair
from fastforks.flow.models.neuflow_v2.net import NeuFlow

if TYPE_CHECKING:
    from pathlib import Path

VARIANTS = {  # id -> training data
    "neuflow-v2": ["sceneflow", "sintel-flow", "kitti", "hd1k"],
    "neuflow-v2-sintel": ["sceneflow", "sintel-flow", "kitti", "hd1k"],  # upstream's "sintel" stage mixes them
    "neuflow-v2-things": ["sceneflow"],
}


def _pair(height: int, width: int, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """A textured pair in [0, 1] on the CPU; item `i` of a batch moves by `4 * (i + 1)` px."""
    pairs = [synthetic_pair(height, width, torch.device("cpu"), shift=4 * (i + 1)) for i in range(batch)]
    return torch.cat([a for a, _ in pairs]), torch.cat([b for _, b in pairs])


def test_card_variants() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("flow").values() if c.family == "neuflow_v2"}
    assert sorted(cards) == sorted(VARIANTS)
    for vid, data in VARIANTS.items():
        card, variant = cards[vid]
        assert list(variant.licence.get("training_data", card.licence.training_data)) == data, vid
        assert card.options["iters"] == 8, vid
    assert set(flow.list_models()) >= set(VARIANTS)


def test_net_shape() -> None:
    torch.manual_seed(0)
    net = NeuFlow().eval()
    a, b = _pair(64, 96, batch=2)
    with torch.inference_mode():
        out = net(a, b, 2)
        one = net(a[1:], b[1:], 2)
    assert out.shape == (2, 2, 64, 96)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[1:], one, rtol=0, atol=1e-4)  # the items of a batch do not mix


@pytest.fixture(scope="module")
def weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A checkpoint with random weights, as `load(weights=...)` takes it."""
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("neuflow_v2") / "random.pth"
    torch.save({"model": NeuFlow().state_dict()}, path)  # the layout of the authors' files
    return path


def test_odd_size_is_padded_and_exports(weights: Path) -> None:
    model = flow.load("neuflow-v2", weights=weights, device="cpu", options={"iters": 1})
    a, b = _pair(67, 93)  # neither a multiple of 16: replicate padding, cropped again
    out = model(a, b).flow
    assert out.shape == (1, 2, 67, 93)
    assert torch.isfinite(out).all()
    a, b = _pair(64, 96)
    with torch.inference_mode():
        exported = model.module()(a, b)  # export does what the call does
    torch.testing.assert_close(exported, model(a, b).flow, rtol=0, atol=1e-5)
