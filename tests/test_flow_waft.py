"""WAFT: the card's variants, the shared timm ResNet stages, the network on random weights (batch, odd size), export."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import flow
from fastforks._core import registry
from fastforks._core.bench import synthetic_pair
from fastforks.flow.models.waft.model import _net
from fastforks.shared.timm_blocks import resnet_layers

if TYPE_CHECKING:
    from pathlib import Path

TSKH = ["tartanair", "flyingchairs", "sceneflow", "sintel-flow", "kitti", "hd1k"]
VARIANTS = {"waft-a1": ["tartanair", "flyingchairs", "sceneflow"], "waft-a1-kitti": TSKH, "waft-a1-sintel": TSKH}


def _pair(height: int, width: int, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """A textured pair in [0, 1] on the CPU; item `i` of a batch moves by `4 * (i + 1)` px."""
    pairs = [synthetic_pair(height, width, torch.device("cpu"), shift=4 * (i + 1)) for i in range(batch)]
    return torch.cat([a for a, _ in pairs]), torch.cat([b for _, b in pairs])


def test_card_variants() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("flow").values() if c.family == "waft"}
    assert sorted(cards) == sorted(VARIANTS)
    for vid, data in VARIANTS.items():
        card, variant = cards[vid]
        assert list(variant.licence.get("training_data", card.licence.training_data)) == data, vid
        assert card.options["iters"] == 5, vid
        assert card.io.pad_mode == "none", vid  # the network pads itself
    assert set(flow.list_models()) >= set(VARIANTS)


@pytest.mark.slow
def test_resnet_stages_match_timm() -> None:
    timm = pytest.importorskip("timm")
    ref = timm.create_model("resnet18", pretrained=False)
    want = {
        f"{n}.{k}": v.shape
        for n in ("layer1", "layer2", "layer3", "layer4")
        for k, v in ref.get_submodule(n).state_dict().items()
    }
    stages = torch.nn.ModuleDict(zip(("layer1", "layer2", "layer3", "layer4"), resnet_layers(), strict=True))
    assert {k: v.shape for k, v in stages.state_dict().items()} == want
    stages.load_state_dict({f"{n}.{k}": v for n in stages for k, v in ref.get_submodule(n).state_dict().items()})
    x = torch.randn(1, 64, 32, 32)
    ref.eval(), stages.eval()
    with torch.inference_mode():
        a, b = x, x
        for n in stages:
            a, b = stages[n](a), ref.get_submodule(n)(b)
    torch.testing.assert_close(a, b, rtol=0, atol=1e-5)


@pytest.fixture(scope="module")
def weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A checkpoint with random weights, as `load(weights=...)` takes it."""
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("waft") / "random.pth"
    torch.save(_net().state_dict(), path)
    return path


@pytest.mark.slow
def test_odd_size_batch_and_export(weights: Path) -> None:
    model = flow.load("waft-a1", weights=weights, device="cpu", options={"iters": 1})
    a, b = _pair(115, 130, batch=2)  # not multiples of 112: the network pads and crops
    out = model(a, b).flow
    assert out.shape == (2, 2, 115, 130)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(model(a[1:], b[1:]).flow, out[1:], rtol=0, atol=1e-4)  # the items do not mix
    with torch.inference_mode():
        exported = model.module()(a[:1], b[:1])  # export does what the call does
    torch.testing.assert_close(exported, out[:1], rtol=0, atol=1e-4)
