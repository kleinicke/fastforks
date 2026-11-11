"""FlowSeek: the card's variants, the motion bases, the network on random weights (batch, odd size), export."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import flow
from fastforks._core import registry
from fastforks._core.bench import synthetic_pair
from fastforks.flow.models.flowseek.net import FlowSeek, motion_bases

if TYPE_CHECKING:
    from pathlib import Path

VARIANTS = {  # id -> training data
    "flowseek-t": ["tartanair", "flyingchairs", "sceneflow", "sintel-flow", "kitti", "hd1k"],
    "flowseek-t-things": ["tartanair", "flyingchairs", "sceneflow"],
    "flowseek-t-things-notartan": ["flyingchairs", "sceneflow"],
}


def _pair(height: int, width: int, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """A textured pair in [0, 1] on the CPU; item `i` of a batch moves by `4 * (i + 1)` px."""
    pairs = [synthetic_pair(height, width, torch.device("cpu"), shift=4 * (i + 1)) for i in range(batch)]
    return torch.cat([a for a, _ in pairs]), torch.cat([b for _, b in pairs])


def test_card_variants() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("flow").values() if c.family == "flowseek"}
    assert sorted(cards) == sorted(VARIANTS)
    for vid, data in VARIANTS.items():
        card, variant = cards[vid]
        assert list(variant.licence.get("training_data", card.licence.training_data)) == data, vid
        assert card.options["iters"] == 4, vid
        assert card.io.pad_mode == "none", vid  # the network pads itself
    assert set(flow.list_models()) >= set(VARIANTS)


def test_motion_bases() -> None:
    torch.manual_seed(0)
    disp = torch.rand(2, 1, 24, 40) + 0.1
    bases = motion_bases(disp)
    assert bases.shape == (2, 16, 24, 40)
    pairs = bases.view(2, 8, 2, 24, 40)
    norms = torch.linalg.vector_norm(pairs[:, 3:], dim=(2, 3, 4))
    torch.testing.assert_close(norms, torch.ones(2, 5), rtol=0, atol=1e-5)  # rotation terms: unit norm per image
    # translation along x: (-1, 0) / sqrt(H * W), times 2 * disp
    torch.testing.assert_close(pairs[:, 0, 0], -2 * disp[:, 0] / (24 * 40) ** 0.5, rtol=0, atol=1e-6)
    assert not pairs[:, 0, 1].any()
    torch.testing.assert_close(motion_bases(disp[1:]), bases[1:], rtol=0, atol=1e-6)  # per image


@pytest.mark.slow
def test_net_shape() -> None:
    torch.manual_seed(0)
    net = FlowSeek().eval()
    a, b = _pair(131, 150, batch=2)  # neither a multiple of 8: the network pads and crops
    with torch.inference_mode():
        out = net(a, b, 2)
        one = net(a[1:], b[1:], 2)
    assert out.shape == (2, 2, 131, 150)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[1:], one, rtol=0, atol=1e-3)  # the items of a batch do not mix


@pytest.fixture(scope="module")
def weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A checkpoint with random weights, as `load(weights=...)` takes it."""
    torch.manual_seed(0)
    sd = FlowSeek().state_dict()
    sd |= {k.replace(".bn3.", ".downsample.1."): v for k, v in sd.items() if ".bn3." in k}  # as the authors' files
    path = tmp_path_factory.mktemp("flowseek") / "random.pth"
    torch.save(sd, path)
    return path


@pytest.mark.slow
def test_odd_size_and_export(weights: Path) -> None:
    model = flow.load("flowseek-t", weights=weights, device="cpu", options={"iters": 1})
    a, b = _pair(131, 150)
    out = model(a, b).flow
    assert out.shape == (1, 2, 131, 150)
    assert torch.isfinite(out).all()
    with torch.inference_mode():
        exported = model.module()(a, b)  # export does what the call does
    torch.testing.assert_close(exported, out, rtol=0, atol=1e-5)
