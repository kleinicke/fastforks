"""NeuFlow v2: the card's variants, the network on random weights (batch, non-square), padding of odd sizes, export."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import flow
from fastforks._core import registry
from fastforks._core.bench import synthetic_pair
from fastforks.flow.models.neuflow_v2 import net as nf
from fastforks.flow.models.neuflow_v2.model import _kernel_lookup
from fastforks.flow.models.neuflow_v2.net import NeuFlow
from fastforks.kernels import raft_flow

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


@pytest.mark.parametrize(("height", "width", "batch"), [(5, 7, 2), (12, 9, 1)])
def test_half_lookup_matches_grid_sample(height: int, width: int, batch: int) -> None:
    """The fp16 volume's gather lookup equals `grid_sample` on the same values, borders and far-off windows included."""
    torch.manual_seed(0)
    features = torch.randn(2 * batch, nf.FEATURE, height, width).half()
    flow_ = torch.randn(batch, 2, height, width) * 6
    flow_[0, :, 0, 0] = torch.tensor([width - 1.0, 0.0])  # exactly on the last column
    lookup = nf.volume_lookup(features)
    corr = lookup.args[0]
    assert corr.dtype == torch.float16
    want = nf._lookup(corr.float().view(-1, 1, height, width), flow_)
    torch.testing.assert_close(lookup(flow_), want, rtol=0, atol=1e-5)
    ref = nf.volume_lookup(features.float()).args[0].view(corr.shape)
    torch.testing.assert_close(corr.float(), ref, rtol=2e-3, atol=2e-3)  # one fp16 rounding of the volume


@pytest.mark.gpu
@pytest.mark.parametrize(("height", "width", "batch"), [(17, 21, 2), (136, 240, 1)])  # odd cells; the 1080p cells
def test_kernel_lookup_matches_portable(height: int, width: int, batch: int) -> None:
    """The on-the-fly window equals the volume lookup (fp16 operands, fp32 sums) up to the fp16 output rounding."""
    torch.manual_seed(0)
    features = torch.randn(2 * batch, nf.FEATURE, height, width, device="cuda").half()
    flow_ = torch.randn(batch, 2, height, width, device="cuda") * 20  # windows far off and over the border
    want = nf.volume_lookup(features.float())(flow_)  # the fp32 volume of the same fp16 operands
    got = _kernel_lookup(raft_flow.load(torch.device("cuda")), features)(flow_)
    assert got.shape == want.shape
    torch.testing.assert_close(got.float(), want.half().float(), rtol=0, atol=2e-3)  # fp16 ulp of |corr| < 4
