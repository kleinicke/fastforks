"""MemFlow: the card's variants, the network on random weights (batch, odd sizes, export), the kernel lookup."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import flow
from fastforks._core import registry
from fastforks._core.bench import synthetic_pair
from fastforks.flow.models.memflow.model import _kernel_lookup
from fastforks.flow.models.memflow.net import CorrBlock, MemFlowNet

if TYPE_CHECKING:
    from pathlib import Path

CT = ["flyingchairs", "sceneflow"]
CTSKH = [*CT, "sintel-flow", "kitti", "hd1k"]
VARIANTS = {"memflow": (6750.0, CT), "memflow-sintel": (6624.0, CTSKH), "memflow-kitti": (6480.0, CTSKH)}


def _pair(height: int, width: int, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """A textured pair in [0, 1] on the CPU; item `i` of a batch moves by `4 * (i + 1)` px."""
    pairs = [synthetic_pair(height, width, torch.device("cpu"), shift=4 * (i + 1)) for i in range(batch)]
    return torch.cat([a for a, _ in pairs]), torch.cat([b for _, b in pairs])


def test_card_variants() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("flow").values() if c.family == "memflow"}
    assert sorted(cards) == sorted(VARIANTS)
    for vid, (length, data) in VARIANTS.items():
        card, variant = cards[vid]
        assert list(variant.licence.get("training_data", card.licence.training_data)) == data, vid
        assert card.options["iters"] == 15, vid
        assert {**card.params, **variant.params}["train_avg_length"] == length, vid
    assert set(flow.list_models()) >= set(VARIANTS)


@pytest.fixture(scope="module")
def weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A checkpoint with random weights in the authors' layout (`{"model": ...}` with the unused position embedding)."""
    torch.manual_seed(0)
    sd = {k: v.contiguous() for k, v in MemFlowNet(6750.0).state_dict().items()}
    sd["att.pos_emb.rel_height.weight"] = torch.zeros(1)  # dropped by `convert`
    path = tmp_path_factory.mktemp("memflow") / "random.pth"
    torch.save({"model": sd}, path)
    return path


@pytest.mark.slow
def test_batch_odd_size_and_export(weights: Path) -> None:
    model = flow.load("memflow", weights=weights, device="cpu", options={"iters": 2})
    a, b = _pair(131, 141, batch=2)  # above the 128 px minimum, not multiples of 8: replicate padding, cropped again
    out = model(a, b).flow
    assert out.shape == (2, 2, 131, 141)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[1:], model(a[1:], b[1:]).flow, rtol=0, atol=1e-4)  # the items do not mix
    a, b = _pair(128, 160)
    with torch.inference_mode():
        exported = model.module()(a, b)  # export does what the call does
    torch.testing.assert_close(exported, model(a, b).flow, rtol=0, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.parametrize(("height", "width", "batch"), [(17, 21, 2), (135, 240, 1)])  # odd cells; the 1080p cells
def test_kernel_lookup_matches_portable(height: int, width: int, batch: int) -> None:
    """The on-the-fly windows equal upstream's volume lookup (fp32) up to fp16 operands and output rounding."""
    from fastforks.kernels import raft_flow  # noqa: PLC0415

    torch.manual_seed(0)
    f1, f2 = (torch.randn(batch, 256, height, width, device="cuda").half() for _ in range(2))
    coords = torch.randn(batch, 2, height, width, device="cuda") * 20  # windows far off and over the border
    coords += torch.stack(torch.meshgrid(torch.arange(width), torch.arange(height), indexing="xy")).cuda()
    want = CorrBlock(f1.float(), f2.float())(coords)
    got = _kernel_lookup(raft_flow.load(torch.device("cuda")), f1, f2)(coords)
    assert got.shape == want.shape
    torch.testing.assert_close(got.float(), want, rtol=0, atol=4e-3)  # fp16 pooling and output of |corr| < 8
