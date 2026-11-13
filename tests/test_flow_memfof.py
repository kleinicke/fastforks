"""MEMFOF: the card's variants, the network on random weights (batch, odd sizes, 2x scale), the kernel lookup."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import flow
from fastforks._core import registry
from fastforks._core.bench import synthetic_pair
from fastforks.flow.models.memfof.model import _kernel_lookup
from fastforks.flow.models.memfof.net.corr import CorrBlock
from fastforks.flow.models.memfof.net.model import MEMFOF

if TYPE_CHECKING:
    from pathlib import Path

TSKH = ["tartanair", "sceneflow", "sintel-flow", "kitti", "hd1k"]
VARIANTS = {
    "memfof": (0, TSKH),
    "memfof-kitti": (1, TSKH),
    "memfof-sintel": (1, TSKH),
    "memfof-spring": (0, [*TSKH, "spring"]),
}


def _pair(height: int, width: int, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """A textured pair in [0, 1] on the CPU; item `i` of a batch moves by `4 * (i + 1)` px."""
    pairs = [synthetic_pair(height, width, torch.device("cpu"), shift=4 * (i + 1)) for i in range(batch)]
    return torch.cat([a for a, _ in pairs]), torch.cat([b for _, b in pairs])


def test_card_variants() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("flow").values() if c.family == "memfof"}
    assert sorted(cards) == sorted(VARIANTS)
    for vid, (scale, data) in VARIANTS.items():
        card, variant = cards[vid]
        assert list(variant.licence.get("training_data", card.licence.training_data)) == data, vid
        assert card.options["iters"] == 8, vid
        assert {**card.params, **variant.params}["scale"] == scale, vid
    assert set(flow.list_models()) >= set(VARIANTS)


@pytest.fixture(scope="module")
def weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A checkpoint with random weights, as `load(weights=...)` takes it."""
    from safetensors.torch import save_file  # noqa: PLC0415

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("memfof") / "random.safetensors"
    save_file({k: v.contiguous() for k, v in MEMFOF().state_dict().items()}, path)  # the authors' file layout
    return path


@pytest.mark.slow
def test_batch_odd_size_scale_and_export(weights: Path) -> None:
    model = flow.load("memfof", weights=weights, device="cpu", options={"iters": 1})
    a, b = _pair(259, 275, batch=2)  # the 256 px minimum, neither a multiple of 16: replicate padding, cropped again
    out = model(a, b).flow
    assert out.shape == (2, 2, 259, 275)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[1:], model(a[1:], b[1:]).flow, rtol=0, atol=1e-4)  # the items do not mix
    a, b = _pair(256, 288)
    with torch.inference_mode():
        exported = model.module()(a, b)  # export does what the call does
    torch.testing.assert_close(exported, model(a, b).flow, rtol=0, atol=1e-5)
    up = flow.load("memfof-kitti", weights=weights, device="cpu", options={"iters": 1})  # frames upsampled 2x
    assert up(a, b).flow.shape == (1, 2, 256, 288)


@pytest.mark.gpu
@pytest.mark.parametrize(("height", "width", "batch"), [(17, 21, 2), (68, 120, 1)])  # odd cells; the 1080p cells
def test_kernel_lookup_matches_portable(height: int, width: int, batch: int) -> None:
    """The on-the-fly windows equal upstream's volume lookup (fp32) up to fp16 operands and output rounding."""
    from fastforks.kernels import raft_flow  # noqa: PLC0415

    torch.manual_seed(0)
    f1, f2 = (torch.randn(batch, 1024, height, width, device="cuda").half() for _ in range(2))
    coords = torch.randn(batch, 2, height, width, device="cuda") * 20  # windows far off and over the border
    coords += torch.stack(torch.meshgrid(torch.arange(width), torch.arange(height), indexing="xy")).cuda()
    want = CorrBlock(f1.float(), f2.float(), 4, 4)(coords)
    got = _kernel_lookup(raft_flow.load(torch.device("cuda")), f1, f2)(coords)
    assert got.shape == want.shape
    torch.testing.assert_close(got.float(), want, rtol=0, atol=4e-3)  # fp16 pooling and output of |corr| < 8
