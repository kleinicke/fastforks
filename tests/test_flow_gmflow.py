"""GMFlow: the card's variants, the forward pass on random weights (batch, three configurations), chunked attention,
padding of odd sizes, export."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import flow
from fastforks._core import registry
from fastforks._core.bench import synthetic_pair
from fastforks.flow.models.gmflow import model as family
from fastforks.flow.models.gmflow.net.unimatch import UniMatch
from fastforks.flow.models.gmflow.net.utils import normalize_img

if TYPE_CHECKING:
    from pathlib import Path

VARIANTS = {  # id -> (scales, refine, pad multiple)
    "gmflow-scale1-things": (1, 0, 16),
    "gmflow-scale1-mixdata": (1, 0, 16),
    "gmflow-scale2-things": (2, 0, 32),
    "gmflow-scale2-sintel": (2, 0, 32),
    "gmflow-scale2-mixdata": (2, 0, 32),
    "gmflow-scale2-regrefine6-things": (2, 6, 32),
    "gmflow-scale2-regrefine6-sintelft": (2, 6, 32),
    "gmflow-scale2-regrefine6-kitti": (2, 6, 32),
    "gmflow-scale2-regrefine6-mixdata": (2, 6, 32),
}


def _pair(height: int, width: int, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """A textured pair in [0, 1] on the CPU; item `i` of a batch moves by `4 * (i + 1)` px."""
    pairs = [synthetic_pair(height, width, torch.device("cpu"), shift=4 * (i + 1)) for i in range(batch)]
    return torch.cat([a for a, _ in pairs]), torch.cat([b for _, b in pairs])


def _net(scales: int, refine: int) -> UniMatch:
    torch.manual_seed(0)
    return UniMatch(num_scales=scales, upsample_factor=8 // scales, reg_refine=refine > 0, task="flow").eval()


def test_card_variants() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("flow").values() if c.family == "gmflow"}
    assert sorted(cards) == sorted(VARIANTS)
    for vid, (scales, refine, multiple) in VARIANTS.items():
        card, variant = cards[vid]
        params = card.params | variant.params
        assert (params["scales"], params["refine"], variant.io.pad_multiple) == (scales, refine, multiple), vid
        assert ("unstated" in variant.licence.get("training_data", ())) == vid.endswith(("-mixdata", "-kitti")), vid
    assert set(flow.list_models()) >= set(VARIANTS)


@pytest.mark.parametrize(("scales", "refine"), [(1, 0), (2, 0), (2, 2)])
def test_forward_shape(scales: int, refine: int) -> None:
    net = _net(scales, refine)
    params = family.Params(scales=scales, refine=refine)
    a, b = _pair(64, 96, batch=2)
    with torch.inference_mode():
        out = family.forward(net, a, b, params, fp16=False)
        one = family.forward(net, a[1:], b[1:], params, fp16=False)
    assert out.shape == (2, 2, 64, 96)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[1:], one, rtol=0, atol=1e-4)  # the items of a batch do not mix


def test_forward_is_upstreams() -> None:
    """Our forward pass against the vendored `UniMatch.forward` (same weights, upstream's evaluation settings)."""
    net = _net(2, 3)
    a, b = _pair(64, 96)
    with torch.inference_mode():
        ours = family.forward(net, *normalize_img(a * 255, b * 255), family.Params(scales=2, refine=3), fp16=False)
        ref = net(
            a * 255,
            b * 255,
            attn_type="swin",
            attn_splits_list=[2, 8],
            corr_radius_list=[-1, 4],
            prop_radius_list=[-1, 1],
            num_reg_refine=3,
            task="flow",
        )["flow_preds"][-1]
    torch.testing.assert_close(ours, ref, rtol=0, atol=1e-4)


def test_chunked_attention_is_the_same(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    query, key, value = torch.randn(2, 96, 16), torch.randn(2, 96, 16), torch.randn(2, 96, 2)
    whole = family.attend(query, key, value)
    monkeypatch.setattr(family, "CHUNK_ELEMENTS", 2 * 96 * 10)  # 10 rows per chunk
    torch.testing.assert_close(family.attend(query, key, value), whole, rtol=0, atol=1e-6)


@pytest.fixture(scope="module")
def weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A scale1 checkpoint with random weights, as `load(weights=...)` takes it."""
    path = tmp_path_factory.mktemp("gmflow") / "random.pth"
    torch.save({"model": _net(1, 0).state_dict()}, path)  # the layout of the authors' files
    return path


def test_odd_size_is_padded_and_exports(weights: Path) -> None:
    model = flow.load("gmflow-scale1-things", weights=weights, device="cpu")
    a, b = _pair(67, 93)  # neither a multiple of 16: replicate padding, cropped again
    out = model(a, b).flow
    assert out.shape == (1, 2, 67, 93)
    assert torch.isfinite(out).all()
    a, b = _pair(64, 96)
    with torch.inference_mode():
        exported = model.module()(a, b)  # export does what the call does
    torch.testing.assert_close(exported, model(a, b).flow, rtol=0, atol=1e-5)
