"""EMA-VFI: the card's variants, the forward against vendored upstream on random weights, TTA, padding, export."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import interpolation
from fastforks._core import registry
from fastforks._core.bench import synthetic_pair
from fastforks.interpolation.models.ema_vfi.model import Params, _config, _frame, _interpolate
from fastforks.interpolation.models.ema_vfi.net import MotionFormer, MultiScaleFlow

if TYPE_CHECKING:
    from pathlib import Path

SMALL = Params(width=16, depths=(2, 2, 2, 2, 2), window=7, tta=False, cuda_graphs=False)


def test_card_variants() -> None:
    cards = {v.id for c, v in registry.cards("interpolation").values() if c.family == "ema_vfi"}
    assert cards == {"ema-vfi", "ema-vfi-small", "ema-vfi-t", "ema-vfi-small-t"}


@pytest.fixture(scope="module")
def net() -> MultiScaleFlow:
    torch.manual_seed(0)
    backbone, flow = _config(SMALL)
    return MultiScaleFlow(MotionFormer(**backbone), **flow).eval()


@pytest.mark.parametrize("size", [(64, 96), (96, 160)])  # 1/16: 4x6 (padded to 7x7 windows), 6x10 (+ shifts)
def test_forward_matches_upstream(net: MultiScaleFlow, size: tuple[int, int]) -> None:
    """`_interpolate` (masks per shape, one attention product, float32 flow) is the vendored upstream forward."""
    a, b = synthetic_pair(*size, torch.device("cpu"), shift=3)
    with torch.inference_mode():
        want = net(torch.cat((a, b), 1), timestep=0.3)[3]
        got = _interpolate(net, a, b, torch.full((1, 1, 1, 1), 0.3), {})
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)


def test_tta_is_upstream_fast_tta(net: MultiScaleFlow) -> None:
    """`tta`: the mean of the pair and its 180-degree rotation (flipped back), as upstream `fast_TTA`."""
    a, b = synthetic_pair(64, 96, torch.device("cpu"), shift=3)
    t = torch.full((1, 1, 1, 1), 0.5)
    with torch.inference_mode():
        x = torch.cat((a.flip(1), b.flip(1)), 1)  # upstream BGR
        p = net(torch.cat((x, x.flip(2, 3))), timestep=0.5)[3]
        want = ((p[:1] + p[1:].flip(2, 3)) / 2).flip(1)
        got = _frame(net, (a, b), t, {}, tta=True)
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)


@pytest.fixture(scope="module")
def weights(tmp_path_factory: pytest.TempPathFactory, net: MultiScaleFlow) -> Path:
    """A checkpoint with random weights in upstream's layout (DDP `module.` prefixes)."""
    path = tmp_path_factory.mktemp("ema") / "ema-vfi-ours_small.pkl"
    torch.save({f"module.{k}": v for k, v in net.state_dict().items()}, path)
    return path


def test_odd_size_is_padded_and_exports(weights: Path) -> None:
    model = interpolation.load("ema-vfi-small", weights=weights, device="cpu")
    a, b = synthetic_pair(71, 101, torch.device("cpu"), shift=3)  # replicate padded to 96 x 128 (centred), cropped
    out = model(a, b).frame
    assert out.shape == (1, 3, 71, 101)
    assert torch.isfinite(out).all()
    with torch.inference_mode():
        exported = model.module()(a, b)
    torch.testing.assert_close(exported, out, rtol=0, atol=1e-5)
