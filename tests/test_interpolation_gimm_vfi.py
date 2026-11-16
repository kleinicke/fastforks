"""GIMM-VFI: the card's variants, the network on random weights, padding, the end times, export, the lookups."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn.functional as F

from fastforks import interpolation
from fastforks._core import registry
from fastforks._core.bench import synthetic_pair
from fastforks.interpolation.models.gimm_vfi.model import (
    Params,
    _banded,
    _config,
    _interpolate,
    _sample,
    _VolumeCorr,
)
from fastforks.interpolation.models.gimm_vfi.net import GIMMVFI_R
from fastforks.interpolation.models.gimm_vfi.net.modules.fi_components import NewMultiFlowDecoder

if TYPE_CHECKING:
    from pathlib import Path


def test_card_variants() -> None:
    cards = {v.id for c, v in registry.cards("interpolation").values() if c.family == "gimm_vfi"}
    assert cards == {"gimm-vfi-r", "gimm-vfi-r-p"}


@pytest.fixture(scope="module")
def net() -> GIMMVFI_R:
    torch.manual_seed(0)
    return GIMMVFI_R(_config()).eval()


@pytest.fixture(scope="module")
def weights(tmp_path_factory: pytest.TempPathFactory, net: GIMMVFI_R) -> Path:
    """A checkpoint with random weights in upstream's layout (`{"epoch", "state_dict"}`)."""
    path = tmp_path_factory.mktemp("gimm") / "gimmvfi_r_arb.pt"
    torch.save({"epoch": 0, "state_dict": net.state_dict()}, path)
    return path


@pytest.mark.slow
def test_odd_size_is_padded_and_exports(weights: Path) -> None:
    model = interpolation.load("gimm-vfi-r", weights=weights, device="cpu")
    a, b = synthetic_pair(131, 157, torch.device("cpu"), shift=3)  # replicate padded to 160 x 160 (centred), cropped
    out = model(a, b).frame
    assert out.shape == (1, 3, 131, 157)
    assert torch.isfinite(out).all()
    with torch.inference_mode():
        exported = model.module()(a, b)
    torch.testing.assert_close(exported, out, rtol=0, atol=1e-5)
    torch.testing.assert_close(model(a, b, timestep=0.0).frame, a, rtol=0, atol=1e-6)  # the end times: the inputs
    torch.testing.assert_close(model(a, b, timestep=1.0).frame, b, rtol=0, atol=1e-6)


@pytest.mark.slow
def test_forward_matches_upstream(net: GIMMVFI_R) -> None:
    """`_interpolate` (no host syncs, the time as a tensor) is the vendored upstream `forward` for one time."""
    a, b = synthetic_pair(128, 160, torch.device("cpu"), shift=4)  # the smallest size (1/64 correlation >= 2 px)
    xs = torch.stack((a, b), 2)
    with torch.inference_mode():
        coord = [(net.sample_coord_input(1, xs.shape[-2:], [0.3], device=xs.device), None)]
        want = net(xs, coord, t=[0.3 * torch.ones(1)], ds_factor=1.0)["imgt_pred"][0]
        got = _interpolate(net, a, b, torch.full((1,), 0.3), Params(iters=20, ds_factor=1.0), make=_VolumeCorr)
    assert torch.isfinite(want).all()
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)


def test_half_volume_sampling_is_grid_sample() -> None:
    """The float16-volume lookup (gathered corners, float32 weights) is `grid_sample` on the same values."""
    torch.manual_seed(0)
    vol = torch.randn(6, 1, 11, 13).half()
    xy = torch.rand(6, 81, 2) * torch.tensor([16.0, 14.0]) - 1.5  # inside and outside the volume
    g = torch.stack((2 * xy[..., 0] / 12 - 1, 2 * xy[..., 1] / 10 - 1), -1)
    want = F.grid_sample(vol.float(), g[:, None], align_corners=True)[:, 0, 0]
    torch.testing.assert_close(_sample(vol, xy), want, rtol=0, atol=1e-5)
    torch.testing.assert_close(_sample(vol.float(), xy), want, rtol=0, atol=1e-6)


def test_decoder_bands_are_one_pass() -> None:
    """Row bands of the decoder's `convblock` (its layout, few channels) are the one-pass result: `HALO` suffices."""
    torch.manual_seed(0)
    block = NewMultiFlowDecoder(16, 4).convblock.eval().double()  # float64: a halo of 16 rows is off by ~1e-12
    x = torch.randn(1, 49, 90, 12, dtype=torch.float64)
    with torch.inference_mode():
        torch.testing.assert_close(_banded(block, x, band_pixels=12 * 7), block(x), rtol=0, atol=1e-14)
