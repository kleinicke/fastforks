"""SwinIR: the card's variants, the SDPA attention with cached masks against the vendored upstream network, export."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import restoration
from fastforks._core import registry
from fastforks._core.bench import synthetic_image
from fastforks.restoration.models.swinir.net import SwinIR

if TYPE_CHECKING:
    from pathlib import Path

VARIANTS = {"swinir-m-x4": 4, "swinir-m-x2": 2, "swinir-m-x4-real": 4, "swinir-l-x4-real": 4}


def _cards() -> dict:
    return {v.id: (c, v) for c, v in registry.cards("restoration").values() if c.family == "swinir"}


def test_card_variants() -> None:
    cards = _cards()
    assert sorted(cards) == sorted(VARIANTS)
    for vid, scale in VARIANTS.items():
        card, variant = cards[vid]
        assert {**card.params, **variant.params}["scale"] == scale, vid


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:torch.meshgrid")  # upstream network, built here directly
@pytest.mark.parametrize("ref", ["swinir-m-x2", "swinir-l-x4-real"])  # pixelshuffle / nearest+conv, 1conv / 3conv
def test_matches_vendored_upstream(tmp_path: Path, ref: str) -> None:
    card, variant = _cards()[ref]
    p = {**card.params, **variant.params}
    torch.manual_seed(0)
    net = SwinIR(
        upscale=p["scale"], in_chans=3, img_size=p["img_size"], window_size=8, img_range=1.0, depths=p["depths"],
        embed_dim=p["embed_dim"], num_heads=p["heads"], mlp_ratio=2, upsampler=p["upsampler"],
        resi_connection=p["resi_connection"],
    ).eval()  # fmt: skip
    torch.save({p["key"]: net.state_dict()}, tmp_path / "w.pth")
    m = restoration.load(ref, weights=tmp_path / "w.pth", device="cpu", dtype="float32")
    h, w, s = 13, 21, p["scale"]
    img = synthetic_image(h, w, torch.device("cpu"))
    x = torch.cat([img, img.flip(2)], 2)[:, :, :16]  # main_test_swinir.py: to (size // 8 + 1) * 8
    x = torch.cat([x, x.flip(3)], 3)[:, :, :, :24]
    padded, region = m._runner.pad(img)
    assert torch.equal(padded, x)
    assert (region.top, region.left, region.height, region.width) == (0, 0, h, w)
    with torch.inference_mode():
        want = net(x)  # upstream: CPU masks per size, bias gathered per call, matmul + softmax attention
        raw = m._runner(x)["image"]
        exported = m.module()(img)  # scale, pad, network, crop, clamp
    torch.testing.assert_close(raw, want, rtol=1e-4, atol=1e-4 * float(want.abs().max()))
    got = m(img).image
    assert got.shape == (1, 3, h * s, w * s)
    torch.testing.assert_close(got, raw[..., : h * s, : w * s].clamp(0, 1), rtol=0, atol=1e-6)
    torch.testing.assert_close(exported, got, rtol=0, atol=1e-6)


@pytest.mark.filterwarnings("ignore:torch.meshgrid")
def test_last_group_scale() -> None:
    """The float16 runner's scaled last residual group: the same features after the LayerNorm (its eps scaled)."""
    from fastforks._core.patch import override  # noqa: PLC0415
    from fastforks.restoration.models.swinir import model  # noqa: PLC0415

    torch.manual_seed(0)
    net = SwinIR(upscale=2, img_size=16, window_size=8, img_range=1.0, depths=[2, 2], embed_dim=24, num_heads=[2, 2],
                 mlp_ratio=2, upsampler="pixelshuffle", resi_connection="3conv").eval()  # fmt: skip
    x = torch.rand(1, 3, 16, 24)
    with torch.inference_mode():
        want = net.forward_features(net.conv_first(x))
        last = net.layers[-1]
        conv = last.conv[-1]
        conv.weight.div_(model._LAST_SCALE)
        conv.bias.div_(model._LAST_SCALE)
        override(last, model._rstb_scaled)
        net.norm.eps /= model._LAST_SCALE**2
        got = net.forward_features(net.conv_first(x))
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
