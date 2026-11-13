"""HAT: the card's variants, the SDPA attentions with cached masks against the vendored upstream network, export."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn.functional as F

from fastforks import restoration
from fastforks._core import registry
from fastforks._core.bench import synthetic_image
from fastforks.restoration.models.hat.net import HAT

if TYPE_CHECKING:
    from pathlib import Path

VARIANTS = ("hat-x4", "hat-s-x4", "real-hat-gan-x4")


def _cards() -> dict:
    return {v.id: (c, v) for c, v in registry.cards("restoration").values() if c.family == "hat"}


def _net(p: dict) -> HAT:
    return HAT(
        upscale=p["scale"], in_chans=3, img_size=p["img_size"], window_size=p["window"],
        compress_ratio=p["compress_ratio"], squeeze_factor=p["squeeze_factor"], conv_scale=0.01, overlap_ratio=0.5,
        img_range=1.0, depths=p["depths"], embed_dim=p["embed_dim"], num_heads=p["heads"], mlp_ratio=2,
        upsampler="pixelshuffle", resi_connection="1conv",
    ).eval()  # fmt: skip


def test_card_variants() -> None:
    assert sorted(_cards()) == sorted(VARIANTS)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:torch.meshgrid")  # upstream network, built here directly
def test_matches_vendored_upstream(tmp_path: Path) -> None:
    card, variant = _cards()["hat-s-x4"]
    p = {**card.params, **variant.params}
    torch.manual_seed(0)
    net = _net(p)
    for m in net.modules():  # random tables: upstream initializes them to small values
        if hasattr(m, "relative_position_bias_table"):
            torch.nn.init.normal_(m.relative_position_bias_table, std=1.0)
    torch.save({p["key"]: net.state_dict()}, tmp_path / "w.pth")
    m = restoration.load("hat-s-x4", weights=tmp_path / "w.pth", device="cpu", dtype="float32")
    h, w, s = 21, 37, p["scale"]
    img = synthetic_image(h, w, torch.device("cpu"))
    x = F.pad(img, (0, 48 - w, 0, 32 - h), "reflect")  # hat_model.py pre_process: to a multiple of 16
    padded, region = m._runner.pad(img)
    assert torch.equal(padded, x)
    assert (region.top, region.left, region.height, region.width) == (0, 0, h, w)
    with torch.inference_mode():
        want = net(x)  # upstream: CPU mask per call, biases gathered per call, matmul + softmax attention
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
    from fastforks.restoration.models.hat import model  # noqa: PLC0415

    torch.manual_seed(0)
    p = {"scale": 2, "img_size": 16, "window": 16, "compress_ratio": 3, "squeeze_factor": 6, "depths": [2, 2],
         "embed_dim": 24, "heads": [2, 2]}  # fmt: skip
    net = _net(p)
    x = torch.rand(1, 3, 32, 48)
    with torch.inference_mode():
        want = net.forward_features(net.conv_first(x))
        last = net.layers[-1]
        for t in last.get_submodule("conv").parameters():
            t.div_(model._LAST_SCALE)
        override(last, model._rhag_scaled)
        net.norm.eps /= model._LAST_SCALE**2
        got = net.forward_features(net.conv_first(x))
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
