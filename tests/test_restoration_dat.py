"""DAT: the card's variants, the SDPA attention with cached biases / masks and folded BatchNorms against the vendored
upstream network, the float16 last-group scaling, export."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import restoration
from fastforks._core import registry
from fastforks._core.bench import synthetic_image
from fastforks.restoration.models.dat.net import DAT

if TYPE_CHECKING:
    from pathlib import Path

VARIANTS = ("dat-x4", "dat-2-x4", "dat-light-x4")


def _cards() -> dict:
    return {v.id: (c, v) for c, v in registry.cards("restoration").values() if c.family == "dat"}


def _net(p: dict) -> DAT:
    return DAT(
        upscale=p["scale"], in_chans=3, img_size=p["img_size"], img_range=1.0, split_size=p["split"], depth=p["depths"],
        embed_dim=p["embed_dim"], num_heads=p["heads"], expansion_factor=p["expansion"], resi_connection=p["resi"],
        upsampler=p["upsampler"],
    ).eval()  # fmt: skip


def _randomize(net: DAT) -> None:
    """Random BatchNorm statistics (upstream initializes them to the identity), so the folding is tested."""
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.normal_(0, 0.5)
            m.running_var.uniform_(0.5, 2)
            m.weight.data.normal_(1, 0.2)
            m.bias.data.normal_(0, 0.2)


def test_card_variants() -> None:
    assert sorted(_cards()) == sorted(VARIANTS)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:torch.meshgrid")  # upstream network, built here directly
def test_matches_vendored_upstream(tmp_path: Path) -> None:
    ref = "dat-light-x4"  # the small variant: 18 blocks of both attentions, shifted-window masks, BatchNorms
    card, variant = _cards()[ref]
    p = {**card.params, **variant.params}
    torch.manual_seed(0)
    net = _net(p)
    _randomize(net)
    torch.save({p["key"]: net.state_dict()}, tmp_path / "w.pth")
    m = restoration.load(ref, weights=tmp_path / "w.pth", device="cpu", dtype="float32")
    h, w, s = 21, 37, p["scale"]
    img = synthetic_image(h, w, torch.device("cpu"))
    with torch.inference_mode():
        want = net(img)  # upstream: position-bias MLP, CPU masks per call, BatchNorm, matmul + softmax attention
        raw = m._runner(img)["image"]
        exported = m.module()(img)
    torch.testing.assert_close(raw, want, rtol=1e-4, atol=1e-4 * float(want.abs().max()))
    got = m(img).image
    assert got.shape == (1, 3, h * s, w * s)
    torch.testing.assert_close(got, raw.clamp(0, 1), rtol=0, atol=1e-6)
    torch.testing.assert_close(exported, got, rtol=0, atol=1e-6)


@pytest.mark.filterwarnings("ignore:torch.meshgrid")
@pytest.mark.parametrize("resi", ["1conv", "3conv"])
def test_last_group_scale(resi: str) -> None:
    """The float16 runner's scaled last residual group: the same features after the LayerNorm (its eps scaled)."""
    from fastforks.restoration.models.dat import model  # noqa: PLC0415

    torch.manual_seed(0)
    p = {"scale": 2, "img_size": 16, "split": [4, 8], "depths": [2, 2], "embed_dim": 32, "heads": [2, 2],
         "expansion": 2, "resi": resi, "upsampler": "pixelshuffle"}  # fmt: skip
    net = _net(p)
    x = torch.rand(1, 3, 16, 24)
    with torch.inference_mode():
        want = net.forward_features(net.conv_first(x))
        model._scale_last_group(net)
        got = net.forward_features(net.conv_first(x))
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
