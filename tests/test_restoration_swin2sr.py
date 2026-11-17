"""Swin2SR: the card's variants, the folded cosine attention against the vendored upstream network, `load(path)`."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import restoration
from fastforks._core import registry
from fastforks._core.bench import synthetic_image
from fastforks.restoration._api import _checkpoint_family
from fastforks.restoration.models.swin2sr.net import Swin2SR

if TYPE_CHECKING:
    from pathlib import Path

VARIANTS = {"swin2sr-x4": 4, "swin2sr-x2": 2, "swin2sr-x4-real": 4}


def _cards() -> dict:
    return {v.id: (c, v) for c, v in registry.cards("restoration").values() if c.family == "swin2sr"}


def test_card_variants() -> None:
    cards = _cards()
    assert sorted(cards) == sorted(VARIANTS)
    for vid, scale in VARIANTS.items():
        card, variant = cards[vid]
        assert {**card.params, **variant.params}["scale"] == scale, vid


def _small(upsampler: str) -> torch.nn.Module:
    scale = 2 if upsampler == "pixelshuffle" else 4  # nearest+conv: x4 only
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "torch.meshgrid", UserWarning)
        net = Swin2SR(upscale=scale, img_size=16, window_size=8, img_range=1.0, depths=[2, 2], embed_dim=24,
                      num_heads=[2, 2], mlp_ratio=2, upsampler=upsampler, resi_connection="1conv").eval()  # fmt: skip
    for name, p in net.named_parameters():  # trained-like q / v biases and logit scales (init: zeros, log 10)
        if name.endswith(("q_bias", "v_bias", "logit_scale")):
            p.data.add_(torch.randn_like(p) * 0.5)
    return net


@pytest.mark.parametrize("upsampler", ["pixelshuffle", "nearest+conv"])
def test_load_file_matches_vendored_upstream(tmp_path: Path, upsampler: str) -> None:
    """`load(path)` finds the family; padding, the precomputed bias and the SDPA cosine attention match upstream."""
    net = _small(upsampler)
    torch.save(net.state_dict(), tmp_path / "w.pth")
    assert _checkpoint_family(tmp_path / "w.pth") == "swin2sr"
    m = restoration.load(tmp_path / "w.pth", mode="portable", device="cpu", dtype="float32")
    assert m.variant.id == ("swin2sr-x2" if upsampler == "pixelshuffle" else "swin2sr-x4-real")
    h, w = 13, 21
    img = synthetic_image(h, w, torch.device("cpu"))
    x = torch.cat([img, img.flip(2)], 2)[:, :, :16]  # main_test_swin2sr.py: to (size // 8 + 1) * 8
    x = torch.cat([x, x.flip(3)], 3)[:, :, :, :24]
    padded, region = m._runner.pad(img)
    assert torch.equal(padded, x)
    assert (region.top, region.left, region.height, region.width) == (0, 0, h, w)
    with torch.inference_mode():
        want = net(x)  # upstream: cpb MLP + gather per call, CPU masks per size, matmul + softmax attention
        raw = m._runner(x)["image"]
        sixteen = m._runner(x[:, :, :16, :16])["image"]  # the training size: the stored masks
        want16 = net(x[:, :, :16, :16])
    torch.testing.assert_close(raw, want, rtol=1e-4, atol=1e-4 * float(want.abs().max()))
    torch.testing.assert_close(sixteen, want16, rtol=1e-4, atol=1e-4 * float(want16.abs().max()))
