"""Restormer: the card's variants, the folded norms and attention against the vendored upstream, `load(path)`."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import restoration
from fastforks._core import registry
from fastforks._core.bench import synthetic_image
from fastforks._core.errors import WeightsError
from fastforks.restoration._api import _checkpoint_family
from fastforks.restoration.models.restormer.model import _fold_head_attention, _gate_ffn, _pad_ffn
from fastforks.restoration.models.restormer.net import Restormer
from fastforks.restoration.models.restormer.net.restormer_arch import Attention, FeedForward

if TYPE_CHECKING:
    from pathlib import Path

KW = {"dim": 8, "num_blocks": [1, 1, 1, 2], "num_refinement_blocks": 1, "heads": [1, 2, 2, 4]}


def test_card_variants() -> None:
    found = registry.cards("restoration").values()
    cards = {v.id: {**c.params, **v.params} for c, v in found if c.family == "restormer"}
    assert sorted(cards) == ["restormer-deblur", "restormer-derain", "restormer-sidd"]
    assert [cards[k]["norm"] for k in sorted(cards)] == ["WithBias", "WithBias", "BiasFree"]


def _trained_like(net: torch.nn.Module) -> torch.nn.Module:
    """Random norm affines and temperatures (init: ones / zeros, which would hide folding errors)."""
    torch.manual_seed(1)
    for name, p in net.named_parameters():
        if ".norm" in name or name.endswith("temperature"):
            p.data.add_(torch.randn_like(p) * 0.5)
    return net.eval()


@pytest.mark.parametrize(("norm", "ref"), [("WithBias", "restormer-deblur"), ("BiasFree", "restormer-sidd")])
def test_matches_vendored_upstream(tmp_path: Path, norm: str, ref: str) -> None:
    """`load(path)` finds the family and the variant; the folded blocks and the padding match upstream's test call."""
    torch.manual_seed(0)
    net = _trained_like(Restormer(**KW, LayerNorm_type=norm))
    torch.save({"params": net.state_dict()}, tmp_path / "w.pth")
    assert _checkpoint_family(tmp_path / "w.pth") == "restormer"
    m = restoration.load(tmp_path / "w.pth", mode="portable", device="cpu", dtype="float32")
    assert m.variant.id == ref
    x = synthetic_image(41, 57, torch.device("cpu"))
    runner = m._runner
    with torch.inference_mode():
        want = net(torch.nn.functional.pad(x, (0, 7, 0, 7), "reflect"))[..., :41, :57]
        xp, crop = runner.pad(x)
        got = runner(xp)["image"][..., crop.top : crop.top + crop.height, crop.left : crop.left + crop.width]
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_rejects_dual_pixel(tmp_path: Path) -> None:
    torch.save(Restormer(**KW, inp_channels=6, dual_pixel_task=True).state_dict(), tmp_path / "w.pth")
    with pytest.raises(WeightsError, match="dual-pixel"):
        restoration.load(tmp_path / "w.pth", mode="portable", device="cpu", dtype="float32")


@pytest.mark.parametrize("heads", [1, 2, 4])
def test_headwise_attention(heads: int) -> None:
    """The per-head products match upstream, including interleaved batch strides and unequal head temperatures."""

    torch.manual_seed(8)
    module = _trained_like(Attention(24, heads, bias=False))
    original = deepcopy(module)
    _fold_head_attention(module, "cpu")
    x = torch.randn(3, 24, 11, 7).contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        torch.testing.assert_close(module(x), original(x), atol=1e-6, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("bias", [False, True])
def test_fused_depthwise_gate(*, bias: bool) -> None:
    """Uneven widths exercise zero padding, the right/bottom image border and the batched CUDA path."""

    torch.manual_seed(1)
    layer = FeedForward(24, 2.66, bias=bias)
    original = deepcopy(layer).cuda().half().to(memory_format=torch.channels_last)
    _pad_ffn(layer)
    _gate_ffn(layer, "cuda")
    layer.cuda().half().to(memory_format=torch.channels_last)
    x = torch.randn(2, 24, 9, 13, device="cuda", dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        torch.testing.assert_close(layer(x), original(x), rtol=0.002, atol=0.0003)
