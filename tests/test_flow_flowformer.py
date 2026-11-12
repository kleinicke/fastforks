"""FlowFormer Small: local tensor rearrangements, batch isolation and flow module parity."""

from __future__ import annotations

import pytest
import torch

from fastforks import flow
from fastforks.flow.models.flowformer.model import Params
from fastforks.flow.models.flowformer.net.ffops import rearrange


def test_rearrange_matches_explicit_axes() -> None:
    """The torch-only replacement keeps grouped axes and inserted singleton dimensions."""
    tensor = torch.arange(2 * 12 * 3 * 5).reshape(2, 12, 3, 5)
    expected = tensor.reshape(2, 3, 4, 3, 5).permute(0, 1, 3, 4, 2).reshape(2, 3, 15, 4)
    torch.testing.assert_close(rearrange(tensor, "b (heads d) h w -> b heads (h w) d", heads=3), expected)
    tensor = torch.arange(24).reshape(6, 4)
    torch.testing.assert_close(rearrange(tensor, "(x u) d -> x u () d", x=2), tensor.reshape(2, 3, 1, 4))


def test_released_variants() -> None:
    """The three checkpoints use the Small CNN configuration."""
    for stage in ("chairs", "things", "sintel"):
        card = flow.model_card(f"flowformer-small-{stage}")
        assert card.family == "flowformer"
        assert card.params["cnet"] == card.params["fnet"] == "basicencoder"
        assert card.options["iters"] == 32


@pytest.mark.slow
def test_batched_network_is_independent() -> None:
    """Each batch member is independent through latent cost memory and recurrent decoding."""
    pytest.importorskip("timm")
    from fastforks.flow.models.flowformer.net.FlowFormer.LatentCostFormer.transformer import FlowFormer  # noqa: PLC0415

    card = flow.model_card("flowformer-small-things")
    params = Params(**card.params, iters=2)
    torch.manual_seed(0)
    net = FlowFormer(params).eval()
    a, b = torch.rand(2, 3, 64, 96) * 255, torch.rand(2, 3, 64, 96) * 255
    with torch.inference_mode():
        batch = net(a, b)[0]
        single = net(a[1:], b[1:])[0]
    assert batch.shape == (2, 2, 64, 96)
    assert bool(torch.isfinite(batch).all())
    torch.testing.assert_close(batch[1:], single, rtol=0, atol=1e-3)
