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


@pytest.mark.gpu
def test_attention_softmax_preserves_cuda_alignment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The half-precision decoder must not pass an unaligned odd row to CUDA softmax."""
    from fastforks.flow.models.flowformer.net.FlowFormer.LatentCostFormer import gma  # noqa: PLC0415

    attention = gma.Attention(args=None, dim=16, max_pos_size=7, heads=1, dim_head=16).eval().cuda().half()
    features = torch.randn(1, 16, 7, 147, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        expected = attention(features)
        row_bytes = 7 * 147 * features.element_size()
        monkeypatch.setattr(gma, "_SOFTMAX_MIN_BYTES", 1)
        monkeypatch.setattr(gma, "_SOFTMAX_GROUP_BYTES", 18 * row_bytes)
        softmax = torch.Tensor.softmax
        alignments = []

        def checked_softmax(scores: torch.Tensor, dim: int) -> torch.Tensor:
            alignments.append(scores.data_ptr() % 16)
            return softmax(scores, dim)

        monkeypatch.setattr(torch.Tensor, "softmax", checked_softmax)
        actual = attention(features)
    assert len(alignments) > 1
    assert not any(alignments)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.slow
def test_batched_network_is_independent() -> None:
    """Each batch member is independent through latent cost memory and recurrent decoding."""
    pytest.importorskip("timm")
    from fastforks.flow.models.flowformer.net.FlowFormer.LatentCostFormer.transformer import FlowFormer  # noqa: PLC0415

    card = flow.model_card("flowformer-small-things")
    params = Params(**card.params, iters=2)
    torch.manual_seed(0)
    net = FlowFormer(params).eval()
    net.memory_decoder.update_block.gru.fuse()  # the runner fuses z and r after loading
    a, b = torch.rand(2, 3, 64, 96) * 255, torch.rand(2, 3, 64, 96) * 255
    with torch.inference_mode():
        batch = net(a, b)[0]
        single = net(a[1:], b[1:])[0]
    assert batch.shape == (2, 2, 64, 96)
    assert bool(torch.isfinite(batch).all())
    torch.testing.assert_close(batch[1:], single, rtol=0, atol=1e-3)
