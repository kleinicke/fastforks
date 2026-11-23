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


@pytest.mark.gpu
def test_kernels_windows_match_the_volume_lookup() -> None:
    """Kernels' windows (raft_flow) are upstream's bilinear lookup of the stored volume, also off the map's edges."""
    from types import SimpleNamespace  # noqa: PLC0415

    from fastforks.flow.models.flowformer import model  # noqa: PLC0415
    from fastforks.flow.models.flowformer.net.FlowFormer.LatentCostFormer.decoder import MemoryDecoder  # noqa: PLC0415
    from fastforks.kernels import raft_flow  # noqa: PLC0415

    torch.manual_seed(0)
    b, c, h, w = 2, 256, 7, 11
    f1, f2 = (0.1 * torch.randn(b, c, h, w, dtype=torch.float64)).half(), (0.1 * torch.randn(b, c, h, w)).half()
    cost_maps = torch.einsum("bci,bcj->bij", f1.double().flatten(2), f2.double().flatten(2)).reshape(b * h * w, 1, h, w)
    grid = torch.stack(torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy"), 0).to(torch.float64)
    coords = grid + torch.randn(b, 2, h, w, dtype=torch.float64) * 4  # many windows reach past the borders
    want = MemoryDecoder.encode_flow_token(None, cost_maps, coords)  # ty: ignore[invalid-argument-type]  # no self use
    decoder = SimpleNamespace(_raft_flow=raft_flow.load(torch.device("cuda")))
    vol = model._Volume(f1.cuda().float(), f2.cuda().float())
    got = model._kernel_windows(decoder, vol, coords.cuda().float())  # ty: ignore[invalid-argument-type]
    torch.testing.assert_close(got.cpu(), want.float(), rtol=0, atol=2e-3)


def test_kernels_chunked_encoder_matches_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kernels' encoder (cost maps per chunk of query pixels, never the volume) gives upstream's cost memory."""
    pytest.importorskip("timm")
    from fastforks._core.patch import override  # noqa: PLC0415
    from fastforks.flow.models.flowformer import model  # noqa: PLC0415
    from fastforks.flow.models.flowformer.net.FlowFormer.LatentCostFormer.transformer import FlowFormer  # noqa: PLC0415

    card = flow.model_card("flowformer-small-things")
    torch.manual_seed(0)
    net = FlowFormer(Params(**card.params, iters=2)).eval()
    a, b = torch.rand(1, 3, 64, 96) * 255, torch.rand(1, 3, 64, 96) * 255
    enc = net.memory_encoder
    with torch.inference_mode():
        want = enc(a, b, data := {})
        override(enc, model._corr, "corr")
        override(enc.cost_perceiver_encoder, model._encode)
        monkeypatch.setattr(enc.cost_perceiver_encoder.patch_embed, "chunk", 7)  # several chunks, a ragged last one
        got = enc(a, b, mine := {})
    assert mine["H3W3"] == data["H3W3"]
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
