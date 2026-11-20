"""SeedVR2: the card, the NF4 quantiser, the frame-wise causal convolution and its chunked memory (CPU)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import fastforks.restoration
from fastforks._core import registry
from fastforks.restoration.models.seedvr2.net import vae
from fastforks.restoration.models.seedvr2.net.vae import CausalConv3d, State
from fastforks.shared import host_blocks, nf4


def test_card_variants() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("restoration").values() if c.family == "seedvr2"}
    assert sorted(cards) == ["seedvr2-3b", "seedvr2-7b"]
    for vid, (card, variant) in cards.items():
        p = {**card.params, **variant.params}
        assert (p["scale"], p["variant"]) == (2, vid.removeprefix("seedvr2-"))
        assert 16 % p["scale"] == 0
        assert card.io.pad_multiple * p["scale"] % 16 == 0  # the padded, scaled frame is a multiple of 16


def test_nf4_quantize_picks_the_nearest_code() -> None:
    torch.manual_seed(0)
    w = torch.randn(48, 256).bfloat16()
    q, absmax = nf4.quantize(w, rows=20)  # several row chunks
    assert q.dtype == torch.uint8
    assert q.shape == (48 * 256 // 2,)
    assert absmax.shape == (48, 4)
    lut = nf4.pair_lut(nf4.code(torch.device("cpu")))
    got = nf4.dequant_lut(q, absmax, lut, 48).float()
    blocks = w.float().view(48, 4, 64)
    torch.testing.assert_close(absmax.float(), blocks.abs().amax(-1).bfloat16().float())
    cand = (nf4.code(torch.device("cpu")) * absmax.float()[..., None, None]).bfloat16().float()  # [48, 4, 1, 16]
    best = (blocks[..., None] - cand).abs().amin(-1).view(48, 256)
    assert torch.allclose((got - w.float()).abs(), best, atol=1e-2 * float(absmax.float().max()))
    exact = (nf4.code(torch.device("cpu"))[torch.randint(16, (8, 64))] * 2.0).bfloat16()  # code values x absmax 2
    exact[:, 0] = 2.0  # each block's absmax
    q, absmax = nf4.quantize(exact)
    assert torch.equal(nf4.dequant_lut(q, absmax, lut, 8), exact)


def test_nf4_quantize_needs_blocks_of_64() -> None:
    with pytest.raises(ValueError, match="multiple of 64"):
        nf4.quantize(torch.zeros(4, 96))


def _conv(kernel: int, stride: int, padding: int) -> CausalConv3d:
    torch.manual_seed(1)
    c = CausalConv3d(3, 5, kernel, stride, padding).to_empty(device="cpu")
    torch.nn.init.normal_(c.weight, std=0.2)
    torch.nn.init.normal_(c.bias)
    return c


@pytest.mark.parametrize(("kernel", "stride", "padding"), [(3, 1, 1), (3, 2, 0), (1, 1, 0)])
def test_causal_conv_framewise_and_chunked(kernel: int, stride: int, padding: int) -> None:
    """Frame-by-frame = one Conv3d over the replicate-padded clip; chunks with memory = the whole clip."""
    c = _conv(kernel, stride, padding)
    x = torch.randn(1, 3, 9, 6, 7)
    head = x[:, :, :1].expand(-1, -1, 2 * padding, -1, -1)
    want = F.conv3d(torch.cat([head, x], 2), c.weight, c.bias, (stride,) * 3, (0, padding, padding))
    with torch.no_grad():
        whole = c(x, State.DISABLED)
        first = c(x[:, :, :5], State.INITIALIZING)
        rest = c(x[:, :, 5:], State.ACTIVE)
    torch.testing.assert_close(whole, want)
    if stride == 1:
        torch.testing.assert_close(torch.cat([first, rest], 2), want)


def test_stream_and_resident_ram_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Streaming pins host RAM: a clearly too small free RAM is an error at load, not an OOM kill (CPU)."""
    monkeypatch.setattr(host_blocks, "available_host_bytes", lambda: 2**30)
    with pytest.raises(MemoryError, match="stream=False"):
        host_blocks.require_host_ram(8 * 2**30, "seedvr2-3b")
    monkeypatch.setattr(host_blocks, "available_host_bytes", lambda: 40 * 2**30)
    host_blocks.require_host_ram(8 * 2**30, "seedvr2-3b")
    monkeypatch.setattr(host_blocks, "available_host_bytes", lambda: None)  # unreadable (not Linux): no check
    host_blocks.require_host_ram(8 * 2**30, "seedvr2-3b")


@pytest.mark.gpu
@pytest.mark.slow
def test_stream_false_is_bit_identical() -> None:
    """The resident path (`stream=False`) gives the streamed default's output bit for bit (3 B, small frame)."""
    x = torch.rand(3, 64, 64, generator=torch.Generator().manual_seed(0)).mul(255).byte().permute(1, 2, 0).numpy()
    out = []
    for s in (True, False):
        model = fastforks.restoration.load("seedvr2-3b", mode="portable", device="cuda", stream=s)
        out.append(torch.as_tensor(model(x).image).clone())
        del model
        torch.cuda.empty_cache()
    assert torch.equal(*out)


@pytest.mark.gpu
def test_nf4_portable_tiles_preserve_scale_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nf4, "_PACKED_TILE", 256)
    q = torch.arange(13 * 64, device="cuda").to(torch.uint8).view(13, 64).t()
    scales = torch.linspace(0, 9, q.numel() // nf4.BLOCK * 2, device="cuda")
    lut = nf4.pair_lut(nf4.code(q.device))
    decoded = torch.nn.functional.embedding(q.int(), lut).reshape(-1, nf4.BLOCK)
    expected = (decoded * scales[:, None]).bfloat16().view(13, -1)
    torch.testing.assert_close(nf4.dequant_lut(q, scales, lut, 13), expected, atol=0, rtol=0)


@pytest.mark.gpu
def test_group_norm_split_preserves_affine_and_partial_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(2)
    x = torch.randn(2, 28, 3, 19, 17, device="cuda", dtype=torch.bfloat16)
    norm = torch.nn.GroupNorm(7, 28, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        norm.weight.copy_(torch.randn_like(norm.weight))
        norm.bias.copy_(torch.randn_like(norm.bias))
        expected = vae.group_norm(norm, x, upcast=True, silu=True)
        monkeypatch.setattr(vae, "_GROUP_NORM_MIN_BYTES", 1)
        monkeypatch.setattr(vae, "_GROUP_NORM_CHUNK_BYTES", 23_000)
        actual = vae.group_norm(norm, x, upcast=True, silu=True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
