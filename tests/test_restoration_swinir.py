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


@pytest.mark.gpu
@pytest.mark.parametrize("channels", [3, 6, 16])
def test_shifted_window_roundtrip(channels: int) -> None:
    from fastforks.kernels.shift_windows import rearrange  # noqa: PLC0415

    h, w, win = 12, 20, 4
    tokens = torch.randn(2, h * w, channels * 2, dtype=torch.float16, device="cuda")[:, :, 1::2]
    shifted = torch.roll(tokens.reshape(2, h, w, channels), (-2, -2), (1, 2))
    windows = shifted.reshape(2, h // win, win, w // win, win, channels).transpose(2, 3).reshape(-1, win**2, channels)
    got = rearrange(tokens, (h, w), win, 2)
    assert torch.equal(got, windows)
    assert torch.equal(rearrange(got, (h, w), win, 2, reverse=True), tokens)


@pytest.mark.gpu
@pytest.mark.parametrize(("channels", "shift"), [(144, 0), (180, 8), (182, 8)])
def test_window_residual_preserves_half_rounding(channels: int, shift: int) -> None:
    from fastforks.kernels.shift_windows import rearrange, reverse_residual  # noqa: PLC0415

    h, w, window = 32, 48, 16
    skip = torch.randn(2, h * w, channels, device="cuda", dtype=torch.float16)
    conv = torch.randn(2, h * w, channels + 4, device="cuda", dtype=torch.float16)[..., :channels]
    tokens = torch.randn(2 * h * w // window**2, window**2, channels, device="cuda", dtype=torch.float16)
    expected = skip + rearrange(tokens, (h, w), window, shift, reverse=True) + conv * 0.01
    actual = reverse_residual(tokens, skip, conv, (h, w), window=window, shift=shift, scale=0.01)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("scale", [2, 3, 4])
def test_pixelshuffle_stripes_keep_convolution_context(scale: int, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastforks.restoration.models.swinir.net.network_swinir import Upsample  # noqa: PLC0415
    from fastforks.shared import swin_sr  # noqa: PLC0415

    channels, height, width = 4, 11, 7
    x = torch.randn(2, channels, height, width)
    net = torch.nn.Module()
    net.upsample = Upsample(scale, channels)
    net.conv_last = torch.nn.Conv2d(channels, 3, 3, padding=1)
    monkeypatch.setattr(swin_sr, "_PIXELSHUFFLE_BYTES", x.element_size() * 2 * channels * width * scale**2 * 2)
    with torch.inference_mode():
        expected = net.conv_last(net.upsample(x))
        swin_sr.stripe_pixelshuffle(net, scale)
        actual = net.conv_last(net.upsample(x))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.gpu
@pytest.mark.parametrize(("height", "width", "window", "shift", "channels"), [(16, 20, 4, 0, 8), (12, 16, 4, 2, 12)])
def test_window_groups_write_the_same_raster(height: int, width: int, window: int, shift: int, channels: int) -> None:
    from fastforks._core.kernels import linear  # noqa: PLC0415 - CUDA kernel only
    from fastforks.kernels.shift_windows import load, rearrange  # noqa: PLC0415 - CUDA kernel only

    g = torch.Generator("cuda").manual_seed(71)
    grid = (height, width)
    gen = {"device": "cuda", "dtype": torch.float16, "generator": g}
    x = torch.randn(2, height * width, channels, **gen)
    out = torch.empty_like(x)
    rows, cols = height // window, width // window
    tiled = rearrange(x, grid, window, shift).view(2, rows, cols, window**2, channels)
    updates = torch.randn(tiled.shape, **gen)
    library = load(x.device)
    for b in range(x.shape[0]):
        # The regions cover the raster once, with partial final groups.
        for y0, group_rows in ((0, rows - 1), (rows - 1, 1)):
            for x0, group_cols in ((0, cols - 1), (cols - 1, 1)):
                ys, xs = slice(y0, y0 + group_rows), slice(x0, x0 + group_cols)
                region = tiled[b, ys, xs].reshape(-1, window**2, channels)
                delta = updates[b, ys, xs].reshape_as(region).contiguous()
                for first in range(0, len(region), 3):
                    pack = torch.empty(region[first : first + 3].shape, device=x.device, dtype=x.dtype)
                    threads = pack.numel() // 2
                    geometry = height, width, channels // 2, window, shift, y0, x0, group_cols, first, threads
                    library.launch("stream_window2", *linear(threads), x[b], None, pack, *geometry)
                    torch.testing.assert_close(pack, region[first : first + 3], atol=0, rtol=0)
                    library.launch(
                        "stream_window2", *linear(threads), delta[first : first + 3], x[b], out[b], *geometry
                    )
    expected = x + rearrange(updates.flatten(0, 2), grid, window, shift, reverse=True)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.parametrize(("growths", "shift"), [(1, 0), (4, 2)])
def test_dense_window_groups_preserve_packing(growths: int, shift: int) -> None:
    from fastforks._core.kernels import linear  # noqa: PLC0415 - CUDA kernel only
    from fastforks.kernels.shift_windows import load, rearrange  # noqa: PLC0415 - CUDA kernel only

    g = torch.Generator("cuda").manual_seed(83)
    height, width, window = 12, 16, 4
    channels = 8
    size = height, width
    growth_channels = 6
    feats = [
        torch.randn(2, height * width, c, device="cuda", dtype=torch.float16, generator=g)
        for c in (channels, *([growth_channels] * growths))
    ]
    combined = torch.cat(feats, -1)
    output = torch.empty_like(combined)
    rows, cols = height // window, width // window
    total = combined.shape[-1]
    expected = rearrange(combined, size, window, shift).view(2, rows, cols, window**2, total)
    k = load(combined.device)
    for b in range(combined.shape[0]):
        inputs = [part[b] for part in feats] + [None] * (4 - growths)
        for y0, nr in ((0, rows - 1), (rows - 1, 1)):
            for x0, nc in ((0, cols - 1), (cols - 1, 1)):
                group = expected[b, y0 : y0 + nr, x0 : x0 + nc].reshape(-1, window**2, total)
                for first in range(0, len(group), 3):
                    reference = group[first : first + 3]
                    packed = torch.empty(reference.shape, dtype=reference.dtype, device=reference.device)
                    threads = packed.numel() // 2
                    args = height, width, total // 2, window, shift, y0, x0, nc, first, threads
                    k.launch(
                        "pack_dense_windows2",
                        *linear(threads),
                        *inputs,
                        packed,
                        channels // 2,
                        growth_channels // 2,
                        *args,
                    )
                    torch.testing.assert_close(packed, reference, atol=0, rtol=0)
                    k.launch("scatter_windows2", *linear(threads), packed, output[b], *args)
    torch.testing.assert_close(combined, output, atol=0, rtol=0)
