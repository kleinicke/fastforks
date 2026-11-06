"""Segmentation labels-only call (`Runner.labels`, `seginfer.upsample_argmax`): the argmax of the logits (exact here;
the contract allows near-ties)."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as F

from fastforks import segmentation
from fastforks._core import kernels
from fastforks._core.errors import WeightsError
from fastforks.kernels import cudnn_hacc
from fastforks.shared import seginfer


def _full(s: torch.Tensor, sizes: list[tuple[int, int]]) -> torch.Tensor:
    y = F.interpolate(s, sizes[0], mode="bilinear", align_corners=False)
    for size in sizes[1:]:
        y = seginfer.resize(y, size)
    return y


def _assert_labels(got: torch.Tensor, scores: torch.Tensor, rtol: float = 1e-6) -> None:
    """`got` is the argmax of `scores` [B, C, H, W]; a label may differ only where the two scores are a near tie."""
    want = scores.argmax(1)
    bad = got != want
    if not bad.any():
        return
    a = scores.gather(1, got.unsqueeze(1)).squeeze(1)[bad]
    b = scores.gather(1, want.unsqueeze(1)).squeeze(1)[bad]
    assert bool(((a - b).abs() <= rtol * torch.maximum(a.abs(), b.abs()).clamp(min=1.0)).all()), "not a near tie"


@pytest.mark.parametrize("chunk", [1 << 24, 4 * 40 * 48, 40 * 48])  # batch 2: one chunk, 2-class, 1-class chunks
@pytest.mark.parametrize("layout", [torch.contiguous_format, torch.channels_last])
def test_upsample_argmax_exact(monkeypatch: pytest.MonkeyPatch, chunk: int, layout: torch.memory_format) -> None:
    monkeypatch.setattr(seginfer, "_ARGMAX_CHUNK", chunk)
    g = torch.Generator().manual_seed(0)
    s = torch.randn(2, 23, 10, 12, generator=g)
    s[:, 7] = s[:, 3]  # ties across chunks: the first index wins
    s[:, 12] = s[:, 11]  # ties inside a chunk
    s[:, :, 4:6] = torch.round(s[:, :, 4:6])  # flat ties
    s[0, 20, 2, 3] = float("nan")  # NaN wins, as argmax
    s = s.contiguous(memory_format=layout)
    for sizes in ([(20, 24), (40, 48)], [(40, 48), (40, 48)], [(37, 45)]):
        got = seginfer.upsample_argmax(s, sizes)
        assert got.dtype == torch.int64
        _assert_labels(got, _full(s, sizes))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
def test_fused_upsample_argmax(dtype: torch.dtype, layout: str) -> None:
    """`seg_up`: the labels are exactly the argmax of the fused logits, which are close to PyTorch's resize."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    g = torch.Generator("cuda").manual_seed(0)
    s = torch.randn(2, 23, 10, 12, device="cuda", generator=g)
    s[:, 7] = s[:, 3]  # ties: the first index wins
    s[:, :, 4:6] = torch.round(s[:, :, 4:6])
    s[0, 20, 2, 3] = float("nan")  # NaN wins, as argmax
    s = s.to(dtype)
    if layout == "nhwc":
        s = s.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
    for sizes in ([(20, 24), (40, 48)], [(40, 48), (40, 48)], [(37, 45)], [(10, 12), (7, 5)], [(43, 50), (17, 90)]):
        kernels.reset_fallbacks()
        up = seginfer.upsample(s, sizes, fused=True)
        got = seginfer.upsample_argmax(s, sizes, fused=True)
        assert not kernels.fallbacks(), sizes
        assert got.dtype == torch.int64
        assert torch.equal(got, up.argmax(1)), sizes
        ref = seginfer.upsample(s.float(), sizes)
        torch.testing.assert_close(up, ref, rtol=0, atol=1e-5, equal_nan=True)


# one ref per family with a `labels` override; the smallest variants
REFS = ["deeplabv3-r101-d8-ade20k@1", "segformer-b0-ade20k@1", "dinov2-s-linear-ade20k@1", "sapiens-seg-0.3b-goliath@1"]


@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["portable", "kernels"])
@pytest.mark.parametrize("ref", REFS)
def test_labels_match_logits(ref: str, mode: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    try:
        m = segmentation.load(ref, mode=mode, device="cuda")
    except WeightsError as e:
        pytest.skip(f"{ref} weights not available: {e}")
    g = torch.Generator("cuda").manual_seed(0)
    for h, w in ((480, 640), (300, 1100)):  # resized up / down, two class chunks at the larger output
        img = (torch.rand(3, h, w, device="cuda", generator=g) * 255).to(torch.uint8)
        full = m(img, logits=True)
        assert full.logits is not None
        assert torch.equal(m(img).labels, full.labels), (h, w)


@pytest.mark.model
@pytest.mark.gpu
def test_mmseg_kernels_graphs(monkeypatch: pytest.MonkeyPatch) -> None:
    """mmseg kernels replay a graph for a repeated small input (<= 512 x 512): == eager; larger inputs stay eager."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    monkeypatch.setattr(cudnn_hacc, "BUILD_AFTER", 0)  # plans on the first call, the graph on the second
    monkeypatch.setattr(cudnn_hacc, "CAPTURE_AFTER", 1)
    try:
        m = segmentation.load(REFS[0], mode="kernels", device="cuda")
    except WeightsError as e:
        pytest.skip(f"{REFS[0]} weights not available: {e}")
    r: Any = m._runner
    g = r._graphs
    gen = torch.Generator("cuda").manual_seed(1)
    x, y = (torch.rand(1, 3, 512, 512, device="cuda", generator=gen) for _ in range(2))
    pixels, g.pixels = g.pixels, 0
    eager = [r(x)["logits"], r(y)["logits"]]
    g.pixels = pixels
    r(x)  # first call: eager
    assert not g.cache.graphs
    first = r(x)["logits"]  # second: captures
    assert len(g.cache.graphs) == 1
    second = r(y)["logits"]  # replay
    torch.testing.assert_close(first, eager[0], rtol=0, atol=0)
    torch.testing.assert_close(second, eager[1], rtol=0, atol=0)
    assert torch.equal(r.labels(y), eager[1].argmax(1))
    big = torch.rand(1, 3, 512, 640, device="cuda", generator=gen)
    for _ in range(3):
        r(big)
    assert len(g.cache.graphs) == 1
    m.release_graphs()
    assert not g.cache.graphs
