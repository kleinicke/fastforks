"""Kernels CUDA graphs of the classification families (`libnet.Kernels`): replay == eager, shapes, release."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from fastforks import classification
from fastforks._core.errors import WeightsError

# fx-traced CNNs (including StarNet), ViT (fused attention), Swin (window masks), TinyViT (its own kernels runner)
REFS = ["mobilenet_v2.tv2_in1k@1", "vit_base_patch16_224.augreg2_in21k_ft_in1k@1",
        "swin_base_patch4_window7_224.ms_in22k_ft_in1k@1", "tiny_vit_5m_224.dist_in22k_ft_in1k@1",
        "starnet_s1.in1k@1"]  # fmt: skip

pytestmark = [pytest.mark.model, pytest.mark.gpu]


def _load(ref: str) -> classification.ClassificationModel:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    try:
        return classification.load(ref, mode="kernels", device="cuda")
    except WeightsError as e:
        pytest.skip(f"{ref} weights not available: {e}")


def _eager(r: Any, x: torch.Tensor) -> torch.Tensor:
    g = r._graphs
    pixels, g.pixels = g.pixels, 0
    try:
        return r(x)["logits"]
    finally:
        g.pixels = pixels


@pytest.mark.parametrize("ref", REFS)
def test_replay_matches_eager(ref: str) -> None:
    model = _load(ref)
    r = model._runner
    g = r._graphs
    assert g.cache is not None
    gen = torch.Generator("cuda").manual_seed(0)
    x, y, z = (torch.randn(b, 3, 224, 224, device="cuda", generator=gen) for b in (1, 1, 2))
    eager = _eager(r, x)
    r(x)  # first call: eager
    assert not g.cache.graphs
    first = r(x)["logits"]  # second call: captures
    assert len(g.cache.graphs) == 1
    second = r(y)["logits"]  # replay with new data; `first` must not change
    torch.testing.assert_close(first, eager, rtol=0, atol=0)
    torch.testing.assert_close(second, _eager(r, y), rtol=0, atol=0)
    assert first.data_ptr() != second.data_ptr()
    for _ in range(2):  # another batch size: its own graph
        two = r(z)["logits"]
    assert len(g.cache.graphs) == 2
    torch.testing.assert_close(two, _eager(r, z), rtol=0, atol=0)
    # past the runner's pixel limit (the big-input path, cudnn_hacc in `libnet.Kernels`) the call stays eager
    big = torch.randn(max(8, g.pixels // 224**2 + 1), 3, 224, 224, device="cuda", generator=gen)
    for _ in range(3):
        r(big)
    assert len(g.cache.graphs) == 2
    model.release_graphs()
    assert not g.cache.graphs
    torch.testing.assert_close(r(x)["logits"], eager, rtol=0, atol=0)
