"""Kernels CUDA graphs of the launch-bound CNN stereo families (`SmallShapeGraphs`): replay == eager, opt-out."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from fastforks import stereo
from fastforks._core.errors import WeightsError
from fastforks.kernels import cudnn_hacc

# one ref per family / code path: LAS2 feed-forward + iterative, CoEx, Fast-ACVNet + plus, HITNet, IGEV++, GREAT-IGEV
REFS = [
    "las2-m@1",
    "las2-h@1",
    "coex-sceneflow@1",
    "fast-acvnet-sceneflow@1",
    "fast-acvnet-plus-sceneflow@1",
    "hitnet-eth3d@1",
    "igev-plusplus-sceneflow@1",
    "great-igev-sceneflow@1",
]

pytestmark = [pytest.mark.model, pytest.mark.gpu]


@pytest.fixture(autouse=True)
def _plans_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """cuDNN plans on a geometry's first call, graphs on a shape's second (the default is its 4th)."""
    monkeypatch.setattr(cudnn_hacc, "BUILD_AFTER", 0)
    monkeypatch.setattr(cudnn_hacc, "CAPTURE_AFTER", 1)


def _load(ref: str, **options: bool) -> stereo.StereoModel:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    try:
        return stereo.load(ref, mode="kernels", device="cuda", options=options or None)
    except WeightsError as e:
        pytest.skip(f"{ref} weights not available: {e}")


def _pair(r: object, seed: int, h: int = 128, w: int = 192) -> tuple[torch.Tensor, torch.Tensor]:
    """A random pair as runner `r` takes it: 0..255, or [0, 1] for `io = "raw"` runners."""
    g = np.random.default_rng(seed)
    scale = 1 / 255 if getattr(r, "io", "core") == "raw" else 1.0
    a, b = (torch.from_numpy(g.integers(0, 256, (1, 3, h, w), dtype=np.uint8)).cuda().float() * scale for _ in range(2))
    return a, b


def _eager(r: Any, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    g = r._graphs
    pixels, g.pixels = g.pixels, 0
    try:
        return r(a, b)["disparity"]
    finally:
        g.pixels = pixels


@pytest.mark.parametrize("ref", REFS)
def test_replay_matches_eager(ref: str) -> None:
    model = _load(ref, cuda_graphs=True)  # GREAT-IGEV: off by default
    r = model._runner
    g = r._graphs
    assert g.cache is not None
    a, b = _pair(r, 0)
    eager = _eager(r, a, b)
    r(a, b)  # first call: eager
    assert not g.cache.graphs
    first = r(a, b)["disparity"]  # second call: captures
    assert len(g.cache.graphs) == 1
    c, d = _pair(r, 1)
    second = r(c, d)["disparity"]  # replay with new data; `first` must not change
    torch.testing.assert_close(first, eager, rtol=0, atol=1e-3)
    torch.testing.assert_close(second, _eager(r, c, d), rtol=0, atol=1e-3)
    assert first.data_ptr() != second.data_ptr()
    model.release_graphs()
    assert not g.cache.graphs


@pytest.mark.parametrize(
    "ref", ["fast-acvnet-sceneflow@1", "igev-plusplus-sceneflow@1", "rt-monster-plusplus-mix-all@1"]
)
def test_eth3d_size_stream(ref: str) -> None:
    """ETH3D's two sizes alternating (delivery_area_1s, 1l, ...): the larger shape's capture must not break the
    smaller shape's graph (its cuDNN workspace was freed: illegal address in the accuracy campaign)."""
    model = _load(ref)
    g = torch.Generator().manual_seed(0)
    pairs = [
        torch.randint(0, 256, (2, 3, h, w), generator=g, dtype=torch.uint8).cuda() for h, w in ((435, 711), (489, 942))
    ]
    outs = [model(*pairs[i % 2]).disparity for i in range(5)]  # eager, eager, capture, capture, replay
    torch.testing.assert_close(outs[4], outs[0], rtol=0, atol=1e-3)
    torch.testing.assert_close(outs[3], outs[1], rtol=0, atol=1e-3)


@pytest.mark.parametrize("ref", REFS[:1])
def test_big_shapes_stay_eager(ref: str) -> None:
    r = _load(ref)._runner
    g = r._graphs
    g.pixels = 128 * 192 - 1
    a, b = _pair(r, 2)
    for _ in range(3):
        r(a, b)
    assert not g.cache.graphs


@pytest.mark.parametrize("ref", REFS[:1])
def test_opt_out(ref: str) -> None:
    m = _load(ref, cuda_graphs=False)
    r = m._runner
    assert r._graphs.cache is None
    a, b = _pair(r, 3)
    outs = [r(a, b)["disparity"] for _ in range(3)]
    torch.testing.assert_close(outs[2], outs[0], rtol=0, atol=0)
    m.release_graphs()  # no-op
