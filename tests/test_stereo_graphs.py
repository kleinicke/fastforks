"""RT-MonSter++ kernels CUDA graphs: replay == eager, no aliasing, eviction, capture failure, release, opt-out."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastforks import stereo
from fastforks._core.errors import WeightsError
from fastforks.kernels import cudnn_hacc
from fastforks.stereo.models.rt_monster_plusplus import model as rt

REF = "rt-monster-plusplus-mix-all@1"

pytestmark = [pytest.mark.model, pytest.mark.gpu]


@pytest.fixture(autouse=True)
def _plans_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """cuDNN plans on a geometry's first call, graphs on a shape's second (the default is its 4th)."""
    monkeypatch.setattr(cudnn_hacc, "BUILD_AFTER", 0)
    monkeypatch.setattr(cudnn_hacc, "CAPTURE_AFTER", 1)


def _load(**options: bool) -> stereo.StereoModel:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    try:
        return stereo.load(REF, mode="kernels", device="cuda", options=options or None)
    except WeightsError as e:
        pytest.skip(f"{REF} weights not available: {e}")


@pytest.fixture
def model() -> stereo.StereoModel:
    return _load()


def _runner(model: stereo.StereoModel) -> rt.Kernels:
    assert isinstance(model._runner, rt.Kernels)
    return model._runner


def _pair(seed: int, h: int = 128, w: int = 192) -> tuple[torch.Tensor, torch.Tensor]:
    g = np.random.default_rng(seed)
    a, b = (torch.from_numpy(g.integers(0, 256, (1, 3, h, w), dtype=np.uint8)).cuda().float() for _ in range(2))
    return a, b


def _eager(r: rt.Kernels, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return r._forward(r.net, a, b)


def test_replay_matches_eager_and_is_not_aliased(model: stereo.StereoModel) -> None:
    r = _runner(model)
    a, b = _pair(0)
    eager = _eager(r, a, b)
    assert r._cache is not None
    r(a, b)  # first call: eager
    assert not r._cache.graphs
    first = r(a, b)["disparity"]  # second call: captures
    assert len(r._cache.graphs) == 1
    c, d = _pair(1)
    second = r(c, d)["disparity"]  # replay with new data; `first` must not change
    torch.testing.assert_close(first, eager, rtol=0, atol=1e-3)
    torch.testing.assert_close(second, _eager(r, c, d), rtol=0, atol=1e-3)
    assert first.data_ptr() != second.data_ptr()
    assert not torch.equal(first, second)


def test_third_and_alternating_shapes(model: stereo.StereoModel, monkeypatch: pytest.MonkeyPatch) -> None:
    r = _runner(model)
    assert r._cache is not None
    shapes = [(96, 160), (128, 192), (64, 128)]
    for h, w in shapes:
        a, b = _pair(3, h, w)
        r(a, b)
        r(a, b)
    assert len(r._cache.graphs) == 2  # the third shape stays eager while both graphs are fresh
    a, b = _pair(4, *shapes[2])
    torch.testing.assert_close(r(a, b)["disparity"], _eager(r, a, b), rtol=0, atol=1e-3)
    monkeypatch.setattr(r._cache, "_stale", 0)  # stale graphs may now be replaced
    r(a, b)
    assert r._cache.graphs.keys() & {k for k in r._cache.graphs if k[3][-2:] == (64, 128)}
    assert len(r._cache.graphs) == 2


def test_big_shapes_stay_eager(model: stereo.StereoModel, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rt, "_GRAPH_PIXELS", 0)
    r = _runner(model)
    assert r._cache is not None
    a, b = _pair(2, 96, 160)
    r(a, b)
    r(a, b)
    assert not r._cache.graphs


def test_capture_failure_falls_back(model: stereo.StereoModel, monkeypatch: pytest.MonkeyPatch) -> None:
    r = _runner(model)
    assert r._cache is not None
    a, b = _pair(5)
    r(a, b)

    def boom(*_args: object, **_kw: object) -> None:
        msg = "simulated"
        raise torch.OutOfMemoryError(msg)

    monkeypatch.setattr(r._cache, "_capture", boom)
    ref = _eager(r, a, b)
    torch.testing.assert_close(r(a, b)["disparity"], ref, rtol=0, atol=1e-3)  # failing capture: eager result
    assert not r._cache.graphs
    calls = []
    monkeypatch.setattr(r._cache, "_capture", lambda *x: calls.append(x))
    r(a, b)
    assert not calls  # no retry for the failed shape


def test_release_frees_memory(model: stereo.StereoModel) -> None:
    r = _runner(model)
    assert r._cache is not None
    a, b = _pair(6, 256, 512)
    for _ in range(3):
        r(a, b)
    assert r._cache.graphs
    torch.cuda.synchronize()
    held = torch.cuda.memory_reserved()
    model.release_graphs()
    assert not r._cache.graphs
    assert torch.cuda.memory_reserved() < held
    torch.testing.assert_close(r(a, b)["disparity"], _eager(r, a, b), rtol=0, atol=1e-3)  # recaptures cleanly


def test_opt_out() -> None:
    m = _load(cuda_graphs=False)
    r = _runner(m)
    assert r._cache is None
    a, b = _pair(7)
    for _ in range(3):
        out = r(a, b)["disparity"]
    torch.testing.assert_close(out, _eager(r, a, b), rtol=0, atol=1e-3)
    m.release_graphs()  # no-op
