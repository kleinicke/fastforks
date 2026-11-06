"""DINOv3 kernels CUDA graphs: replays on new data, shape changes and eviction, calls from two streams / threads.

Every graphed output must match the same model run eagerly (`_GRAPH_WORK = 0`: the same kernels without graphs).
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
import torch

from fastforks import features
from fastforks._core import kernels
from fastforks._core.errors import WeightsError
from fastforks.features.models.dinov3 import model as dinov3
from fastforks.shared.cuda_graphs import GraphCache

REF = "dinov3-vits16@1"
LAYERS = (0, 3)

pytestmark = [pytest.mark.model, pytest.mark.gpu]


@pytest.fixture(scope="module")
def model() -> features.FeaturesModel:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    try:
        return features.load(REF, mode="kernels", device="cuda")
    except WeightsError as e:
        pytest.skip(f"{REF} weights not available: {e}")


def _img(seed: int, h: int = 120, w: int = 150) -> torch.Tensor:
    """uint8 CHW already on the GPU: the call has no host sync (two-stream test)."""
    img = np.random.default_rng(seed).integers(0, 256, (3, h, w), dtype=np.uint8)
    return torch.from_numpy(img).cuda()


def _feats(model: features.FeaturesModel, img: torch.Tensor, size: int) -> torch.Tensor:
    r = model(img, size=size, layers=LAYERS)
    hs = r.hidden_states
    assert hs is not None
    return torch.cat([r.last_hidden_state.flatten(), *(t.flatten() for t in hs if t is not None)])


def _eager(
    model: features.FeaturesModel, img: torch.Tensor, size: int, monkeypatch: pytest.MonkeyPatch
) -> torch.Tensor:
    with monkeypatch.context() as m:
        m.setattr(dinov3, "_GRAPH_WORK", 0)
        return _feats(model, img, size)


def _max(model: features.FeaturesModel) -> dinov3.Kernels:
    runner = model._runner
    assert isinstance(runner, dinov3.Kernels)
    return runner


def _close(got: torch.Tensor, want: torch.Tensor) -> None:
    assert bool(torch.isfinite(got).all())
    assert float((got - want).norm() / want.norm()) < 1e-3


def _x(seed: int, size: int = 112) -> torch.Tensor:
    """A normalised network input for the runner (no host sync in the call, unlike the public API)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 3, size, size, generator=g).cuda()


def _run(model: features.FeaturesModel, x: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        out = _max(model)(x, LAYERS)
    return torch.cat([out["last_hidden_state"].flatten(), out["hidden_states"].flatten()])


def _run_eager(model: features.FeaturesModel, x: torch.Tensor, monkeypatch: pytest.MonkeyPatch) -> torch.Tensor:
    with monkeypatch.context() as m:
        m.setattr(dinov3, "_GRAPH_WORK", 0)
        return _run(model, x)


def test_replay_new_data(model, monkeypatch):
    model.release_graphs()
    a, b = _img(1), _img(2)
    ga = _feats(model, a, 112)  # first call of the shape: eager
    assert not _max(model)._cache.graphs
    gb, ga2 = _feats(model, b, 112), _feats(model, a, 112)  # capture + replay, replay
    assert len(_max(model)._cache.graphs) == 1
    ea, eb = _eager(model, a, 112, monkeypatch), _eager(model, b, 112, monkeypatch)
    assert float((ea - eb).norm() / ea.norm()) > 0.1  # the inputs differ
    _close(ga, ea)
    _close(gb, eb)
    _close(ga2, ea)


def test_shape_changes_and_eviction(model, monkeypatch):
    model.release_graphs()
    cache = _max(model)._cache
    monkeypatch.setattr(cache, "_max", 2)
    monkeypatch.setattr(cache, "_stale", 8)
    sizes = (112, 128, 144)
    want = {s: _eager(model, _img(s), s, monkeypatch) for s in sizes}
    graphs = _max(model)._cache.graphs
    for s in (
        sizes * 3
    ):  # 3 shapes (key: input height = size) cycling through 2 slots: 112 / 128 keep their graphs, 144 runs eagerly
        _close(_feats(model, _img(s), s), want[s])
    assert {k[0][-2] for k in graphs} == {112, 128}
    for _ in range(8):  # 112 unused for 8 graph-sized calls: stale, 144 replaces it
        _close(_feats(model, _img(144), 144), want[144])
        _close(_feats(model, _img(128), 128), want[128])
    assert {k[0][-2] for k in graphs} == {128, 144}
    for _ in range(2):  # 112 is not captured again while 128 / 144 are in use
        _close(_feats(model, _img(112), 112), want[112])
    assert {k[0][-2] for k in graphs} == {128, 144}
    model.release_graphs()
    assert not graphs
    _close(_feats(model, _img(112), 112), want[112])


def test_two_streams(model, monkeypatch):
    """Replays on two streams that start together (gate event) must not share the graph's buffers."""
    model.release_graphs()
    a, b = _x(3), _x(4)
    want_a, want_b = _run_eager(model, a, monkeypatch), _run_eager(model, b, monkeypatch)
    _run(model, a), _run(model, a)  # capture
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    busy = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    for _ in range(5):
        gate = torch.cuda.Event()
        with torch.cuda.stream(s1):
            for _ in range(10):  # s2 would otherwise finish before s1 starts
                busy @ busy
            gate.record()
            ga = _run(model, a)
        with torch.cuda.stream(s2):
            s2.wait_event(gate)
            gb = _run(model, b)
        torch.cuda.synchronize()
        _close(ga, want_a)
        _close(gb, want_b)


def test_default_stream_then_stream(monkeypatch):
    """A first graph call on the default stream (handle 0), the next one on another stream: the second must wait."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    try:
        m = features.load(REF, mode="kernels", device="cuda")  # fresh: no stream seen yet
    except WeightsError as e:
        pytest.skip(f"{REF} weights not available: {e}")
    a, b = _x(7), _x(8)
    want_a, want_b = _run_eager(m, a, monkeypatch), _run_eager(m, b, monkeypatch)  # no graph path: no stream seen
    s0, s2 = torch.cuda.default_stream(), torch.cuda.Stream()
    busy = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    with torch.cuda.stream(s0):
        _run(m, a), _run(m, a)  # eager, then captured: both on the default stream
        for _ in range(20):
            busy @ busy
        gate = torch.cuda.Event()
        gate.record()
        ga = _run(m, a)
    with torch.cuda.stream(s2):
        s2.wait_event(gate)
        gb = _run(m, b)
    torch.cuda.synchronize()
    _close(ga, want_a)
    _close(gb, want_b)


def test_two_threads(model, monkeypatch):
    model.release_graphs()
    xs = [_x(5), _x(6)]
    want = [_run_eager(model, x, monkeypatch) for x in xs]
    got: list[list[torch.Tensor]] = [[], []]

    def work(i: int) -> None:
        with torch.cuda.stream(torch.cuda.Stream()):
            got[i] = [_run(model, xs[i]) for _ in range(10)]
            torch.cuda.current_stream().synchronize()

    threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(_max(model)._cache.graphs) == 1
    for i in range(2):
        assert len(got[i]) == 10
        for g in got[i]:
            _close(g, want[i])


@pytest.mark.parametrize("ref", ["dinov3-vits16@1", "dinov3-vith16plus@1"])  # fh / hh attention, GELU / SwiGLU MLP
def test_eager_above_graph_work(ref, monkeypatch):
    """Above `_GRAPH_WORK` the kernels run eagerly (no graph, no PyTorch fallback) and match the graphed call.

    The graphed call is the one the GPU goldens validate (`test_golden.py`: 112 px is below the threshold).
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    try:
        m = features.load(ref, mode="kernels", device="cuda")
    except WeightsError as e:
        pytest.skip(f"{ref} weights not available: {e}")
    img = _img(5)
    _feats(m, img, 112)
    graphed = _feats(m, img, 112)  # second call: captured
    assert _max(m)._cache.graphs
    m.release_graphs()
    kernels.reset_fallbacks()
    with monkeypatch.context() as p:
        p.setattr(dinov3, "_GRAPH_WORK", 0)
        p.setattr(GraphCache, "_capture", lambda *_: pytest.fail("captured a CUDA graph above _GRAPH_WORK"))
        r = m(img, size=112, layers=LAYERS)
        eager = _feats(m, img, 112)
    assert r.backend.mode == "kernels"
    assert kernels.fallbacks() == {}
    _close(eager, graphed)
