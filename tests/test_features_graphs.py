"""Kernels CUDA graphs of the feature families: replays on new data, shape changes and eviction, streams / threads.

DINOv3 (its own runner): every graphed output must match the same model run eagerly (`_GRAPH_WORK = 0`: the same
kernels without graphs). The timm families on `libnet.features.Kernels` (SigLIP 2, CLIP, Perception Encoder,
C-RADIO; the `test_timm_*` tests): a small call dispatches its GEMMs the same way with and without a graph, so a
replay must equal the eager call (`runner.graphs = None`) bit for bit, and so must a call with hidden states.

A capture that fails marks its shape eager without a word (`GraphCache._failed`), and the goldens make a single call,
so they cannot see it: the tests also check that every family's graph is really captured and replayed.
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
from fastforks.shared.libnet import features as timm_features

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


# --- the timm families on `libnet.features.Kernels` -----------------------------------------------------------------

# the smallest variant of each family: SigLIP 2, CLIP (pre-norm, QuickGELU), PE (Eva, RoPE), C-RADIO; its native size
# (C-RADIO has no native size: its position table is resampled from 128 x 128 at every size)
TIMM = {"siglip2-b16-256@1": 256, "clip-b16-openai@1": 224, "pe-core-t16-384@1": 384, "c-radio-v3-b@1": 224}
# relative L2 bounds, fixed: 1.5 x the worst of the four refs measured on an RTX 5060 Ti, rounded up to 2 digits
NATIVE_REL = 0.037  # worst 0.0246 (SigLIP 2 B/16): kernels (fp16) against portable float32 at the native size
# worst 0.0262 (SigLIP 2 B/16): a graph-sized call (small GEMMs on cuBLAS) against the same call on the fused kernels
GEMM_REL = 0.040
OUTPUTS = ("last_hidden_state", "pooler_output")


@pytest.fixture(scope="module", params=list(TIMM))
def timm_model(request: pytest.FixtureRequest) -> features.FeaturesModel:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    try:
        return features.load(request.param, mode="kernels", device="cuda")
    except WeightsError as e:
        pytest.skip(f"{request.param} weights not available: {e}")


def _runner(model: features.FeaturesModel) -> timm_features.Kernels:
    runner = model._runner
    assert isinstance(runner, timm_features.Kernels)
    return runner


def _cache(model: features.FeaturesModel) -> GraphCache:
    cache = _runner(model).graphs
    assert cache is not None
    return cache


def _t(seed: int, h: int = 112, w: int = 112) -> torch.Tensor:
    """A normalised network input (the sides are multiples of 14 and 16 at the defaults)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 3, h, w, generator=g).cuda()


def _timm(model: features.FeaturesModel, x: torch.Tensor, layers: tuple[int, ...] = ()) -> torch.Tensor:
    with torch.inference_mode():
        out = _runner(model)(x, layers)
    return torch.cat([out[k].flatten() for k in OUTPUTS])


def _timm_eager(model: features.FeaturesModel, x: torch.Tensor, monkeypatch: pytest.MonkeyPatch) -> torch.Tensor:
    """The same call without the graph cache (the same kernels and GEMM dispatch)."""
    with monkeypatch.context() as m:
        m.setattr(_runner(model), "graphs", None)
        return _timm(model, x)


def _same(got: torch.Tensor, want: torch.Tensor) -> None:
    assert bool(torch.isfinite(got).all())
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_timm_replay_new_data(timm_model, monkeypatch):
    model = timm_model
    model.release_graphs()
    graphs = _cache(model).graphs
    a, b = _t(1), _t(2)
    ga = _timm(model, a)  # first call of the shape: eager
    assert not graphs
    gb, ga2 = _timm(model, b), _timm(model, a)  # capture + replay, replay
    assert len(graphs) == 1, f"no graph captured (failed: {list(_cache(model)._failed)})"
    ea, eb = _timm_eager(model, a, monkeypatch), _timm_eager(model, b, monkeypatch)
    assert float((ea - eb).norm() / ea.norm()) > 0.1  # the inputs differ
    for got, want in ((ga, ea), (gb, eb), (ga2, ea)):
        _same(got, want)
    _same(_timm(model, a, (0, 1)), ea)  # with hidden states (never graphed): the same features
    assert len(graphs) == 1
    model.release_graphs()
    assert not graphs


def test_timm_shape_changes_and_eviction(timm_model, monkeypatch):
    model = timm_model
    model.release_graphs()
    cache = _cache(model)
    monkeypatch.setattr(cache, "_max", 2)
    monkeypatch.setattr(cache, "_stale", 8)
    sizes = (112, 224, 336)  # multiples of both patch sizes
    want = {s: _timm_eager(model, _t(s, s, s), monkeypatch) for s in sizes}
    graphs = cache.graphs
    for s in sizes * 3:  # 3 shapes cycling through 2 slots: 112 / 224 keep their graphs, 336 runs eagerly
        _same(_timm(model, _t(s, s, s)), want[s])
    assert {k[2][-2] for k in graphs} == {112, 224}
    for _ in range(8):  # 112 unused for 8 graph-sized calls: stale, 336 replaces it
        _same(_timm(model, _t(336, 336, 336)), want[336])
        _same(_timm(model, _t(224, 224, 224)), want[224])
    assert {k[2][-2] for k in graphs} == {224, 336}
    for _ in range(2):  # 112 is not captured again while 224 / 336 are in use
        _same(_timm(model, _t(112)), want[112])
    assert {k[2][-2] for k in graphs} == {224, 336}
    model.release_graphs()
    assert not graphs
    _same(_timm(model, _t(112)), want[112])


def test_timm_tables_outlive_their_cache(timm_model, monkeypatch):
    """A graph replays with the address of its position / RoPE tables: `_Walk` dropping them must not free them.

    Capture grid A, call `_TABLES` other grids once each (A's tables leave the walk's cache), fill the freed blocks,
    replay A.
    """
    model = timm_model
    model.release_graphs()
    walk = _runner(model).walk
    a = _t(1, 224, 224)
    want = _timm_eager(model, a, monkeypatch)
    _timm(model, a), _timm(model, a)  # eager, captured
    assert len(_cache(model).graphs) == 1
    for i in range(timm_features._TABLES):  # other grids, never captured (one call each)
        _timm(model, _t(2, 112, 112 + 112 * i))
    assert len(walk._tables) == timm_features._TABLES
    assert all(k[:2] != (224 // _runner(model).patch,) * 2 for k in walk._tables)  # A's entry is gone
    torch.cuda.synchronize()
    junk = [torch.full((1 << (i % 12 + 8),), 1e4, device="cuda", dtype=torch.half) for i in range(4096)]
    _same(_timm(model, a), want)  # a replay
    assert len(_cache(model).graphs) == 1
    del junk


def test_timm_tables_lru(timm_model):
    """The walk keeps the most recently used grids: a grid in use is not dropped by `_TABLES` newer ones."""
    model = timm_model
    walk = _runner(model).walk
    a = _t(1)
    p = _runner(model).patch

    def table() -> tuple:
        _timm(model, a, (1,))  # eager (hidden states): reads A's tables from the walk's cache
        return next(v for k, v in walk._tables.items() if k[:2] == (112 // p, 112 // p))

    first = table()
    for i in range(timm_features._TABLES + 2):
        _timm(model, _t(2, 112, 224 + 112 * i), (1,))
        assert table() is first  # never resampled again
    assert len(walk._tables) == timm_features._TABLES


def test_timm_two_streams(timm_model, monkeypatch):
    """Replays on two streams that start together (gate event) must not share the graph's buffers."""
    model = timm_model
    model.release_graphs()
    a, b = _t(3), _t(4)
    want_a, want_b = _timm_eager(model, a, monkeypatch), _timm_eager(model, b, monkeypatch)
    _timm(model, a), _timm(model, a)  # capture
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    busy = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    for _ in range(5):
        gate = torch.cuda.Event()
        with torch.cuda.stream(s1):
            for _ in range(10):  # s2 would otherwise finish before s1 starts
                busy @ busy
            gate.record()
            ga = _timm(model, a)
        with torch.cuda.stream(s2):
            s2.wait_event(gate)
            gb = _timm(model, b)
        torch.cuda.synchronize()
        _same(ga, want_a)
        _same(gb, want_b)


def test_timm_two_threads(timm_model, monkeypatch):
    model = timm_model
    model.release_graphs()
    xs = [_t(5), _t(6)]
    want = [_timm_eager(model, x, monkeypatch) for x in xs]
    got: list[list[torch.Tensor]] = [[], []]

    def work(i: int) -> None:
        with torch.cuda.stream(torch.cuda.Stream()):
            got[i] = [_timm(model, xs[i]) for _ in range(10)]
            torch.cuda.current_stream().synchronize()

    threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(_cache(model).graphs) == 1
    for i in range(2):
        assert len(got[i]) == 10
        for g in got[i]:
            _same(g, want[i])


def test_timm_eager_above_graph_work(timm_model, monkeypatch):
    """Above `_GRAPH_WORK` the kernels run eagerly (no graph, no PyTorch fallback), on the fused GEMM kernels.

    The graph-sized call is the one the GPU goldens validate (`test_golden.py`: 112 px is below the threshold); the
    two differ by the GEMMs' fp16 rounding (cuBLAS below `conv_ops.LIN_GRAPH_ROWS` rows in a graph-sized call).
    """
    model = timm_model
    model.release_graphs()
    img = _img(5)
    small = model(img, size=112, layers=(0, 1))
    kernels.reset_fallbacks()
    with monkeypatch.context() as p:
        p.setattr(timm_features, "_GRAPH_WORK", 0)
        p.setattr(GraphCache, "_capture", lambda *_: pytest.fail("captured a CUDA graph above _GRAPH_WORK"))
        for _ in range(3):
            model(img, size=112)
        big = model(img, size=112, layers=(0, 1))
    assert big.backend.mode == "kernels"
    assert kernels.fallbacks() == {}
    assert not _cache(model).graphs
    rel = float((big.last_hidden_state - small.last_hidden_state).norm() / small.last_hidden_state.norm())
    assert rel < GEMM_REL


def test_timm_native_size_matches_portable(timm_model):
    """kernels at the variant's native size (the goldens run at 112 px, a resampled grid) against portable float32."""
    model = timm_model
    size = TIMM[model.variant.ref]
    try:
        ref = features.load(model.variant.ref, mode="portable", device="cuda", dtype="float32")
    except WeightsError as e:
        pytest.skip(f"{model.variant.ref} weights not available: {e}")
    img = _img(7, 300, 300)
    for _ in range(2):  # eager, then the captured graph
        got, want = model(img, size=(size, size)), ref(img, size=(size, size))
        assert got.grid == want.grid == (size // model.stride,) * 2
        for k in OUTPUTS:
            g, w = getattr(got, k), getattr(want, k)
            assert bool(torch.isfinite(g).all())
            rel = float((g - w).norm() / w.norm())
            assert rel < NATIVE_REL, f"{k}: {rel:.3g}"
