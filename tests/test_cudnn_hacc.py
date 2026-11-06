"""cudnn_hacc per-shape dispatch: plans on a geometry's first call (default), or after `BUILD_AFTER` torch calls."""

from __future__ import annotations

import gc
import weakref

import pytest
import torch
import torch.nn.functional as F

from fastforks._core import kernels
from fastforks._core.errors import KernelUnavailableError
from fastforks.kernels import cudnn_hacc
from fastforks.shared.cuda_graphs import GraphCache

pytestmark = pytest.mark.gpu

CL = torch.channels_last


@pytest.fixture
def conv() -> cudnn_hacc.CudnnHalfConv:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    cudnn_hacc.load(torch.device("cuda"))  # the self-test
    return cudnn_hacc.CudnnHalfConv()  # own plan cache


@pytest.fixture
def deferred(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optional deferral: a new geometry runs on torch in its first 3 model calls."""
    monkeypatch.setattr(cudnn_hacc, "BUILD_AFTER", 3)


def _xw(h: int, w: int, c: int = 32, k: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator("cuda").manual_seed(h * 1000 + w)
    x = torch.randn(1, c, h, w, device="cuda", generator=g).half().contiguous(memory_format=CL)
    wt = (0.1 * torch.randn(c, c, k, k, device="cuda", generator=g)).half().contiguous(memory_format=CL)
    return x, wt


def test_default_builds_on_the_first_call(conv: cudnn_hacc.CudnnHalfConv) -> None:
    """BUILD_AFTER 0: the first call of a geometry already runs its plan, so every call gives the same output."""
    assert cudnn_hacc.BUILD_AFTER == 0
    x, w = _xw(44, 52)
    kernels.begin_call()
    y1 = conv.conv(x, w, None, [1, 1], [1, 1])
    assert len(conv._plans) == 1
    assert not conv._seen
    assert not conv._deferred  # nothing ran on torch: no cuBLAS workspace free (device sync) on builds
    for _ in range(3):
        kernels.begin_call()
        torch.testing.assert_close(conv.conv(x, w, None, [1, 1], [1, 1]), y1, rtol=0, atol=0)


@pytest.mark.usefixtures("deferred")
def test_new_geometry_runs_torch_until_built(conv: cudnn_hacc.CudnnHalfConv) -> None:
    x, w = _xw(40, 56)
    want = F.conv2d(x, w, None, 1, 1)
    kernels.reset_fallbacks()
    for _ in range(cudnn_hacc.BUILD_AFTER):
        kernels.begin_call()
        for _ in range(3):  # several uses in one model call count as one
            torch.testing.assert_close(conv.conv(x, w, None, [1, 1], [1, 1]), want, rtol=0, atol=0)
        assert not conv._plans
    kernels.begin_call()
    y = conv.conv(x, w, None, [1, 1], [1, 1])
    assert len(conv._plans) == 1
    assert y.is_contiguous(memory_format=CL)
    torch.testing.assert_close(y, want, rtol=0, atol=0.05)  # fp16 accumulation
    assert kernels.fallbacks() == {}  # deferring is not a fallback


@pytest.mark.usefixtures("deferred")
def test_pointwise_runs_as_gemm_until_built(conv: cudnn_hacc.CudnnHalfConv) -> None:
    x, w = _xw(30, 50, c=64, k=1)
    b = torch.randn(64, device="cuda").half()
    want = F.conv2d(x.float(), w.float(), b.float())
    kernels.begin_call()
    y = conv.conv(x, w, b, [0, 0], [1, 1])  # deferred: a cuBLAS GEMM over the channels
    assert not conv._plans
    assert y.is_contiguous(memory_format=CL)
    torch.testing.assert_close(y.float(), want, rtol=1e-3, atol=1e-2)
    for _ in range(cudnn_hacc.BUILD_AFTER):
        kernels.begin_call()
        y = conv.conv(x, w, b, [0, 0], [1, 1])
    assert len(conv._plans) == 1
    torch.testing.assert_close(y.float(), want, rtol=1e-2, atol=0.1)  # fp16 accumulation


@pytest.mark.usefixtures("deferred")
def test_many_calls_build_without_model_calls(conv: cudnn_hacc.CudnnHalfConv) -> None:
    x, w = _xw(24, 40)
    kernels.begin_call()
    for _ in range(cudnn_hacc.BUILD_CALLS):
        conv.conv(x, w, None, [1, 1], [1, 1])
    assert not conv._plans
    conv.conv(x, w, None, [1, 1], [1, 1])
    assert len(conv._plans) == 1


@pytest.mark.usefixtures("deferred")
def test_capture_builds_at_once(conv: cudnn_hacc.CudnnHalfConv) -> None:
    x, w = _xw(32, 48)
    kernels.begin_call()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        conv.conv(x, w, None, [1, 1], [1, 1])  # eager warm-up: deferred
        assert not conv._plans
        with torch.cuda.graph(graph, stream=side):
            y = conv.conv(x, w, None, [1, 1], [1, 1])
    torch.cuda.current_stream().wait_stream(side)
    assert len(conv._plans) == 1
    graph.replay()
    e = next(iter(conv._plans.values()))
    assert isinstance(e, cudnn_hacc._Plan)
    ref = torch.empty_like(y)
    conv._execute(e, x, w, ref)
    torch.testing.assert_close(y, ref, rtol=0, atol=0)


def test_captured_workspace_is_the_graphs(conv: cudnn_hacc.CudnnHalfConv) -> None:
    """A graph keeps its own workspace, not the stream's cached one.

    A later plan with a bigger workspace on the capture stream (the next shape's capture) replaces the cached one; a
    graph holding it then wrote into freed memory (ETH3D size streams: illegal address on the next replay).
    """
    g = torch.Generator("cuda").manual_seed(0)
    x = torch.randn(1, 64, 112, 184, device="cuda", generator=g).half().contiguous(memory_format=CL)
    w = (0.1 * torch.randn(64, 9, 4, 4, device="cuda", generator=g)).half().contiguous(memory_format=CL)
    conv.deconv(x, w, [1, 1], [2, 2])  # Fast-ACVNet's 1/2 -> 1/1 upsampling: a plan with a workspace
    e = next(iter(conv._plans.values()))
    if not isinstance(e, cudnn_hacc._Plan) or not e.workspace:
        pytest.skip("no plan with a workspace")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        conv.deconv(x, w, [1, 1], [2, 2])  # eager warm-up: caches the side stream's workspace
        with torch.cuda.graph(graph, stream=side):
            y = conv.deconv(x, w, [1, 1], [2, 2])
        conv.free_workspaces()  # as a bigger plan replacing it
        junk = torch.full((e.workspace,), 7, device="cuda", dtype=torch.uint8)  # may reuse the freed block
        graph.replay()
    torch.cuda.current_stream().wait_stream(side)
    assert bool((junk == 7).all())
    torch.testing.assert_close(y, conv.deconv(x, w, [1, 1], [2, 2]), rtol=0, atol=0)


GEOMETRIES = [
    ((1, 32, 60, 80), (32, 32, 3, 3), (1, 1), (1, 1), False),
    ((1, 64, 61, 77), (128, 64, 3, 3), (1, 1), (2, 2), False),
    ((1, 64, 40, 64), (64, 64, 1, 1), (0, 0), (1, 1), False),
    ((1, 8, 12, 30, 40), (16, 8, 3, 3, 3), (1, 1, 1), (1, 1, 1), False),
    ((1, 64, 30, 40), (64, 32, 4, 4), (1, 1), (2, 2), True),
]


def _case(x: tuple[int, ...], w: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.memory_format]:
    mf = CL if len(x) == 4 else torch.channels_last_3d
    g = torch.Generator("cuda").manual_seed(0)
    wt = (0.1 * torch.randn(w, device="cuda", generator=g)).half().contiguous(memory_format=mf)
    xt = torch.randn(x, device="cuda", generator=g).half().contiguous(memory_format=mf)
    return xt, wt, mf


def _ctypes_only() -> cudnn_hacc.CudnnHalfConv:
    """An instance that builds every plan through ctypes (no native one-call build)."""
    c = cudnn_hacc.CudnnHalfConv()
    c._builder = None
    return c


@pytest.mark.parametrize(("x", "w", "pad", "stride", "dgrad"), GEOMETRIES)
def test_first_configs_give_the_full_list_plan(  # noqa: PLR0913, PLR0917 - one geometry per case
    conv: cudnn_hacc.CudnnHalfConv,
    monkeypatch: pytest.MonkeyPatch,
    x: tuple[int, ...],
    w: tuple[int, ...],
    pad: tuple[int, ...],
    stride: tuple[int, ...],
    dgrad: bool,  # noqa: FBT001 - a parametrized case
) -> None:
    """Fetching only the first heuristic configs picks the plan fetching all of them would; output ~ torch."""
    xt, wt, mf = _case(x, w)
    nd = len(x) - 2
    ref = (F.conv_transpose2d if nd == 2 else F.conv_transpose3d) if dgrad else (F.conv2d if nd == 2 else F.conv3d)
    e = conv._build(xt, wt, None, pad, stride, dgrad=dgrad)
    assert e is not None
    y = torch.empty(e.out_shape, device="cuda", dtype=torch.half, memory_format=mf)
    conv._execute(e, xt, wt, y)
    want = ref(xt.float(), wt.float(), None, stride, pad)
    torch.testing.assert_close(y.float(), want, rtol=0, atol=0.02 * float(want.abs().max()))
    monkeypatch.setattr(cudnn_hacc, "_FIRST_CONFIGS", (10_000,))  # one query of the full list
    full = _ctypes_only()._build(xt, wt, None, pad, stride, dgrad=dgrad)
    assert full is not None
    assert full.name == e.name


@pytest.mark.parametrize(("x", "w", "pad", "stride", "dgrad"), GEOMETRIES)
def test_native_build_gives_the_ctypes_plan(  # noqa: PLR0913, PLR0917 - one geometry per case
    conv: cudnn_hacc.CudnnHalfConv,
    x: tuple[int, ...],
    w: tuple[int, ...],
    pad: tuple[int, ...],
    stride: tuple[int, ...],
    dgrad: bool,  # noqa: FBT001 - a parametrized case
) -> None:
    """The native one-call build (`plan_build.cu`) gives the plan, workspace and output of the ctypes build."""
    if conv._builder is None:
        pytest.skip("no cudnn_hacc binary (FASTFORKS_KERNEL_PATH)")
    xt, wt, mf = _case(x, w)
    outs = []
    for c in (conv, _ctypes_only()):
        e = c._build(xt, wt, None, pad, stride, dgrad=dgrad)
        assert e is not None
        y = torch.empty(e.out_shape, device="cuda", dtype=torch.half, memory_format=mf)
        c._execute(e, xt, wt, y)
        outs.append((e.name, e.workspace, y))
    assert outs[0][:2] == outs[1][:2]
    assert torch.equal(outs[0][2], outs[1][2])


def test_native_build_falls_back_to_the_full_query(
    conv: cudnn_hacc.CudnnHalfConv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first config the native build cannot use (here: a skipped engine) goes to the ctypes query of more configs;
    without the binary every plan is built through ctypes."""
    x, w = _xw(36, 52)
    first = conv._build(x, w, None, (1, 1), (1, 1))
    assert first is not None
    monkeypatch.setattr(cudnn_hacc, "_SKIP_ENGINES", (*cudnn_hacc._SKIP_ENGINES, first.engine))
    other = cudnn_hacc.CudnnHalfConv()._build(x, w, None, (1, 1), (1, 1))
    assert other is not None
    assert other.engine != first.engine
    via_ctypes = _ctypes_only()._build(x, w, None, (1, 1), (1, 1))
    assert via_ctypes is not None
    assert other.name == via_ctypes.name

    def unavailable(*_: object) -> None:
        name = "cudnn_hacc"
        raise KernelUnavailableError(name, "no binary")

    monkeypatch.setattr(cudnn_hacc, "shared_library", unavailable)
    plain = cudnn_hacc.CudnnHalfConv()
    assert plain._builder is None
    assert plain._build(x, w, None, (1, 1), (1, 1)) is not None


@pytest.mark.parametrize(
    ("x", "k", "pad", "stride", "dil"),
    [
        ((1, 64, 61, 77), 64, (2, 2), (1, 1), (2, 2)),
        ((2, 32, 33, 45), 48, (4, 4), (1, 1), (4, 4)),
        ((1, 64, 97, 129), 64, (4, 4), (1, 1), (4, 4)),
        ((1, 32, 31, 37), 32, (0, 1), (2, 1), (2, 4)),
        ((1, 8, 9, 21, 23), 16, (2, 2, 2), (1, 1, 1), (2, 2, 2)),
    ],
)
def test_dilated_conv_matches_torch(  # noqa: PLR0913, PLR0917 - one geometry per case
    conv: cudnn_hacc.CudnnHalfConv,
    x: tuple[int, ...],
    k: int,
    pad: tuple[int, ...],
    stride: tuple[int, ...],
    dil: tuple[int, ...],
) -> None:
    """Dilated 3x3(x3) convs (mmseg ResNet-d8: dilation 2 / 4) ~ torch, on torch until built, then on a cuDNN plan."""
    nd = len(x) - 2
    mf = CL if nd == 2 else torch.channels_last_3d
    g = torch.Generator("cuda").manual_seed(1)
    xt = torch.randn(x, device="cuda", generator=g).half().contiguous(memory_format=mf)
    wt = (0.1 * torch.randn((k, x[1], *[3] * nd), device="cuda", generator=g)).half().contiguous(memory_format=mf)
    b = torch.randn(k, device="cuda", generator=g).half()
    ref = F.conv2d if nd == 2 else F.conv3d
    want = ref(xt.float(), wt.float(), b.float(), stride, pad, dil)
    for _ in range(cudnn_hacc.BUILD_AFTER + 1):
        kernels.begin_call()
        y = conv.conv(xt, wt, b, pad, stride, dil)
        assert y.shape == want.shape
        torch.testing.assert_close(y.float(), want, rtol=0, atol=0.02 * float(want.abs().max()))
    assert len(conv._plans) == 1
    assert isinstance(next(iter(conv._plans.values())), cudnn_hacc._Plan)  # cuDNN ran it, not the torch fallback
    assert kernels.fallbacks() == {}


@pytest.mark.parametrize(
    ("x", "w", "pad", "stride", "dil"),
    [
        ((1, 32, 20, 20), (32, 16, 3, 3), (1, 1), (1, 1), (1, 1)),  # groups 2
        ((1, 32, 20, 20), (32, 32, 3, 3), (1,), (1, 1), (1, 1)),  # padding rank
        ((1, 32, 20, 20), (32, 32, 3, 3), (1, 1), (0, 1), (1, 1)),  # stride 0
        ((1, 32, 20, 20), (32, 32, 3, 3), (1, 1), (1, 1), (0, 1)),  # dilation 0
        ((1, 32, 6, 6), (32, 32, 3, 3), (0, 0), (1, 1), (4, 4)),  # empty output
    ],
)
def test_unsupported_geometry_raises(  # noqa: PLR0913, PLR0917 - one geometry per case
    conv: cudnn_hacc.CudnnHalfConv,
    x: tuple[int, ...],
    w: tuple[int, ...],
    pad: tuple[int, ...],
    stride: tuple[int, ...],
    dil: tuple[int, ...],
) -> None:
    xt = torch.randn(x, device="cuda").half().contiguous(memory_format=CL)
    wt = torch.randn(w, device="cuda").half().contiguous(memory_format=CL)
    with pytest.raises(ValueError, match="unsupported geometry"):
        conv._build(xt, wt, None, pad, stride, dilation=dil)
    with pytest.raises(ValueError, match="unsupported geometry"):  # deconv: w `[C, K, ...]` must match x's channels
        conv._build(xt, wt.new_empty((16, 32, 3, 3)), None, (1, 1), (2, 2), dgrad=True)


def test_plan_outlives_its_instance() -> None:
    """A plan keeps its cuDNN handle alive; the handle is destroyed (`cudnnDestroy`) once nothing uses it."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    x, w = _xw(16, 24)
    conv = cudnn_hacc.CudnnHalfConv()
    handle = weakref.ref(conv.handle)
    e = conv._build(x, w, None, (1, 1), (1, 1))
    assert e is not None
    del conv
    gc.collect()
    assert handle() is not None  # held by the plan
    del e
    gc.collect()
    assert handle() is None


def test_graph_cache_capture_after() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    cache = GraphCache(lambda x: x * 2, capture_after=3)
    x = torch.ones(4, device="cuda")
    for _ in range(3):
        assert cache.run("k", x) is None  # eager
    assert not cache.graphs
    torch.testing.assert_close(cache.run("k", x), x * 2)  # 4th call: captured
    assert len(cache.graphs) == 1
    assert GraphCache(lambda x: x, capture_after=0)._capture_after == 1  # one-off shapes never capture


def test_capture_after_follows_the_plans(monkeypatch: pytest.MonkeyPatch) -> None:
    """Graph models capture on a shape's 4th call, and never before its plans are built (`BUILD_AFTER`)."""
    assert cudnn_hacc.capture_after() == 3
    monkeypatch.setattr(cudnn_hacc, "BUILD_AFTER", 5)
    assert cudnn_hacc.capture_after() == 5


def test_workspace_freed_with_the_last_model() -> None:
    """A loaded kernels model keeps the plan workspace; it goes back to the allocator with the last one."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from fastforks import stereo  # noqa: PLC0415

    cudnn_hacc.load(torch.device("cuda"))
    gc.collect()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    a = stereo.load("coex-sceneflow@1", mode="kernels", device="cuda")
    a(*(torch.rand(1, 3, 384, 640, device="cuda") for _ in range(2)))
    held = torch.cuda.memory_allocated()
    b = stereo.load("coex-sceneflow@1", mode="kernels", device="cuda")
    del a
    gc.collect()
    assert cudnn_hacc._users == 1
    assert any(c.numel() for c in cudnn_hacc._instance()._ws.values()) or held > base
    del b
    gc.collect()
    torch.cuda.empty_cache()
    assert cudnn_hacc._users == 0
    assert not cudnn_hacc._instance()._ws
    assert torch.cuda.memory_allocated() - base < 8 * 2**20
