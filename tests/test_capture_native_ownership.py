"""CUDA capture stream ownership and fallback policy."""

# ruff: noqa: N802 - the fake driver implements CUDA driver ABI names
from __future__ import annotations

import ctypes
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fastforks.shared import cuda_graphs as graph


class Driver:
    """Minimal fake driver with context IDs independent of pointer addresses."""

    context = 0x1234
    identity = 10
    index = 0
    failed = False

    def __init__(self):
        self.creates = []
        self.destroys = []

    def cuCtxGetCurrent(self, out):
        ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p))[0] = self.context
        return 0

    def cuCtxGetId(self, context, out):
        assert context.value == self.context
        ctypes.cast(out, ctypes.POINTER(ctypes.c_uint64))[0] = self.identity
        return 0

    def cuCtxGetDevice(self, out):
        ctypes.cast(out, ctypes.POINTER(ctypes.c_int))[0] = self.index
        return 0

    def cuDeviceGet(self, out, index):
        ctypes.cast(out, ctypes.POINTER(ctypes.c_int))[0] = index
        return 0

    def cuStreamCreate(self, out, flags):
        assert flags == 1
        self.creates.append(self.identity)
        if self.failed:
            return 1
        ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p))[0] = 0x1000 + len(self.creates) * 0x100
        return 0

    def cuStreamDestroy_v2(self, pointer):
        self.destroys.append(pointer.value)
        return 0


@pytest.fixture
def driver(monkeypatch):
    native, pooled = Driver(), object()

    def check(code, name):
        if code:
            raise RuntimeError(name)

    monkeypatch.setattr(graph, "_NATIVE_CAPTURE", {})
    monkeypatch.setattr(graph, "_libcuda", lambda: native)
    monkeypatch.setattr(graph, "_check", check)
    monkeypatch.setattr(graph, "_stream", lambda _index, _slot: pooled)
    monkeypatch.setattr(graph, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(torch.cuda, "ExternalStream", lambda p, *, device: SimpleNamespace(handle=p, index=device))
    return native, pooled


def test_cached_by_context_identity_and_process(driver, monkeypatch):
    native, _ = driver
    first = graph.capture_stream("cuda:0")
    assert first is graph.capture_stream("cuda:0")
    # A later context on the same device can reuse the pointer, but not the ID.
    native.identity = 11
    second = graph.capture_stream("cuda:0")
    assert second is not first
    native.identity = 10
    assert graph.capture_stream("cuda:0") is first
    monkeypatch.setattr(graph, "os", SimpleNamespace(getpid=lambda: -1))
    assert graph.capture_stream("cuda:0") is not first
    assert native.creates == [10, 11, 10]
    assert not native.destroys


def test_does_not_create_on_wrong_device(driver):
    native, pooled = driver
    native.index = 1
    assert graph.capture_stream("cuda:0") is pooled
    assert not native.creates


def test_native_stream_created_once_across_threads(driver):
    native, _ = driver
    barrier = threading.Barrier(4)

    def get():
        barrier.wait(timeout=5)
        return graph.capture_stream("cuda:0")

    with ThreadPoolExecutor(max_workers=4) as executor:
        streams = [result.result(timeout=10) for result in [executor.submit(get) for _ in range(4)]]
    assert all(stream is streams[0] for stream in streams)
    assert native.creates == [10]


def test_native_setup_failure_falls_back_for_that_context(driver):
    native, pooled = driver
    native.failed = True
    assert graph.capture_stream("cuda:0") is pooled
    assert graph.capture_stream("cuda:0") is pooled
    native.identity = 11
    native.failed = False
    assert graph.capture_stream("cuda:0") is not pooled
    assert native.creates == [10, 11]


def test_wrapper_failure_destroys_only_unused_same_context_handle(driver, monkeypatch):
    native, pooled = driver

    def fail(*_args, **_kwargs):
        msg = "Cannot wrap stream"
        raise RuntimeError(msg)

    monkeypatch.setattr(torch.cuda, "ExternalStream", fail)
    assert graph.capture_stream("cuda:0") is pooled
    assert len(native.destroys) == 1


def test_context_change_during_wrapping_falls_back(driver, monkeypatch):
    native, pooled = driver

    def switch(pointer, *, device):
        native.context = 0x2345
        return SimpleNamespace(handle=pointer, index=device)

    monkeypatch.setattr(torch.cuda, "ExternalStream", switch)
    assert graph.capture_stream("cuda:0") is pooled
    assert not native.destroys


def test_first_call_can_materialize_lazy_cuda_context(driver, monkeypatch):
    native, pooled = driver
    native.context = None

    def allocate(*args, device):
        assert device == torch.device("cuda:0")
        native.context = 0x1234

    monkeypatch.setattr(torch, "empty", allocate)
    assert graph.capture_stream("cuda:0") is not pooled
    assert native.creates == [10]


@pytest.mark.parametrize(("platform", "allocator"), [("win32", "native"), ("linux", "cudaMallocAsync")])
def test_platform_and_allocator_gates_keep_pooled_stream(driver, monkeypatch, platform, allocator):
    native, pooled = driver
    monkeypatch.setattr(graph.sys, "platform", platform)
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: allocator)
    assert graph.capture_stream("cuda:0") is pooled
    assert not native.creates


def test_missing_context_id_api_keeps_pooled_stream(driver, monkeypatch):
    native, pooled = driver
    monkeypatch.delattr(Driver, "cuCtxGetId")
    assert graph.capture_stream("cuda:0") is pooled
    assert not native.creates


def _device() -> torch.device:
    if sys.platform != "linux" or not torch.cuda.is_available() or torch.cuda.get_allocator_backend() != "native":
        pytest.skip("Linux CUDA with native allocation required")
    return torch.device("cuda", torch.cuda.current_device())


@pytest.mark.gpu
def test_external_stream_with_two_threads_and_caller_streams():
    device = _device()
    callers = [torch.cuda.Stream(device=device) for _ in range(2)]
    barrier = threading.Barrier(2)

    def run(i):
        with torch.cuda.device(device), torch.cuda.stream(callers[i]):
            inp = torch.full((128,), float(i + 1), device=device)
            barrier.wait(timeout=10)
            owner = graph.capture_stream(device)
            cache = graph.GraphCache(torch.square, capture_warmup=False)
            assert cache.run("same", inp) is None
            initial = cache.run("same", inp)
            changed = cache.run("same", inp + 2)
            assert torch.equal(initial, inp.square())
            assert torch.equal(changed, (inp + 2).square())
            return owner, cache

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run, i) for i in range(2)]
        results = [future.result(timeout=40) for future in futures]
    stream = results[0][0]
    assert isinstance(stream, torch.cuda.ExternalStream)
    assert results[1][0] is stream
    for _, cache in results:
        cache.release()
        assert not cache.graphs
    assert graph.capture_stream(device) is stream


@pytest.mark.gpu
def test_omni_dc_direct_graph_replays_changed_system():
    from fastforks.depth.models.omni_dc import model as omni  # noqa: PLC0415 - only needed for CUDA

    device = _device()
    generator = torch.Generator().manual_seed(0)
    d, b = (torch.rand((1, 1, 32, 48), generator=generator).to(device) + 0.5 for _ in range(2))
    with torch.inference_mode():
        cg = omni.GraphCG(d, b)
        stream = graph.capture_stream(device)
        assert isinstance(stream, torch.cuda.ExternalStream)
        for diagonal, rhs in ((d, b), (d + 0.25, b.flip(-1))):
            expected = omni._KernelCG(diagonal, rhs).run()
            actual = omni._graph_solve(cg, diagonal, rhs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.cuda.synchronize()
        del cg
    assert graph.capture_stream(device) is stream


@pytest.mark.gpu
def test_clean_subprocess_exit_with_driver_owned_stream_and_live_graph():
    _device()
    code = """
import torch
from fastforks.shared.cuda_graphs import GraphCache, capture_stream
x = torch.ones(64, device='cuda')
cache = GraphCache(torch.square)
assert cache.run('key', x) is None
assert torch.equal(cache.run('key', x + 2), (x + 2).square())
assert cache.graphs
assert isinstance(capture_stream(x.device), torch.cuda.ExternalStream)
torch.cuda.synchronize()
"""
    subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parents[1], check=True, timeout=45)
