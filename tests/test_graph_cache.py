"""GraphCache with list / dict outputs (DINOv3, SAM 3) and a failing capture (eager fallback)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest
import torch

from fastforks.shared.cuda_graphs import GraphCache

pytestmark = pytest.mark.gpu

# A forward that fails during capture after allocating (intermediates, cuBLAS workspace), and graphs that are evicted
# and released: under cudaMallocAsync, a captured allocation freed after a failed capture aborted the process.
_ASYNC_SCRIPT = textwrap.dedent(
    """
    import torch
    from fastforks.shared.cuda_graphs import GraphCache

    assert torch.cuda.memory.get_allocator_backend() == "cudaMallocAsync"
    w = torch.randn(256, 256, device="cuda")

    def failing(x):
        y = torch.relu(x @ w)
        d = torch.empty(2, device=x.device)
        d[0] = 1.5  # a CPU scalar copy: rejected during capture (RIFE's warp did this)
        return y * d[0]

    def fine(x):
        return torch.relu(x @ w) + 1

    for forward, captures in ((failing, False), (fine, True)):
        cache = GraphCache(forward, graphs=1, stale=1)
        for rows in (64, 64, 128, 128):  # the second shape evicts the first graph
            x = torch.randn(rows, 256, device="cuda")
            for _ in range(3):
                out = cache.run(rows, x)
                if out is not None:
                    torch.testing.assert_close(out, forward(x))
            assert bool(cache.graphs) == captures
        cache.release()
        torch.cuda.synchronize()
    print("survived")
    """
)


def test_failed_capture_under_cuda_malloc_async_does_not_abort():
    """ComfyUI's default allocator: a failed capture stays eager instead of aborting the process (RIFE kernels)."""
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync"}
    p = subprocess.run(
        [sys.executable, "-c", _ASYNC_SCRIPT], env=env, capture_output=True, text=True, timeout=300, check=False
    )
    assert p.returncode == 0, p.stderr[-4000:]
    assert "survived" in p.stdout


def test_failed_capture_is_logged(caplog: pytest.LogCaptureFixture):
    def forward(x: torch.Tensor) -> torch.Tensor:
        y = x * 2  # the graph is not empty (an empty one warns, an error under pytest's filters)
        if torch.cuda.is_current_stream_capturing():
            msg = "unsupported op during capture"
            raise RuntimeError(msg)
        return y

    cache = GraphCache(forward)
    x = torch.ones(4, device="cuda")
    with caplog.at_level("WARNING", logger="fastforks"):
        cache.run("k", x)
        cache.run("k", x)
    assert "CUDA graph capture failed" in caplog.text
    assert "unsupported op during capture" in caplog.text


@pytest.fixture(autouse=True)
def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")


def test_structured_outputs_replay_fresh_copies():
    def forward(x: torch.Tensor) -> dict[str, object]:
        return {"a": x * 2, "b": [x + 1, x - 1]}

    cache = GraphCache(forward)
    x = torch.arange(8.0, device="cuda")
    assert cache.run("k", x) is None  # first call of a key: the caller runs eagerly
    first = cache.run("k", x)  # captured
    assert len(cache.graphs) == 1
    second = cache.run("k", x + 10)  # replayed on new data
    assert torch.equal(first["a"], x * 2)
    assert torch.equal(second["a"], (x + 10) * 2)
    assert torch.equal(second["b"][1], x + 9)
    assert first["a"].data_ptr() != second["a"].data_ptr()  # fresh tensors, not the static buffers


def test_capture_failure_falls_back_to_eager():
    calls = {"capturing": 0}

    def forward(x: torch.Tensor) -> torch.Tensor:
        if torch.cuda.is_current_stream_capturing():
            calls["capturing"] += 1
            msg = "unsupported op during capture"
            raise RuntimeError(msg)
        return x * 2

    cache = GraphCache(forward)
    x = torch.ones(4, device="cuda")
    assert cache.run("k", x) is None
    assert cache.run("k", x) is None  # capture fails: no exception, the key stays eager
    assert not cache.graphs
    assert cache.run("k", x) is None
    assert calls["capturing"] == 1  # not retried
    assert torch.equal(forward(x), x * 2)  # the device is usable after the failed capture


_PAGE = 20 * 2**20  # the allocator's large-pool page (expandable segments) / small-buffer segment size


@pytest.mark.parametrize(
    "forward",
    [
        lambda x: (x * 2).relu() + 1,  # small intermediates
        lambda x: (x.repeat(4, 1) * 3).sum(0),  # large intermediates (32 MiB)
        lambda x: (x[:1024] @ x[:1024].T).softmax(-1),  # cuBLAS (its workspace moves into the graph pool)
    ],
    ids=["small", "large", "matmul"],
)
def test_graph_pool_adds_only_the_intermediates(forward):
    """Steady reserved memory of the replays <= eager reserved + the graph's intermediates in pages.

    A graph's private pool keeps its intermediates reserved (one page at least); anything above that is the eager or
    warm-up blocks kept next to it (igev++ once reserved twice its eager memory).
    """
    x = torch.rand(2048, 1024, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    forward(x)
    torch.cuda.synchronize()
    inter, eager = torch.cuda.max_memory_allocated() - base, torch.cuda.max_memory_reserved()
    cache = GraphCache(forward)
    for _ in range(3):  # eager, capture, replay
        cache.run("k", x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        cache.run("k", x)
    torch.cuda.synchronize()
    pages = -(-inter // _PAGE) * _PAGE + 2 * 2**20  # large pages + one small segment
    assert torch.cuda.max_memory_reserved() <= eager + pages, (inter, eager, torch.cuda.max_memory_reserved())
    cache.release()


def test_a_much_larger_shape_frees_the_small_graphs():
    """A GPU-bound shape must not run next to the pool of a small shape's graph (kernels reserved memory)."""
    cache = GraphCache(lambda x: x * 2)
    small = torch.ones(64, device="cuda")
    cache.run("s", small)
    assert cache.run("s", small) is not None
    assert len(cache.graphs) == 1
    assert cache.run("same size", small) is None  # not larger: the graph stays
    assert len(cache.graphs) == 1
    assert cache.run("large", torch.ones(256, device="cuda")) is None  # eager, and the small graph is released
    assert not cache.graphs
    cache.run("s", small)  # a returning shape captures again on its repeat
    assert cache.run("s", small) is not None


def test_shared_input_requires_matching_layout_and_survives_eviction():
    cache = GraphCache(lambda x, image: x + image, graphs=2, stale=1, share_inputs=(1,))

    def capture(key, x, image):
        assert cache.run(key, x, image) is None
        assert torch.equal(cache.run(key, x, image), x + image)
        return cache.graphs[key][1][1].data_ptr()

    x = torch.arange(9.0, device="cuda").reshape(3, 3)
    a = torch.full_like(x, 4)
    first = capture("a", x, a)
    assert capture("b", x, a + 2) == first
    # "a" is evicted, while "b" still owns the shared buffer.
    assert capture("c", x, (a + 3).T) != first
    assert torch.equal(cache.run("b", x, a + 9), x + a + 9)
    cache.release()
    assert not cache.graphs


def test_uploads_in_a_capture_outlive_the_staging_allocator():
    from fastforks._core.upload import device_tensor, to_device  # noqa: PLC0415

    def forward(x):
        scale = to_device(torch.tensor([2.0, 3.0]), x.device)
        shift = device_tensor([1.0, -1.0], device=x.device)
        return x * scale + shift

    cache = GraphCache(forward)
    x = torch.ones(4, 2, device="cuda")
    assert cache.run("k", x) is None  # first call of a key: the caller runs eagerly
    cache.run("k", x)  # captured
    graph = cache.graphs["k"][0]
    assert len(graph.uploads) == 2  # the pinned staging buffers, held with the graph
    for _ in range(8):  # freed pinned blocks are reused with other values
        to_device(torch.full((2,), 9.0), "cuda")
    torch.cuda.synchronize()
    torch.testing.assert_close(cache.run("k", x), torch.tensor([[3.0, 2.0]] * 4, device="cuda"))


def test_uploads_in_a_bare_capture_raise():
    from fastforks._core.upload import to_device  # noqa: PLC0415

    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    x = torch.zeros(2, device="cuda")
    with torch.cuda.stream(side), pytest.raises(RuntimeError, match="capture_uploads"):  # noqa: PT012
        graph.capture_begin(capture_error_mode="thread_local")
        try:
            x.add_(1)  # not an empty graph
            to_device(torch.ones(2), "cuda")
        finally:
            graph.capture_end()
