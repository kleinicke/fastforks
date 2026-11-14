"""GraphCache with list / dict outputs (DINOv3, SAM 3) and a failing capture (eager fallback)."""

from __future__ import annotations

import pytest
import torch

from fastforks.shared.cuda_graphs import GraphCache

pytestmark = pytest.mark.gpu


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
