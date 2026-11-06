"""A model that owns a `SmallShapeGraphs` of its own method is freed by refcount, without the cycle collector.

Captures and side streams use shared streams: cuBLAS keeps a 32 MB workspace per stream for the life of the process,
so a new stream per capture or per loaded model left 32 MB behind each time (depth-pro, sam3 kernels).
"""

from __future__ import annotations

import gc
import weakref

import pytest
import torch

from fastforks.shared.cuda_graphs import SmallShapeGraphs, capture_stream, side_stream


class _Owner:
    def __init__(self, pixels: int = 10) -> None:
        self._graphs = SmallShapeGraphs(self._forward, pixels=pixels)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ x + 1


def test_owner_is_freed_without_gc():
    gc.collect()
    gc.disable()
    try:
        owner = _Owner()
        ref = weakref.ref(owner)
        assert owner._graphs(torch.zeros(1, 1, 2, 2)).sum() == 4  # CPU input: runs eagerly through the method
        del owner
        assert ref() is None
    finally:
        gc.enable()


def _allocated_mb() -> float:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated() / 2**20


@pytest.mark.gpu
def test_captures_leave_no_memory_behind():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    dev = torch.device("cuda", torch.cuda.current_device())
    assert capture_stream("cuda") is capture_stream(dev)
    assert side_stream(dev) is side_stream("cuda", 0)
    assert len({capture_stream(dev), side_stream(dev, 0), side_stream(dev, 1)}) == 3
    x = torch.randn(1, 1, 64, 64, device=dev)
    left = []
    for _ in range(4):  # a new model each time: its graph is captured on the second call
        owner = _Owner(pixels=64 * 64)
        for _ in range(3):
            torch.testing.assert_close(owner._graphs(x), x @ x + 1)
        assert owner._graphs.cache is not None
        assert len(owner._graphs.cache.graphs) == 1
        del owner
        left.append(_allocated_mb())
    assert left[1:] == left[:-1], f"MB left after each model: {left}"
