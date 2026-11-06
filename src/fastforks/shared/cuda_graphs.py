"""A small per-model cache of CUDA graphs for launch-bound forwards (one graph per input-shape key).

Conventions (as the DINOv3 backbone): a shape is captured after `capture_after` eager calls (at least 1, so one-off
shapes never capture; default 1: on its second call; models on `cudnn_hacc` plans pass `cudnn_hacc.capture_after()`:
on the 4th call, so a stream of new sizes seen 1-3 times captures nothing); when the cache is full a
graph unused for `stale` graph-sized calls makes room, else the new shape runs eagerly; a capture failure (OOM,
unsupported op) marks the shape eager for good and frees the partial pool; a capture (failed or not) empties the
allocator cache once, after the lock is released; calls hold a per-cache lock and wait for the
previous call's stream, so threads / streams never read each other's static buffers; `release()` frees the pool on the
device. Nothing here empties the process-wide allocator cache while the lock is held.

Every warm-up + capture runs on one shared stream per device (`capture_stream`, under `CAPTURE_LOCK`), and kernels
that overlap work on side streams take them from `side_stream`: cuBLAS keeps a 32 MB workspace per stream for the
life of the process, so a new stream per capture or per loaded model grew the memory by 32 MB each time.
"""

from __future__ import annotations

import functools
import inspect
import threading
import weakref
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

_FAILED = 256  # fixed: shapes remembered as not capturable

# serialises the warm-ups + captures on the shared capture stream (reentrant: a captured forward may hold it again)
CAPTURE_LOCK = threading.RLock()


@functools.cache
def _stream(index: int, _slot: int) -> torch.cuda.Stream:
    return torch.cuda.Stream(index)


def _index(device: torch.device | str) -> int:
    index = torch.device(device).index
    return torch.cuda.current_device() if index is None else index


def capture_stream(device: torch.device | str) -> torch.cuda.Stream:
    """The stream of every graph warm-up and capture on `device` (hold `CAPTURE_LOCK` while using it)."""
    return _stream(_index(device), -1)


def side_stream(device: torch.device | str, i: int = 0) -> torch.cuda.Stream:
    """Side stream `i` of `device`, shared by every model that overlaps work (callers order it both ways)."""
    return _stream(_index(device), i)


def _weak(forward: Callable[..., Any]) -> Callable[..., Any]:
    """`forward`, held weakly when it is a bound method: a model owning its cache is then freed without gc."""
    if not inspect.ismethod(forward):
        return forward
    ref = weakref.WeakMethod(forward)

    def call(*args: torch.Tensor) -> Any:  # noqa: ANN401 - a tensor or a list / dict of tensors
        fn = ref()
        if fn is None:
            msg = "the model owning this graph cache was freed"
            raise ReferenceError(msg)
        return fn(*args)

    return call


def _copy(out: Any) -> Any:  # noqa: ANN401 - a tensor or a list / tuple / dict of them
    """Fresh copies of a graph's output, in the same structure."""
    if isinstance(out, torch.Tensor):
        return out.clone()
    if isinstance(out, dict):
        return {k: _copy(v) for k, v in out.items()}
    return type(out)(_copy(v) for v in out)


class GraphCache:
    """Replays `forward(*static inputs)` from a CUDA graph per key (see the module docstring).

    `forward` returns a tensor or a list / tuple / dict of tensors; `run` returns the same structure of fresh copies.
    """

    def __init__(
        self,
        forward: Callable[..., Any],
        *,
        graphs: int = 2,
        stale: int = 64,
        seen: int = 64,
        capture_after: int = 1,
    ) -> None:
        """`forward` runs the model on tensors; at most `graphs` graphs are kept; `capture_after` eager calls first."""
        self._forward = _weak(forward)
        self._capture_after = max(1, capture_after)  # one-off shapes never capture
        self._max, self._stale, self._seen_max = graphs, stale, seen
        # key -> (graph, static inputs, static output)
        self.graphs: dict[Hashable, tuple[torch.cuda.CUDAGraph, tuple[torch.Tensor, ...], Any]] = {}
        self._used: dict[Hashable, int] = {}
        self._seen: OrderedDict[Hashable, int] = OrderedDict()  # key -> eager calls
        self._failed: OrderedDict[Hashable, None] = OrderedDict()
        self._calls = 0
        self._pool: Any = None
        self._stream: torch.cuda.Stream | None = None
        self._lock = threading.Lock()

    def run(self, key: Hashable, *inputs: torch.Tensor, post: Callable[[Any], Any] | None = None) -> Any:  # noqa: ANN401 - as `forward`
        """The output (fresh tensors) from the graph of `key`, or None: the caller runs eagerly.

        `post(static output)` replaces the copy when it returns new tensors anyway (e.g. a dtype conversion).
        """
        failed = captured = False
        result = None
        with self._lock:
            cur = torch.cuda.current_stream(inputs[0].device)
            if self._stream is not None and self._stream != cur:
                cur.wait_stream(self._stream)  # the previous call's replay + copy-out are ordered before ours
            self._stream = cur
            self._calls += 1
            entry = self.graphs.get(key)
            if entry is None and key not in self._failed:
                entry, failed = self._add(key, inputs)
                captured = entry is not None
            if entry is not None:
                self._used[key] = self._calls
                graph, static, out = entry
                for s, x in zip(static, inputs, strict=True):
                    s.copy_(x)
                graph.replay()
                result = _copy(out) if post is None else post(out)
        if failed or captured:
            # outside the lock: the partial pool of a failed capture, or the blocks the eager calls and the capture
            # warm-up cached (the warm-up's on the capture stream, which the caller's stream never reuses): replays
            # allocate nothing, so these blocks only doubled the reserved memory (igev++ 1454 -> 720 MiB)
            torch.cuda.empty_cache()
        return result

    def _add(self, key: Hashable, inputs: tuple[torch.Tensor, ...]) -> tuple[Any, bool]:
        """Capture `key` after `capture_after` eager calls: `(entry or None, capture failed)`."""
        n = self._seen.pop(key, 0)
        if n < self._capture_after:  # eager, so one-off shapes never capture
            self._seen[key] = n + 1
            if len(self._seen) > self._seen_max:
                self._seen.popitem(last=False)
            return None, False
        if len(self.graphs) >= self._max and not self._evict():
            self._seen[key] = n
            return None, False
        try:
            entry = self._capture(inputs)
        except Exception:  # noqa: BLE001 - OOM / unsupported op during capture: this shape stays eager
            self._failed[key] = None
            if len(self._failed) > _FAILED:
                self._failed.popitem(last=False)
            return None, True
        self.graphs[key] = entry
        return entry, False

    def _capture(self, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.cuda.CUDAGraph, tuple[torch.Tensor, ...], Any]:
        device = inputs[0].device
        static = tuple(x.clone() for x in inputs)
        cur = torch.cuda.current_stream(device)
        side = capture_stream(device)
        graph = torch.cuda.CUDAGraph()
        with CAPTURE_LOCK, torch.cuda.stream(side):
            side.wait_stream(cur)
            self._forward(*static)  # warm-up outside the capture (lazy inits, workspaces)
            side.synchronize()
            graph.capture_begin(self._pool, capture_error_mode="thread_local")
            try:
                out = self._forward(*static)
            finally:
                graph.capture_end()
        cur.wait_stream(side)
        self._pool = graph.pool()
        return graph, static, out

    def _evict(self) -> bool:
        """Drop the least recently used graph if it is stale; False: the cache is busy, run eagerly."""
        old = min(self._used, key=self._used.__getitem__)
        if self._calls - self._used[old] < self._stale:
            return False
        if self._stream is not None:
            self._stream.synchronize()  # the last replay + copy-out are done: the buffers may go
        del self.graphs[old], self._used[old]
        if not self.graphs:
            self._pool = None
        return True

    def release(self) -> None:
        """Drop every graph and free the pool on the device (`torch.cuda.empty_cache`)."""
        with self._lock:
            if self._stream is not None:
                self._stream.synchronize()
            self.graphs.clear()
            self._used.clear()
            self._seen.clear()
            self._pool = None
        torch.cuda.empty_cache()  # a dropped graph pool is only freed by the next empty_cache


class SmallShapeGraphs:
    """A kernels forward that replays a `GraphCache` graph for small inputs and runs eagerly otherwise.

    Small means `batch x H x W` of the first input <= `pixels` (launch-bound: the host issues the kernels slower than
    the GPU runs them); larger inputs are GPU-bound, where a graph's pool would only cost memory. `enabled=False`
    (option `cuda_graphs = false`) keeps every call eager; `release()` frees the graphs and their pool.
    """

    def __init__(
        self,
        forward: Callable[..., torch.Tensor],
        *,
        pixels: int,
        graphs: int = 2,
        enabled: bool = True,
        capture_after: int = 1,
    ) -> None:
        """`forward(*tensors) -> Tensor`; at most `graphs` shapes keep a graph (see `GraphCache`)."""
        self.forward = _weak(forward)
        self.pixels = pixels
        self.cache = GraphCache(forward, graphs=graphs, capture_after=capture_after) if enabled else None

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        """`forward(*inputs)`, from a graph when the input is small, CUDA and its shape repeats."""
        x = inputs[0]
        if self.cache is not None and x.is_cuda and x.shape[0] * x.shape[-2] * x.shape[-1] <= self.pixels:
            key = (x.device, tuple((t.dtype, tuple(t.shape)) for t in inputs))
            out = self.cache.run(key, *inputs)
            if out is not None:
                return out
        return self.forward(*inputs)

    def release(self) -> None:
        """Free the graphs and their memory pool (no-op when graphs are off)."""
        if self.cache is not None:
            self.cache.release()
