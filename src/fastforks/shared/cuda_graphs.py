"""A small per-model cache of CUDA graphs for launch-bound forwards (one graph per input-shape key).

Conventions (as the DINOv3 backbone): a shape is captured after `capture_after` eager calls (at least 1, so one-off
shapes never capture; default 1: on its second call. All calls for a key within its first inference request run
eagerly, even when one request repeats the same shape internally. Models on `cudnn_hacc` plans capture on the 4th
call, so a stream of new sizes seen 1-3 times captures nothing. When the cache is full a
graph unused for `stale` graph-sized calls makes room, else the new shape runs eagerly; a capture failure (OOM,
unsupported op) is logged as a warning, marks the shape eager for good and frees what the capture allocated before the
capture ends (required by the `cudaMallocAsync` allocator, see `free_captured`); a capture (failed or not) empties the
allocator cache once, after the lock is released; calls hold a per-cache lock and wait for the
previous call's stream, so threads / streams never read each other's static buffers; `release()` frees the pool on the
device. Under `CAPTURE_LOCK`, the cache is emptied before a separate warm-up, when the capture stream cannot reuse
the caller's cached blocks, and before capture, when the private pool cannot reuse those blocks or the warm-up's.
Either set would otherwise stay reserved beside the next allocations (aliked-lightglue's capture call:
3874 -> 2612 -> 1494 MiB peak reserved).

With the native allocator, runners without capture-time probes can use their preceding eager requests to warm
the shape. They try capturing directly and retry with a checked warm-up if the device or library needs it.
Every warm-up and capture runs on one shared stream per device (`capture_stream`, under `CAPTURE_LOCK`), and
kernels that overlap work on side streams take them from `side_stream`: cuBLAS keeps a 32 MB workspace per stream
for the life of the process, so a new stream per capture or per loaded model grew the memory by 32 MB each time.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import gc
import inspect
import logging
import threading
import traceback
import warnings
import weakref
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from fastforks._core.kernels import checked, trim_cuda_graph_pool
from fastforks._core.lazy import lazy
from fastforks._core.upload import capture_uploads

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Hashable

    import torch
else:
    torch = lazy("torch")


log = logging.getLogger("fastforks")
_FAILED = 256  # fixed: shapes remembered as not capturable
_LARGER = 2  # fixed: a new shape this many times larger than every held graph's input frees the graphs

# serialises the warm-ups + captures on the shared capture stream (reentrant: a captured forward may hold it again)
CAPTURE_LOCK = threading.RLock()
_REQUEST: contextvars.ContextVar[object | None] = contextvars.ContextVar("cuda_graph_request", default=None)


@contextlib.contextmanager
def graph_request() -> Generator[None]:
    """Group the graph-cache calls made by a single inference request."""
    if _REQUEST.get() is not None:
        yield
        return
    token = _REQUEST.set(object())
    try:
        yield
    finally:
        _REQUEST.reset(token)


if TYPE_CHECKING:

    class _Graph(torch.cuda.CUDAGraph):
        uploads: list[torch.Tensor]


@functools.cache
def _graph_class() -> type[_Graph]:
    class _Graph(torch.cuda.CUDAGraph):
        """A CUDA graph that holds the pinned host buffers its captured uploads copy from (`capture_uploads`)."""

        uploads: list[torch.Tensor]

    return cast("Any", _Graph)  # defined at first use: importing this module does not import torch


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


def empty_graph_cache(device: torch.device, *, released: bool) -> None:
    """Empty the allocator cache and trim unused driver graph memory after releasing a graph."""
    torch.cuda.empty_cache()
    if released and torch.cuda.get_allocator_backend() == "cudaMallocAsync":
        # CUDA keeps memory nodes in a separate device pool, even after the graphs are destroyed.
        trim_cuda_graph_pool(_index(device))


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


def free_captured(e: BaseException) -> None:
    """Free, while the capture is still open, what a forward that raised during capture allocated.

    The tensors live on in the frames of the exception's traceback, the cuBLAS workspace in the handle pool. Freed after
    `capture_end`, they would be frees of captured allocations of a graph that never ran: the native allocator returns
    them to the graph's private pool, but `cudaMallocAsync` (ComfyUI's default allocator) calls `cudaFreeAsync` on
    memory that was never mapped, which fails in a tensor destructor and aborts the process. Freed now, they are frees
    in the graph, which is then discarded.
    """
    seen: set[int] = set()
    err: BaseException | None = e
    while err is not None and id(err) not in seen:
        seen.add(id(err))
        traceback.clear_frames(err.__traceback__)  # skips the frames still running (this one and its callers)
        err = err.__cause__ or err.__context__
    gc.collect()  # tensors in reference cycles of the cleared frames
    torch._C._cuda_clearCublasWorkspaces()  # noqa: SLF001 - no public API; a workspace allocated during the capture


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

    def __init__(  # noqa: PLR0913 - cache sizing, reuse and capture policy
        self,
        forward: Callable[..., Any],
        *,
        graphs: int = 2,
        stale: int = 64,
        seen: int = 64,
        capture_after: int = 1,
        share_inputs: tuple[int, ...] = (),
        capture_warmup: bool = True,
    ) -> None:
        """Capture after `capture_after` eager calls and retain at most `graphs` graphs.

        Input positions in `share_inputs` reuse matching static buffers across keys. All warm-ups, replays and
        output copies on this cache are ordered, including when successive calls use different CUDA streams.
        Disable `capture_warmup` only if the preceding eager calls prepare the runner for capture and replay does
        not depend on checks of the capture input. The async allocator always uses a warm-up.
        """
        self._share_inputs = share_inputs
        self._forward = _weak(forward)
        self._capture_after = max(1, capture_after)  # one-off shapes never capture
        self._capture_warmup = capture_warmup
        self._max, self._stale, self._seen_max = graphs, stale, seen
        # key -> (graph, static inputs, static output)
        self.graphs: dict[Hashable, tuple[torch.cuda.CUDAGraph, tuple[torch.Tensor, ...], Any]] = {}
        self._used: dict[Hashable, int] = {}
        self._seen: OrderedDict[Hashable, tuple[int, object | None]] = OrderedDict()  # eager calls, first request
        self._failed: OrderedDict[Hashable, None] = OrderedDict()
        self._calls = 0
        self._pool: Any = None
        self._pool_device: torch.device | None = None
        self._stream: torch.cuda.Stream | None = None
        self._lock = threading.Lock()

    def run(
        self,
        key: Hashable,
        *inputs: torch.Tensor,
        post: Callable[[Any], Any] | None = None,
        before_replay: Callable[[tuple[torch.Tensor, ...]], None] | None = None,
    ) -> Any:  # noqa: ANN401 - the output structure is chosen by the captured forward
        """The output (fresh tensors) from the graph of `key`, or None: the caller runs eagerly.

        `post(static output)` replaces the copy when it returns new tensors anyway (e.g. a dtype conversion).
        `before_replay(static inputs)` can reset persistent state after the preceding replay is ordered.
        """
        failed = captured = False
        released: torch.device | None = None
        result = None
        with self._lock:
            cur = torch.cuda.current_stream(inputs[0].device)
            if self._stream is not None and self._stream != cur:
                cur.wait_stream(self._stream)  # the previous call's replay + copy-out are ordered before ours
            self._stream = cur
            self._calls += 1
            entry = self.graphs.get(key)
            if entry is None and self.graphs and self._much_larger(inputs[0]):
                released = self._pool_device
                self._free_graphs()  # a GPU-bound shape: the small shapes' pools only cost memory next to it
            if entry is None and key not in self._failed:
                had_graphs = bool(self.graphs)
                entry, failed = self._add(key, inputs)
                captured = entry is not None
                if failed and had_graphs and not self.graphs:
                    released, self._pool_device = self._pool_device, None
            if entry is not None:
                self._used[key] = self._calls
                graph, static, out = entry
                for static_input, input_tensor in zip(static, inputs, strict=True):
                    if static_input is not input_tensor:
                        static_input.copy_(input_tensor)
                    del static_input
                if before_replay is not None:
                    before_replay(static)
                graph.replay()
                result = _copy(out) if post is None else post(out)
                # Give release() sole ownership of the cached tensors before it can take the lock.
                del entry, graph, static, out
        if failed or captured or released is not None:
            # outside the lock: the partial pool of a failed capture, or the blocks the eager calls and the capture
            # warm-up cached (the warm-up's on the capture stream, which the caller's stream never reuses): replays
            # allocate nothing, so these blocks only doubled the reserved memory (igev++ 1454 -> 720 MiB)
            empty_graph_cache(released or inputs[0].device, released=released is not None)
        return result

    def static_inputs(self, key: Hashable) -> tuple[torch.Tensor, ...] | None:
        """The static input buffers of `key`'s graph (None: no graph).

        A forward that writes a recurrent state into its own static inputs (in place, at the end) lets the caller pass
        these back on the next call: copying a tensor onto itself launches nothing, so the state never moves.
        """
        entry = self.graphs.get(key)
        return None if entry is None else entry[1]

    def defer_first_request(self, key: Hashable) -> bool:
        """Record a first-request eager call before the caller prepares graph-only inputs."""
        request = _REQUEST.get()
        if request is None:
            return False
        with self._lock:
            n, first = self._seen.get(key, (0, request))
            if key in self.graphs or first is not request:
                return False
            self._calls += 1
            self._remember(key, n + 1, first)
            return True

    def _remember(self, key: Hashable, calls: int, first: object | None) -> None:
        self._seen[key] = (calls, first)
        self._seen.move_to_end(key)
        if len(self._seen) > self._seen_max:
            self._seen.popitem(last=False)

    def _add(self, key: Hashable, inputs: tuple[torch.Tensor, ...]) -> tuple[Any, bool]:
        """Capture `key` after `capture_after` eager calls: `(entry or None, capture failed)`."""
        request = _REQUEST.get()
        n, first = self._seen.pop(key, (0, request))
        if n < self._capture_after or (request is not None and request is first):
            self._remember(key, n + 1, first)
            return None, False
        if len(self.graphs) >= self._max and not self._evict():
            self._seen[key] = (n, first)
            return None, False
        try:
            entry = self._capture(inputs)
        except Exception as e:  # noqa: BLE001 - OOM / unsupported op during capture: this shape stays eager
            log.warning("CUDA graph capture failed, this input shape runs eagerly: %s: %s", type(e).__name__, e)
            self._failed[key] = None
            if len(self._failed) > _FAILED:
                self._failed.popitem(last=False)
            return None, True
        self.graphs[key] = entry
        return entry, False

    def _static_input(self, i: int, x: torch.Tensor) -> torch.Tensor:
        if i in self._share_inputs:
            for _, static, _ in self.graphs.values():
                if i < len(static):
                    other = static[i]
                    if (other.device, other.dtype, other.shape, other.stride()) == (
                        x.device,
                        x.dtype,
                        x.shape,
                        x.stride(),
                    ):
                        return other.copy_(x)
        return x.clone()

    def _capture(self, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.cuda.CUDAGraph, tuple[torch.Tensor, ...], Any]:
        # With cudaMallocAsync, freeing live allocations after an invalidated capture can abort the process.
        if not self._capture_warmup and torch.cuda.get_allocator_backend() == "native":
            try:
                return self._capture_once(inputs, warmup=False)
            except Exception as e:  # noqa: BLE001 - a library may still need a forward on the capture stream
                log.debug("Retrying CUDA graph capture with a warm-up: %s: %s", type(e).__name__, e)
        return self._capture_once(inputs, warmup=True)

    def _capture_once(
        self, inputs: tuple[torch.Tensor, ...], *, warmup: bool
    ) -> tuple[torch.cuda.CUDAGraph, tuple[torch.Tensor, ...], Any]:
        device = inputs[0].device
        static = tuple(self._static_input(i, x) for i, x in enumerate(inputs))
        cur = torch.cuda.current_stream(device)
        side = capture_stream(device)
        graph = _graph_class()()
        with CAPTURE_LOCK, torch.cuda.stream(side):
            side.wait_stream(cur)
            try:
                if warmup:
                    # The warm-up cannot reuse the blocks cached by earlier requests on the caller's stream.
                    torch.cuda.empty_cache()
                    checked(self._forward, *static)  # lazy inits, workspaces, overflow checks
            finally:
                # Order workspace release after the caller's work, including any partially completed warm-up.
                side.synchronize()
            # Give each capture its own cuBLAS workspace. The async allocator warns when the workspace from an
            # earlier capture is freed here. Its allocation node restores it on the next replay.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Attempting uncaptured free of a captured allocation with address",
                    category=UserWarning,
                )
                torch._C._cuda_clearCublasWorkspaces()  # noqa: SLF001 - no public API
            torch.cuda.empty_cache()  # the warm-up's cached blocks: the capture's pool cannot reuse them
            graph.capture_begin(self._pool, capture_error_mode="thread_local")
            try:
                with capture_uploads() as graph.uploads:  # replays copy from these host buffers again
                    out = self._forward(*static)
            except BaseException as e:
                free_captured(e)
                raise
            finally:
                graph.capture_end()
        cur.wait_stream(side)
        self._pool = graph.pool()
        self._pool_device = device
        return graph, static, out

    def _much_larger(self, x: torch.Tensor) -> bool:
        """`x` has at least `_LARGER` x the elements of the first input of every held graph."""
        return all(x.numel() >= _LARGER * static[0].numel() for _, static, _ in self.graphs.values())

    def _free_graphs(self) -> None:
        """Drop every graph (the caller's `empty_cache` frees their pool, outside the lock); they capture again."""
        if self._stream is not None:
            self._stream.synchronize()  # the last replay + copy-out are done: the buffers may go
        self.graphs.clear()
        self._used.clear()
        self._seen.clear()
        self._pool = None
        self._pool_device = None

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
        """Drop every graph and return its unused pool memory to the device."""
        with self._lock:
            if self.graphs and self._stream is not None:
                self._stream.synchronize()
            self.graphs.clear()
            self._used.clear()
            self._seen.clear()
            self._pool = None
            device, self._pool_device = self._pool_device, None
        # a dropped graph pool is only freed by the next empty_cache; without a capture since the last release there is
        # none, and an empty_cache would only make the next eager call allocate again (PP-OCRv5 mobile 1080 x 1920,
        # releasing its recogniser graphs on every large page: 176 -> 163 ms, paired in one process)
        if device is not None:
            empty_graph_cache(device, released=True)


_Output = TypeVar("_Output")


class SmallShapeGraphs(Generic[_Output]):
    """A kernels forward that replays a `GraphCache` graph for small inputs and runs eagerly otherwise.

    Small means `batch x H x W` of the first input (or `size(*inputs)`) <= `pixels` (launch-bound: the host issues the
    kernels slower than the GPU runs them); larger inputs are GPU-bound, where a graph's pool would only cost memory:
    such an input first frees the graphs of earlier small shapes (they capture again when a shape repeats).
    `enabled=False` (option `cuda_graphs = false`) keeps every call eager; `release()` frees the graphs and their pool.
    """

    def __init__(  # noqa: PLR0913 - the cache policy
        self,
        forward: Callable[..., _Output],
        *,
        pixels: int,
        graphs: int = 2,
        enabled: bool = True,
        capture_after: int = 1,
        size: Callable[..., int] | None = None,
        one_shape: bool = False,
        capture_warmup: bool = True,
    ) -> None:
        """`forward(*tensors) -> Tensor`; at most `graphs` shapes keep a graph (see `GraphCache`).

        `size(*inputs)` replaces `batch x H x W` of the first input as the size compared with `pixels` (a forward
        that resizes its input inside: the network size).
        `one_shape` captures only consecutive requests of one signature and releases it before running another.
        Use it when the forward replaces size-dependent buffers a graph may read or to limit overlapping allocations.
        `capture_warmup` controls whether the native allocator runs a separate forward before the first capture attempt.
        """
        self.forward = _weak(forward)
        self.pixels = pixels
        self.size = _weak(size) if size is not None else None
        self.one_shape = one_shape
        self._last_key: Hashable | None = None
        self.cache = (
            GraphCache(
                forward, graphs=1 if one_shape else graphs, capture_after=capture_after, capture_warmup=capture_warmup
            )
            if enabled
            else None
        )

    def __call__(self, *inputs: torch.Tensor) -> _Output:
        """`forward(*inputs)`, from a graph when the input is small, CUDA and its shape repeats."""
        x = inputs[0]
        cache = self.cache
        if cache is None or not x.is_cuda:
            return self.forward(*inputs)
        small = (self.size(*inputs) if self.size else x.shape[0] * x.shape[-2] * x.shape[-1]) <= self.pixels
        key = None
        if small:
            if self.one_shape:
                key = (x.device, tuple((t.dtype, t.shape, t.stride()) for t in inputs))
            else:
                key = (x.device, tuple((t.dtype, tuple(t.shape)) for t in inputs))
        if self.one_shape and self._last_key != key:
            if self._last_key is not None:
                cache.release()
            self._last_key = key
        elif key is None and cache.graphs:
            # A large eager forward can reuse memory freed from a held small graph.
            cache.release()
        if key is not None:
            out = cache.run(key, *inputs)
            if out is not None:
                return out
        return self.forward(*inputs)

    def release(self) -> None:
        """Free the graphs and their memory pool (no-op when graphs are off)."""
        if self.cache is not None:
            self.cache.release()


@functools.cache
def _captured_module() -> type[CapturedModule]:
    class CapturedModule(torch.nn.Module):
        """A dense inference submodule with a bounded per-shape CUDA graph cache."""

        def __init__(self, network: torch.nn.Module, *, pixels: int) -> None:
            """Keep the underlying module registered so its parameters remain visible."""
            super().__init__()
            self.network = network
            self.graphs = SmallShapeGraphs(network, pixels=pixels)

        def forward(self, *inputs: torch.Tensor) -> Any:  # noqa: ANN401 - preserve tensor or tuple module outputs
            """Replay the complete dense output for a recurring shape."""
            return self.graphs(*inputs)

    return cast("Any", CapturedModule)


if TYPE_CHECKING:

    class CapturedModule(torch.nn.Module):  # noqa: D101 - the class `_captured_module` makes
        network: torch.nn.Module
        graphs: SmallShapeGraphs[Any]

        def __init__(self, network: torch.nn.Module, *, pixels: int) -> None: ...  # noqa: D107

else:

    def __getattr__(name: str) -> object:
        """`CapturedModule` (it subclasses `torch.nn.Module`: made at its first use)."""
        if name == "CapturedModule":
            return _captured_module()
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
