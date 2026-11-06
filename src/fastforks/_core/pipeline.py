"""Many inputs through one model: reading, GPU work and writing overlap in threads; results come in input order.

Threads (all daemon):

- a feeder iterates the source (so a generator runs in this thread, up to `prefetch` items ahead), takes a ticket
  per item and hands the item to the readers. The second item goes to the readers once the first one is decoded, the
  others once the first one's GPU work is queued: their decoding would compete with the first item's decoding and
  its (first, CPU-heavy) model call for the cores and delay the first result;
- readers run `read(item)` (decode; CPU only);
- one GPU thread loads the CUDA math libraries while the first item decodes, then runs `run(data, copies)` for one
  item after the other: every CUDA call of the pipeline happens in this thread. Compute stays on the default stream,
  as in a plain loop; uploads go through a copy stream into a fixed device ring and downloads into page-locked host
  slots, ordered against the compute by events. The first item takes the direct path of a loop (`.to(device)`,
  `.cpu()`, no slot set-up on its critical path);
- a collector waits for each item's download event (the GPU thread may block inside the next item's model call) and
  hands the item on: it frees the item's device memory, the slots stay with the item until its writer is done;
- writers run `finish(payload)` once an item's downloads are complete (copy out of the host slot, save a file). Their
  pool is separate from `io._pool()`, which `write_tiff` itself waits on.

`prefetch` tickets bound everything between the feeder and the consumer (decoded inputs, device and host slots,
finished results). The consumer only takes finished futures from a FIFO deque, which keeps the input order and raises
an item's error at its position.

Several pipelines (and calls of other threads) may run at once: a model call takes the process-wide CUDA call lock
(`Model._session`), so each call's kernel-library accounting (`kernels.begin_call`) and global torch flags stay its
own; the transfers and the geometry around the calls set no global state.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import queue
import sys
import threading
import time
import weakref
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast

import numpy as np
import torch

from fastforks._core.errors import FastforksError, InputError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

T = TypeVar("T")
OnError = Literal["raise", "skip"]

_POLL = 0.05  # s: blocking waits wake up this often to notice close()
_SPIN = 0.001  # s: wait step of the GPU thread while every page-locked slot is in use
_BUDGET = 1 << 30  # bytes of frames in flight for the default prefetch (capped to RAM / 8)
_MAX_PREFETCH = 8
_PAGE = 4096
_DEVICE_SLOTS = 3  # device ring depth in items: uploads of item i wait for the compute of item i - 3


def default_workers() -> int:
    """Default reader / writer threads: half the cores, at most 4."""
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def _ram_bytes() -> int | None:
    try:
        if sys.platform == "win32":
            import ctypes  # noqa: PLC0415 - Windows only

            class _Status(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("load", ctypes.c_ulong),
                    ("total", ctypes.c_ulonglong),
                    ("rest", ctypes.c_ulonglong * 6),
                ]

            status = _Status(ctypes.sizeof(_Status))
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.total)
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None


def default_prefetch(item_bytes: int) -> int:
    """Items in flight: as many as fit in 1 GiB (at most RAM / 8), between 2 and 8.

    Sized by the first item: much larger later items (mixed sizes) can exceed that budget.
    """
    ram = _ram_bytes()
    budget = min(_BUDGET, ram // 8) if ram else _BUDGET
    return min(_MAX_PREFETCH, max(2, budget // max(1, item_bytes)))


@dataclasses.dataclass(frozen=True)
class Done(Generic[T]):
    """Returned by `read` for an item that needs no GPU work (an existing output): `value` is its result."""

    value: T


@dataclasses.dataclass
class Stats:
    """Timings of a run, in seconds (filled while it runs).

    Attributes:
        setup_s: GPU-thread setup before the first item (e.g. rectification maps), overlapping the first decode.
        first_result_s: From the start to the first result handed out.
        input_wait_s: Time the GPU thread waited for decoded inputs (the readers were the bottleneck).
        total_s: From the start to the end of the iteration.
        items: Results handed out.
    """

    setup_s: float = 0.0
    first_result_s: float | None = None
    input_wait_s: float = 0.0
    total_s: float | None = None
    items: int = 0


class _Host:
    """A downloaded map: a numpy view of a page-locked slot (or a plain array), valid once the item's event is done."""

    __slots__ = ("_array", "_release")

    def __init__(self, array: np.ndarray, release: Callable[[], None] | None) -> None:
        self._array = array
        self._release = release

    def view(self) -> np.ndarray:
        """The map, valid until `release` (for a writer that reads it once)."""
        return self._array

    def release(self) -> None:
        """Give the slot back (idempotent)."""
        if self._release is not None:
            release, self._release = self._release, None
            release()

    def take(self) -> np.ndarray:
        """The map as an array of its own, and the slot released."""
        if self._release is None:
            return self._array
        out = self._array.copy()
        self.release()
        return out


class PinnedRing:
    """Page-locked host slots for downloads; created, grown and freed in the GPU thread only.

    Slots are numpy buffers registered with CUDA (`cudaHostRegister`: no power-of-two rounding and ~4x cheaper than
    a fresh `pin_memory` block), allocated lazily up to `n_slots` and replaced by a larger one when a bigger map
    comes. `release` may be called from any thread; the slot is reusable once the GPU thread has collected it.
    """

    def __init__(self, n_slots: int, *, register: Callable[[np.ndarray], Any] | None = None) -> None:
        self.n_slots = n_slots
        self._free: list[np.ndarray] = []
        self._count = 0
        self._returned: queue.SimpleQueue[np.ndarray] = queue.SimpleQueue()
        self._pinned: dict[int, Any] = {}  # buffer id -> what unregisters it
        self._register = register or _register
        self.peak_bytes = 0
        self._bytes = 0

    def collect(self) -> None:
        """Move the slots released by other threads back to the free list."""
        with contextlib.suppress(queue.Empty):
            while True:
                self._free.append(self._returned.get_nowait())

    def try_acquire(self, nbytes: int) -> np.ndarray | None:
        """A free slot of at least `nbytes` (a free smaller one is replaced), a new one, or None if all are in use."""
        self.collect()
        for i, buf in enumerate(self._free):
            if buf.nbytes >= nbytes:
                return self._free.pop(i)
        if self._free:
            self._drop(self._free.pop())
        elif self._count >= self.n_slots:
            return None
        return self._new(nbytes)

    def release(self, buf: np.ndarray) -> None:
        """Return a slot (any thread)."""
        self._returned.put(buf)

    def _new(self, nbytes: int) -> np.ndarray:
        raw = np.empty(nbytes + _PAGE, np.uint8)
        start = -raw.ctypes.data % _PAGE
        buf = raw[start : start + nbytes]
        self._pinned[id(buf)] = self._register(buf)
        self._count += 1
        self._bytes += nbytes
        self.peak_bytes = max(self.peak_bytes, self._bytes)
        return buf

    def _drop(self, buf: np.ndarray) -> None:
        handle = self._pinned.pop(id(buf))
        if callable(handle):
            handle()
        self._count -= 1
        self._bytes -= buf.nbytes

    def close(self) -> None:
        """Unregister every slot, also those of dropped items (all downloads complete, no writer running)."""
        self.collect()
        self._free.clear()
        for handle in self._pinned.values():
            if callable(handle):
                handle()
        self._pinned.clear()
        self._count = self._bytes = 0


def _register(buf: np.ndarray) -> Callable[[], None] | torch.Tensor:
    """Page-lock `buf` in place; returns its unregister call (or a pinned twin's owner when registering is refused)."""
    cudart = torch.cuda.cudart()
    if int(cudart.cudaHostRegister(buf.ctypes.data, buf.nbytes, 0)) == 0:
        ptr = buf.ctypes.data
        return lambda: cudart.cudaHostUnregister(ptr)
    msg = "cudaHostRegister failed"
    raise _RegisterError(msg)


class _RegisterError(RuntimeError):
    pass


class Copies:
    """Transfers of the GPU thread, given to `run`.

    `overlap=False` (CPU, or the ablation without copy streams) and `direct` (the first item) upload with
    `.to(device)` and download with `.cpu()` on the default stream, as a plain loop does. `queued` is called when a
    download is about to start (the item's GPU work is queued). Each item ends with `end_item` (its downloads go on
    to the writers) or, when `run` failed, `drop_item` (their slots come back once the copies are done).
    """

    def __init__(
        self,
        device: torch.device,
        *,
        overlap: bool,
        slots: int,
        wait_slot: Callable[[], None],
        queued: Callable[[], None] = lambda: None,
    ) -> None:
        self.device = device
        self.overlap = overlap and device.type == "cuda"
        self.direct = False
        self._wait_slot = wait_slot
        self._queued = queued
        self.ring = PinnedRing(slots)
        self._dev: list[torch.Tensor | None] = [None] * (2 * _DEVICE_SLOTS)
        self._dev_free: list[torch.cuda.Event | None] = [None] * (2 * _DEVICE_SLOTS)
        self._next = 0
        self._used: list[int] = []
        self._keep: list[torch.Tensor] = []
        self._hosts: list[_Host] = []  # page-locked downloads of the current item
        self._fresh = False
        self._stream = torch.cuda.Stream(device) if self.overlap else None
        self._pinned_ok = True

    def upload(self, t: torch.Tensor) -> torch.Tensor:
        """`t` on the device, same shape, dtype and strides (a fixed device slot when it comes from the host)."""
        if t.device == self.device or not self.overlap or self.direct:
            return t.to(self.device, non_blocking=True)
        if not (t.is_contiguous() or t.movedim(-3, -1).is_contiguous()):
            t = t.contiguous()
        nbytes = t.numel() * t.element_size()
        k = self._next
        if k in self._used:  # more uploads in one item than the ring holds
            return t.to(self.device, non_blocking=True)
        self._next = (k + 1) % len(self._dev)
        buf, free = self._dev[k], self._dev_free[k]
        compute = torch.cuda.current_stream(self.device)
        stream = self._stream
        assert stream is not None  # noqa: S101 - overlap implies a copy stream
        if buf is None or buf.numel() < nbytes:
            buf = self._dev[k] = torch.empty(nbytes, dtype=torch.uint8, device=self.device)
            stream.wait_stream(compute)  # fresh memory may be a block the compute stream still uses
        elif free is not None:
            stream.wait_event(free)
        dst = buf[:nbytes].view(t.dtype).as_strided(t.shape, t.stride())
        with torch.cuda.stream(stream):
            dst.copy_(t, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(stream)
        compute.wait_event(ready)
        self._used.append(k)
        return dst

    def download(self, t: torch.Tensor) -> _Host:
        """Start copying `t` (computed on the default stream) to the host; read it after the item's event."""
        self._queued()
        if t.device.type == "cpu":
            return _Host(t.numpy(), None)
        if not self.overlap or self.direct or not self._pinned_ok:
            return _Host(t.cpu().numpy(), None)
        t = t.contiguous()
        nbytes = t.numel() * t.element_size()
        try:
            buf = self.ring.try_acquire(nbytes)
            while buf is None:
                self._wait_slot()
                buf = self.ring.try_acquire(nbytes)
        except _RegisterError:  # no page-locked memory here (e.g. WSL limits): synchronous downloads
            self._pinned_ok = False
            return _Host(t.cpu().numpy(), None)
        host = buf[:nbytes].view(np.dtype(str(t.dtype).removeprefix("torch."))).reshape(t.shape)
        stream = self._stream
        assert stream is not None  # noqa: S101 - overlap implies a copy stream
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            torch.from_numpy(host).copy_(t, non_blocking=True)
        self._keep.append(t)  # the compute stream must not reuse its memory before the copy has read it
        ring = self.ring
        out = _Host(host, lambda: ring.release(buf))
        self._hosts.append(out)
        return out

    def end_item(self) -> tuple[torch.cuda.Event | None, list[torch.Tensor]]:
        """Close the current item: its device slots become reusable after its compute; returns its download event."""
        keep, self._keep = self._keep, []
        self._hosts = []
        if not self.overlap or self.direct:
            return None, keep
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(self.device))
        for k in self._used:
            self._dev_free[k] = done
        self._used = []
        copied = torch.cuda.Event(blocking=True)  # the collector thread sleeps on it instead of spinning
        assert self._stream is not None  # noqa: S101
        copied.record(self._stream)
        return copied, keep

    def drop_item(self) -> None:
        """Close the current item after `run` raised: wait for its downloads, then return their page-locked slots."""
        hosts = self._hosts
        event, _ = self.end_item()
        if event is not None:
            event.synchronize()
        for host in hosts:
            host.release()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.synchronize()
        self.ring.close()


@dataclasses.dataclass
class _Record:
    index: int
    item: Any
    decoded: Future[Any] | None
    out: Future[Any]


_Pending = tuple[_Record, Any, "torch.cuda.Event | None", list[torch.Tensor]]  # record, payload, download event


class _End:
    pass


_END = _End()


class _Engine:
    """State shared by the threads; the user-facing `Pipeline` only holds it (so an abandoned iterator is collected)."""

    def __init__(  # noqa: PLR0913 - the parts of a pipeline
        self,
        source: Iterable[Any],
        *,
        read: Callable[[Any], Any],
        run: Callable[[Any, Copies], Any],
        finish: Callable[[Any], Any],
        on_skip: Callable[[int, Any, Exception], Any],
        describe: Callable[[int, Any], str],
        setup: Callable[[], None] | None,
        device: torch.device,
        readers: int,
        writers: int,
        prefetch: int,
        on_error: OnError,
        overlap: bool,
    ) -> None:
        self.source, self.read, self.run, self.finish = source, read, run, finish
        self.on_skip, self.describe, self.setup = on_skip, describe, setup
        self.device, self.on_error = device, on_error
        self.overlap = overlap
        self.prefetch = prefetch
        self.tickets = threading.Semaphore(prefetch)
        self.outs: deque[Future[Any]] = deque()
        self.cv = threading.Condition()
        self.fed_all = False
        self.closing = threading.Event()
        self.dropped = False  # closed by the consumer: results not yet handed out are dropped
        self.fatal: BaseException | None = None
        self.gpu_q: queue.Queue[_Record | _End] = queue.Queue()
        self.collect_q: queue.SimpleQueue[_Pending | _End] = queue.SimpleQueue()
        self.direct = True
        self.first_queued = threading.Event()  # the first item's GPU work is queued (or it needs none)
        self.first_decoded: Future[Any] | None = None
        self.submit_lock = threading.Lock()
        self.readers_closed = False
        self.writes: set[Future[Any]] = set()  # submitted writes not yet done
        self.readers = ThreadPoolExecutor(readers, thread_name_prefix="fastforks-read") if readers else None
        self.writers = ThreadPoolExecutor(writers, thread_name_prefix="fastforks-write") if writers else None
        self.stats = Stats()
        self.autocast = (
            (torch.is_autocast_enabled(device.type), torch.get_autocast_dtype(device.type))
            if device.type in ("cuda", "cpu")
            else (False, None)
        )
        self.start = time.perf_counter()
        self.copies: Copies | None = None
        self.threads = [
            threading.Thread(target=self._feed, name="fastforks-feed", daemon=True),
            threading.Thread(target=self._gpu, name="fastforks-gpu", daemon=True),
        ]
        self.collector = threading.Thread(target=self._collect, name="fastforks-collect", daemon=True)
        self.collector.start()  # joined by the GPU thread
        for t in self.threads:
            t.start()

    # feeder ----------------------------------------------------------------------------------------------------
    def _take_ticket(self) -> bool:
        while not self.closing.is_set():
            if self.tickets.acquire(timeout=_POLL):
                return True
        return False

    def _feed(self) -> None:
        try:
            it = iter(self.source)
            index = 0
            while self._take_ticket():
                out: Future[Any] = Future()
                try:
                    item = next(it)
                except StopIteration:
                    self.tickets.release()
                    return
                except Exception as e:  # noqa: BLE001 - the source's error surfaces at its position
                    e.add_note(f"while reading input {index} of run_many")
                    out.set_exception(e)
                    self._append(out)
                    return
                ready = {1: self._first_decoded, 2: self.first_queued.is_set}.get(index)
                if ready is not None and self.readers and not self._wait_until(ready):
                    self.tickets.release()
                    return
                with self.submit_lock:  # the GPU thread may have shut the readers down meanwhile
                    if self.readers_closed:
                        self.tickets.release()
                        return
                    decoded = self.readers.submit(self.read, item) if self.readers else None
                if index == 0:
                    self.first_decoded = decoded
                self._append(out)
                self.gpu_q.put(_Record(index, item, decoded, out))
                index += 1
        finally:
            self.gpu_q.put(_END)
            with self.cv:
                self.fed_all = True
                self.cv.notify_all()

    def _first_decoded(self) -> bool:
        return self.first_decoded is None or self.first_decoded.done()

    def _wait_until(self, ready: Callable[[], bool]) -> bool:
        while not ready():
            if self.closing.is_set():
                return False
            time.sleep(_SPIN)
        return True

    def _append(self, out: Future[Any]) -> None:
        with self.cv:
            self.outs.append(out)
            self.cv.notify_all()

    # GPU thread ------------------------------------------------------------------------------------------------
    def _gpu(self) -> None:
        guard = torch.cuda.device(self.device) if self.device.type == "cuda" else contextlib.nullcontext()
        enabled, dtype = self.autocast
        amp = torch.autocast(self.device.type, dtype) if enabled else contextlib.nullcontext()
        try:
            with guard, amp, torch.inference_mode():
                self.copies = Copies(
                    self.device,
                    overlap=self.overlap,
                    slots=self.prefetch + 1,
                    wait_slot=self._wait_slot,
                    queued=self.first_queued.set,
                )
                t0 = time.perf_counter()
                if self.device.type == "cuda":
                    _init_libraries(self.device)
                if self.setup is not None:
                    self.setup()
                self.stats.setup_s = time.perf_counter() - t0
                while not self.closing.is_set():
                    rec = self._next_record()
                    if rec is None:
                        break
                    try:
                        self._process(rec)
                    finally:
                        self.first_queued.set()
        except PipelineClosedError:
            pass
        except BaseException as e:  # noqa: BLE001 - a CUDA / internal error ends the run; raised to the consumer
            self.fatal = e
            self.closing.set()
        finally:
            self._shutdown_gpu()

    def _shutdown_gpu(self) -> None:
        self.first_queued.set()
        self.collect_q.put(_END)
        self.collector.join()  # every download is complete (and handed to the writers) after this
        with self.submit_lock:
            self.readers_closed = True
        if self.readers is not None:
            self.readers.shutdown(wait=True, cancel_futures=True)
        if self.writers is not None:
            self.writers.shutdown(wait=True, cancel_futures=self.dropped)  # queued writes of a closed run never start
        if self.copies is not None:
            with contextlib.suppress(BaseException):
                self.copies.close()
        while True:  # records the GPU thread never took
            try:
                rec = self.gpu_q.get_nowait()
            except queue.Empty:
                break
            if isinstance(rec, _Record):
                self._fail_out(rec.out)
        with self.cv:
            self.cv.notify_all()

    def _fail_out(self, out: Future[Any]) -> None:
        if not out.done():
            out.set_exception(self.fatal or PipelineClosedError())

    def _wait_slot(self) -> None:
        """Wait step of the GPU thread while every page-locked slot is in use (writers release them)."""
        if self.closing.is_set():
            raise PipelineClosedError
        time.sleep(_SPIN)

    def _next_record(self) -> _Record | None:
        while True:
            try:
                rec = self.gpu_q.get(timeout=_POLL)
            except queue.Empty:
                if self.closing.is_set():
                    return None
                continue
            return None if isinstance(rec, _End) else rec

    def _wait_decoded(self, rec: _Record) -> object:
        t0 = time.perf_counter()
        if rec.decoded is None:
            return self.read(rec.item)
        while not rec.decoded.done():
            wait([rec.decoded], timeout=_POLL)
            if self.closing.is_set():
                raise PipelineClosedError
        if rec.index > 0:
            self.stats.input_wait_s += time.perf_counter() - t0
        return rec.decoded.result()

    def _process(self, rec: _Record) -> None:
        try:
            data = self._wait_decoded(rec)
        except PipelineClosedError:
            raise
        except Exception as e:  # noqa: BLE001 - decode errors go to the item
            self._item_error(rec, e)
            return
        if isinstance(data, Done):
            rec.out.set_result(data.value)
            return
        copies = self.copies
        assert copies is not None  # noqa: S101
        copies.direct, self.direct = self.direct, False  # the first item: no slot set-up and no collector hop
        try:
            payload = self.run(data, copies)
        except (InputError, OSError) as e:
            copies.drop_item()
            self._item_error(rec, e)
            return
        except BaseException as e:
            self._name(rec, e)
            raise
        event, keep = copies.end_item()
        if event is None:  # synchronous copies: done
            self._dispatch(rec, payload)
        else:
            self.collect_q.put((rec, payload, event, keep))

    def _collect(self) -> None:
        """Collector thread: waits for each item's downloads in order and hands it on.

        Not done in the GPU thread, which may block inside the next item's model call (a host sync) for the whole
        compute of that item: the result would come out an item late.
        """
        while True:
            entry = self.collect_q.get()
            if isinstance(entry, _End):
                return
            rec, payload, event, keep = entry
            copied = False
            try:
                assert event is not None  # noqa: S101 - only overlapped items come here
                event.synchronize()
                copied = True
            except BaseException as e:  # noqa: BLE001 - a CUDA error: the run ends
                if self.fatal is None:
                    self.fatal = e
                self.closing.set()
            del keep  # the downloads are complete: the compute stream may reuse those blocks
            if self.closing.is_set() and (self.dropped or not copied):  # an error of a later item: deliver this one
                self._fail_out(rec.out)
                continue
            self._dispatch(rec, payload)  # the writers shut down only after this thread ends

    def _dispatch(self, rec: _Record, payload: object) -> None:
        if self.writers is None:
            self._finish(rec, payload)
            return
        future = self.writers.submit(self._finish, rec, payload)
        self.writes.add(future)
        future.add_done_callback(lambda f: self._written(f, rec))

    def _written(self, future: Future[Any], rec: _Record) -> None:
        self.writes.discard(future)
        if future.cancelled():  # a write dropped by close() never started
            self._fail_out(rec.out)

    def _finish(self, rec: _Record, payload: object) -> None:
        try:
            rec.out.set_result(self.finish(payload))
        except (InputError, OSError) as e:
            self._item_error(rec, e)
        except BaseException as e:  # noqa: BLE001 - surfaces at the item's position
            self._name(rec, e)
            rec.out.set_exception(e)

    def _name(self, rec: _Record, e: BaseException) -> None:
        e.add_note(f"while processing {self.describe(rec.index, rec.item)}")

    def _item_error(self, rec: _Record, e: Exception) -> None:
        if self.on_error == "skip" and isinstance(e, (InputError, OSError)):
            rec.out.set_result(self.on_skip(rec.index, rec.item, e))
            return
        if isinstance(e, InputError):
            named = InputError(f"{self.describe(rec.index, rec.item)}: {e}")
            named.__cause__ = e
            rec.out.set_exception(named)
            return
        self._name(rec, e)
        rec.out.set_exception(e)

    # consumer --------------------------------------------------------------------------------------------------
    def next(self) -> object:
        with self.cv:
            while not self.outs:
                if self.fed_all or self.fatal is not None or not self.threads[1].is_alive():
                    if self.fatal is not None:
                        raise self.fatal
                    raise StopIteration
                self.cv.wait(_POLL)
            out = self.outs.popleft()
        try:
            while True:
                done, _ = wait([out], timeout=_POLL)
                if done:
                    break
                if not self.threads[1].is_alive():
                    self._fail_out(out)
            return out.result()
        finally:
            self.tickets.release()

    def close(self) -> None:
        self.dropped = True
        self.closing.set()
        for write in list(self.writes):
            write.cancel()  # only the writes in progress finish
        if threading.current_thread().name.startswith("fastforks-"):
            return  # a pipeline thread (e.g. a garbage collection running the finalizer there): it must not join itself
        for t in self.threads:
            while t.is_alive():
                t.join(_POLL)
                if t.name == "fastforks-feed" and not self.threads[1].is_alive():
                    break  # the feeder may be blocked inside the user's generator; it is a daemon


def _init_libraries(device: torch.device) -> None:
    """Load cuDNN and cuBLAS and create this thread's handles while the readers decode the first item.

    About 0.1 s in a fresh process, which a loop pays inside its first call.
    """
    x = torch.zeros(1, 1, 8, 8, device=device)
    torch.nn.functional.conv2d(x, x[..., :3, :3])
    x[0, 0].half() @ x[0, 0].half()


class PipelineClosedError(FastforksError, RuntimeError):
    """Raised when a closed pipeline is iterated (a `FastforksError`, so it is caught with the library's)."""

    def __init__(self) -> None:
        super().__init__("the pipeline was closed")


class Pipeline(Generic[T]):
    """Iterator over the results of a pipelined run, in input order; see the module docstring.

    Closing it (`close()`, leaving a `with` block, `break` followed by garbage collection) stops the threads after
    the writes in progress; results not yet taken are dropped.

    Attributes:
        stats: Timings, filled while it runs (`Stats`).
    """

    def __init__(self, source: Iterable[object], **parts: Any) -> None:  # noqa: ANN401 - _Engine keywords
        self._engine = _Engine(source, **parts)
        self.stats = self._engine.stats
        self._finalizer = weakref.finalize(self, self._engine.close)
        self._exhausted = False
        self._closed = False  # closed by the consumer before the end

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        engine = self._engine
        if self._closed:
            raise PipelineClosedError
        try:
            value = engine.next()
        except StopIteration:
            self._exhausted = True
            if self.stats.total_s is None:
                self.stats.total_s = time.perf_counter() - engine.start
            self.close()
            raise
        except BaseException:
            self.close()
            raise
        self.stats.items += 1
        if self.stats.first_result_s is None:
            self.stats.first_result_s = time.perf_counter() - engine.start
        return cast("T", value)

    def close(self) -> None:
        """Stop the threads (writes in progress complete); idempotent. Sets `stats.total_s` if still unset."""
        if self.stats.total_s is None:
            self.stats.total_s = time.perf_counter() - self._engine.start
        self._closed = self._closed or not self._exhausted
        self._finalizer()
        if self._closed:
            self._engine.outs.clear()  # untaken results are dropped

    def __enter__(self) -> Pipeline[T]:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
