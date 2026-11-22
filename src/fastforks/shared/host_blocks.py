"""Streamed transformer blocks: one page-locked host buffer, uploaded one block ahead into two device buffers.

All blocks live in the host buffer; each is uploaded into one of two fixed device buffers (ping-pong) on a side stream
while the current block computes.

Each block's tensors are packed contiguously (one copy per block). The host buffer is ordinary memory registered with
`cudaHostRegister`: torch's pinned allocator rounds each allocation up to a power of two (8.4 GB of blocks would pin
~13-15 GB). The two device buffers are allocated once and reused across blocks and calls, so the allocator's reserved
memory stays at the first call's.
"""

from __future__ import annotations

import ctypes
import functools
import itertools
import logging
import sys
import threading
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

log = logging.getLogger("fastforks")
ALIGN = 256  # bytes: start of every tensor in the packed buffers


def _up(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


def available_host_bytes() -> int | None:
    """Host RAM available without swapping (Linux `MemAvailable`, Windows `ullAvailPhys`); None where unreadable."""
    if sys.platform == "win32":
        return _windows_available()
    try:
        with open("/proc/meminfo", encoding="ascii") as f:  # noqa: PTH123
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def _windows_available() -> int | None:
    class _Status(ctypes.Structure):
        _fields_: ClassVar = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            *((name, ctypes.c_ulonglong) for name in ("total", "avail", "total_pf", "avail_pf", "total_v", "avail_v")),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = _Status()
    status.dwLength = ctypes.sizeof(_Status)
    kernel32 = getattr(ctypes, "windll").kernel32  # noqa: B009 - ctypes.windll exists only on Windows
    ok = kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return int(status.avail) if ok else None


def require_host_ram(need: int, what: str) -> None:
    """Fail with a clear error when the free host RAM is clearly too small for `need` bytes of page-locked memory.

    Page-locked memory cannot be swapped: without the check the process is killed for lack of memory. Clearly too small:
    less than `need` plus 1 GiB for the rest of the process is available.
    """
    free = available_host_bytes()
    if free is not None and free < need + 2**30:
        msg = (
            f"{what} keeps its transformer blocks in {need / 2**30:.1f} GiB of page-locked host memory, but only "
            f"{free / 2**30:.1f} GiB of host RAM are available: free memory (close other programs), pass "
            "stream=False (the blocks then need the device memory instead) or use a machine with more RAM"
        )
        raise MemoryError(msg)


def _unregister(ptr: int) -> None:
    torch.cuda.cudart().cudaHostUnregister(ptr)


def _register_prefix(ptr: int, ends: list[int]) -> int:
    """Register `[ptr, ptr + ends[k])` for the largest `k` the OS allows (one registration); its size, 0 if none.

    `ends`: the block ends, increasing. The whole buffer is tried first; the OS limits page-locked memory (Windows: to
    about half of the RAM), so on failure the longest block prefix that registers is found by bisection.
    """
    cudart = torch.cuda.cudart()

    def register(n: int) -> bool:
        if int(cudart.cudaHostRegister(ptr, n, 0)) == 0:
            return True
        _clear_last_error()  # else the next CUDA call checked by torch (an allocation) raises this out-of-memory
        return False

    if register(ends[-1]):
        return ends[-1]
    lo, hi = 0, len(ends)  # block counts: a prefix of `hi` blocks fails, one of `lo` fits (0: nothing)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if register(ends[mid - 1]):
            cudart.cudaHostUnregister(ptr)
            lo = mid
        else:
            hi = mid
    if lo == 0 or not register(ends[lo - 1]):
        return 0
    return ends[lo - 1]


@functools.cache
def _runtime() -> ctypes.CDLL | None:
    """The CUDA runtime library torch loaded (the same instance, hence the same error state); None if not found."""
    if sys.platform == "win32":
        found = sorted(str(p) for p in Path(torch.__file__).parent.glob("lib/cudart64_*.dll"))
    else:
        try:
            maps = Path("/proc/self/maps").read_text(encoding="utf-8")
        except OSError:
            maps = ""
        found = sorted({line.split()[-1] for line in maps.splitlines() if "/libcudart.so" in line})
    return ctypes.CDLL(found[0]) if len(found) == 1 else None


def _clear_last_error() -> None:
    """Reset the CUDA runtime's last error (`cudaGetLastError`), which a failed `cudaHostRegister` leaves set."""
    lib = _runtime()
    if lib is None:
        msg = "cudaHostRegister failed, and the CUDA runtime library to reset its error was not found"
        raise RuntimeError(msg)
    lib.cudaGetLastError()


class HostBlocks:
    """The blocks' tensors, packed per block in one registered host buffer; `run` streams them to the device."""

    def __init__(
        self, blocks: list[dict[str, tuple[torch.dtype, torch.Size]]], fill: Callable[[str, torch.Tensor], None]
    ) -> None:
        """`blocks`: per block the checkpoint key -> (dtype, shape); `fill(key, dst)` copies that tensor into `dst`."""
        self._lock = threading.Lock()
        self.layout: list[list[tuple[str, torch.dtype, torch.Size, int, int]]] = []
        self.starts: list[int] = []
        pos = 0
        for spec in blocks:
            self.starts.append(pos)
            rows, off = [], 0
            for k, (dt, shape) in spec.items():
                n = shape.numel() * dt.itemsize
                rows.append((k, dt, shape, off, n))
                off = _up(off + n)
            self.layout.append(rows)
            pos += off
        self.sizes = [s - a for a, s in zip(self.starts, [*self.starts[1:], pos], strict=True)]
        require_host_ram(pos, "this model")
        self.buf = torch.empty(pos, dtype=torch.uint8)
        for i, rows in enumerate(self.layout):
            for k, dt, shape, off, n in rows:
                a = self.starts[i] + off
                fill(k, self.buf[a : a + n].view(dt).view(shape))
        ptr = self.buf.data_ptr()
        self.pinned = _register_prefix(ptr, [*self.starts[1:], pos])
        if self.pinned == 0:
            msg = f"cudaHostRegister of the {pos / 2**30:.1f} GiB block buffer failed"
            raise RuntimeError(msg)
        if self.pinned < pos:  # Windows page-locks at most about half of the RAM (15.1 GiB of 32 GB)
            log.warning(
                "the OS page-locked %.1f of the %.1f GiB of streamed blocks; the other blocks upload from pageable"
                " memory, without overlapping the computation (slower). More RAM, or a smaller variant / NF4 where"
                " the model offers one, keeps every block page-locked",
                self.pinned / 2**30,
                pos / 2**30,
            )
        weakref.finalize(self, _unregister, ptr)  # before the buffer is freed (the finalizer runs first)
        self.dev: list[torch.Tensor] = []
        self.side: torch.cuda.Stream | None = None
        self.uploaded = [torch.cuda.Event(), torch.cuda.Event()]
        self.free = [torch.cuda.Event(), torch.cuda.Event()]

    def __len__(self) -> int:
        """Number of blocks."""
        return len(self.layout)

    def _views(self, i: int, dev: torch.Tensor) -> dict[str, torch.Tensor]:
        return {k: dev[off : off + n].view(dt).view(shape) for k, dt, shape, off, n in self.layout[i]}

    def _upload(self, i: int) -> None:
        j = i % 2
        assert self.side is not None  # noqa: S101 - set by run
        with torch.cuda.stream(self.side):
            self.side.wait_event(self.free[j])  # the block that used this buffer has finished
            a, n = self.starts[i], self.sizes[i]
            self.dev[j][:n].copy_(self.buf[a : a + n], non_blocking=True)
            self.uploaded[j].record(self.side)

    def run(self, device: torch.device) -> Iterable[dict[str, torch.Tensor]]:
        """Yield each block's tensors on `device`; the next block uploads while the caller computes this one."""
        # Two callers share the upload buffers and (for functional_call) block shells.
        # Serialize enqueueing; CUDA events still overlap transfers with computation.
        with self._lock:
            yield from self._run(device)

    def _run(self, device: torch.device) -> Iterable[dict[str, torch.Tensor]]:
        main = torch.cuda.current_stream(device)
        if not self.dev or self.dev[0].device != device:
            self.dev = [torch.empty(max(self.sizes), dtype=torch.uint8, device=device) for _ in range(2)]
            self.side = torch.cuda.Stream(device)
            self.side.wait_stream(main)
            for buf in self.dev:
                buf.record_stream(self.side)
        for buf in self.dev:
            buf.record_stream(main)
        self._upload(0)
        for i in range(len(self)):
            j = i % 2
            main.wait_event(self.uploaded[j])
            if i + 1 < len(self):
                self._upload(i + 1)
            try:
                yield self._views(i, self.dev[j])
            finally:
                self.free[j].record(main)


def _call_block(block: torch.nn.Module, state: dict[str, torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401 - the block's own signature
    return torch.func.functional_call(block, state, args, kwargs, strict=True)


class StreamedModuleList(torch.nn.Module):
    """An inference-only block sequence. Each yielded callable is valid for its iteration."""

    target: torch.Tensor

    def __init__(self, blocks: Iterable[torch.nn.Module], device: torch.device) -> None:
        """Pack CPU weights and buffers once, preserving aliases introduced by fp16 kernel rewrites."""
        super().__init__()
        self.shells = list(blocks)  # kept on meta outside the registered tree when the parent moves
        self.register_buffer("target", torch.empty(0, device=device), persistent=False)
        tensors: dict[str, torch.Tensor] = {}
        layout = []
        self.aliases: list[dict[str, tuple[str, torch.Size]]] = []
        for i, block in enumerate(self.shells):
            unique = {}
            spec = {}
            aliases = {}
            for name, tensor in itertools.chain(block.named_parameters(), block.named_buffers()):
                value = tensor.detach()
                # A linear matrix and its contiguous 1x1-convolution view share the same bytes.
                # For other strides preserve each view's logical element order during packing.
                identity = (
                    value.data_ptr(),
                    value.numel(),
                    value.dtype,
                    () if value.is_contiguous() else (value.shape, value.stride()),
                )
                key = unique.get(identity)
                if key is None:
                    key = unique[identity] = f"{i}.{name}"
                    tensors[key] = value
                    spec[key] = (value.dtype, value.shape)
                aliases[name] = (key, value.shape)
            layout.append(spec)
            self.aliases.append(aliases)
            block.to("meta")

        def fill(key: str, dst: torch.Tensor) -> None:
            dst.copy_(tensors.pop(key))

        self.host = HostBlocks(layout, fill)

    def __len__(self) -> int:
        """Number of transformer blocks."""
        return len(self.shells)

    def __iter__(self) -> Iterator[Callable[..., torch.Tensor]]:
        """Stage one block ahead while the caller computes the current block."""
        for block, aliases, buffers in zip(self.shells, self.aliases, self.host.run(self.target.device), strict=True):
            state = {name: buffers[key].view(shape) for name, (key, shape) in aliases.items()}
            yield functools.partial(_call_block, block, state)
