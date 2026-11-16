"""Streamed transformer blocks: one page-locked host buffer, uploaded one block ahead into two device buffers.

All blocks live in the host buffer; each is uploaded into one of two fixed device buffers (ping-pong) on a side stream
while the current block computes.

Each block's tensors are packed contiguously (one copy per block). The host buffer is ordinary memory registered with
`cudaHostRegister`: torch's pinned allocator rounds each allocation up to a power of two (8.4 GB of blocks would pin
~13-15 GB). The two device buffers are allocated once and reused across blocks and calls, so the allocator's reserved
memory stays at the first call's.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

ALIGN = 256  # bytes: start of every tensor in the packed buffers


def _up(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


def available_host_bytes() -> int | None:
    """Host RAM available without swapping (Linux `MemAvailable`); None where it cannot be read."""
    try:
        with open("/proc/meminfo", encoding="ascii") as f:  # noqa: PTH123
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


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


class HostBlocks:
    """The blocks' tensors, packed per block in one registered host buffer; `run` streams them to the device."""

    def __init__(
        self, blocks: list[dict[str, tuple[torch.dtype, torch.Size]]], fill: Callable[[str, torch.Tensor], None]
    ) -> None:
        """`blocks`: per block the checkpoint key -> (dtype, shape); `fill(key, dst)` copies that tensor into `dst`."""
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
        err = int(torch.cuda.cudart().cudaHostRegister(ptr, pos, 0))
        if err != 0:
            msg = f"cudaHostRegister of the {pos / 2**30:.1f} GiB block buffer failed (cudaError {err})"
            raise RuntimeError(msg)
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
        if not self.dev or self.dev[0].device != device:
            self.dev = [torch.empty(max(self.sizes), dtype=torch.uint8, device=device) for _ in range(2)]
            self.side = torch.cuda.Stream(device)
        main = torch.cuda.current_stream(device)
        self._upload(0)
        for i in range(len(self)):
            j = i % 2
            main.wait_event(self.uploaded[j])
            if i + 1 < len(self):
                self._upload(i + 1)
            yield self._views(i, self.dev[j])
            self.free[j].record(main)
