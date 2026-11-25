"""Small host values onto the device without a host sync.

`torch.tensor(values, device="cuda")` copies from pageable memory, which waits for the copy and so empties the
launch queue; under Windows' WDDM, where each launch costs more, a few of these per call cost milliseconds.
`device_tensor` stages the values in pinned memory and copies them asynchronously instead. PyTorch's caching host
allocator keeps the staging buffer alive until the copy has run, so the values can't be overwritten early.

Inside a CUDA graph capture an upload becomes a copy node that every replay repeats from the same host memory, so the
staging buffer must outlive the graph instead: `capture_uploads` collects those buffers for the capturing code to keep
with its graph (`fastforks.shared.cuda_graphs` does), and an upload in a capture without it raises.
"""

from __future__ import annotations

import contextlib
import threading
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Generator

_capture = threading.local()  # .uploads: the staging buffers of the capture open on this thread


@contextlib.contextmanager
def capture_uploads() -> Generator[list[torch.Tensor]]:
    """Around a CUDA graph capture: the list of pinned staging buffers the uploads on this thread make in it.

    Keep the list as long as the graph: its replays copy from those buffers again.
    """
    outer = getattr(_capture, "uploads", None)
    held: list[torch.Tensor] = []
    _capture.uploads = held
    try:
        yield held
    finally:
        _capture.uploads = outer


def _staged(pinned: torch.Tensor) -> torch.Tensor:
    """`pinned`, kept by the open `capture_uploads` list when the current stream is capturing a CUDA graph.

    Raises:
        RuntimeError: A capture without `capture_uploads`, whose replays would read a freed (reused) host buffer.
    """
    if torch.cuda.is_current_stream_capturing():
        held = getattr(_capture, "uploads", None)
        if held is None:
            msg = "a host upload inside a CUDA graph capture needs `capture_uploads` to keep its staging buffer"
            raise RuntimeError(msg)
        held.append(pinned)
    return pinned


def device_tensor(data: Any, *, device: torch.device | str, dtype: torch.dtype | None = None) -> torch.Tensor:  # noqa: ANN401 - whatever torch.tensor takes
    """`torch.tensor(data, dtype=dtype, device=device)`, without a host sync on CUDA."""
    device = torch.device(device)
    if isinstance(data, torch.Tensor):  # torch.tensor would copy it (and warn); a CUDA tensor cannot be pinned
        return data.to(device, dtype).clone()
    if device.type != "cuda" or torch.compiler.is_compiling() or torch.jit.is_tracing():  # traced: a constant
        return torch.tensor(data, dtype=dtype, device=device)
    if isinstance(data, np.ndarray):
        # PyTorch cannot create a pinned tensor directly from an ndarray.
        return to_device(torch.tensor(data, dtype=dtype, device="cpu"), device)
    return _staged(torch.tensor(data, dtype=dtype, pin_memory=True)).to(device, non_blocking=True)


def to_device(t: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """`t.to(device)` for a CPU tensor, without a host sync on CUDA (the values as made on the CPU, bit for bit)."""
    device = torch.device(device)
    if device.type != "cuda" or t.device.type != "cpu" or torch.compiler.is_compiling() or torch.jit.is_tracing():
        return t.to(device)
    return _staged(t.pin_memory()).to(device, non_blocking=True)
