"""One-time CUDA library start-up, run in a background thread while a model loads.

The first use of an op class in a process pays a one-time cost on the CPU: the CUDA context, cuBLAS/cuBLASLt and
cuDNN handles and the lazy loading of each kernel module (e.g. ~120 ms for the first fp16 matmul, ~70 ms for the
first fp16 convolution, ~85 ms for the first `masked_fill`: 300-500 ms per model in total). Torch ops release the
GIL, so tiny calls of these ops in a thread overlap with the load's checkpoint reading and network building on the
main thread; handles return to torch's per-device pools when the thread ends and the main thread reuses them.
`start` runs once per device and process; the first model call on that device ends it (`wait`: the running step
finishes, the rest is skipped), so it never runs next to a model call (nor a CUDA graph capture), and a short load
does not wait for steps its model may not need. An exit hook ends a warm-up still running at interpreter exit (e.g.
after a failing load): a daemon thread inside a torch op at finalization ends in `std::terminate` (exit code 134).
"""

from __future__ import annotations

import atexit
import logging
import threading
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["start", "wait"]

_threads: dict[int, tuple[threading.Thread, threading.Event]] = {}
_lock = threading.Lock()


def _steps(device: torch.device) -> list[Callable[[], object]]:
    """Tiny calls of common op classes, the most widely used first."""

    def t(*shape: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        return torch.ones(*shape, device=device, dtype=dtype)

    def mm(dtype: torch.dtype) -> Callable[[], object]:
        return lambda: t(64, 64, dtype=dtype) @ t(64, 64, dtype=dtype)  # cuBLAS / cuBLASLt

    def conv(dtype: torch.dtype) -> Callable[[], object]:
        return lambda: F.conv2d(t(1, 16, 16, 16, dtype=dtype), t(16, 16, 3, 3, dtype=dtype), padding=1)  # cuDNN

    return [
        mm(torch.float16),
        conv(torch.float16),
        mm(torch.float32),
        lambda: t(64, 64).masked_fill(t(64, 64) > 0, 0),
        lambda: F.gelu(t(64, 64)).softmax(-1),
        lambda: F.layer_norm(t(64, 64), (64,)),
        lambda: F.scaled_dot_product_attention(t(1, 4, 64, 32), t(1, 4, 64, 32), t(1, 4, 64, 32)),
        lambda: F.interpolate(t(1, 16, 16, 16), scale_factor=2, mode="bilinear", align_corners=False),
        conv(torch.float32),
        mm(torch.bfloat16),
        conv(torch.bfloat16),
    ]


def _ops(device: torch.device, stop: threading.Event) -> None:
    with torch.cuda.device(device), torch.inference_mode():
        for step in _steps(device):
            if stop.is_set():
                break
            step()
        torch.cuda.synchronize(device)
        # Free the cuBLAS workspace this thread's handle got (torch does the same before a graph capture): a model
        # that never calls cuBLAS must not keep 32 MiB more, and one that does allocates its own on its first call.
        clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)  # noqa: SLF001 - no public equivalent
        if clear is not None:
            clear()
        torch.cuda.empty_cache()  # and the segments the steps cached: the model lays out its memory as without


def _run(device: torch.device, stop: threading.Event) -> None:
    try:
        _ops(device, stop)
    except Exception as e:  # noqa: BLE001 - only a head start: the model's own calls report real errors
        logging.getLogger("fastforks").debug("CUDA warm-up on %s failed: %s", device, e)


def start(device: torch.device) -> None:
    """Start the warm-up of `device` (a CUDA device with an index) unless it already ran in this process."""
    with _lock:
        if device.index in _threads:
            return
        stop = threading.Event()
        th = threading.Thread(target=_run, args=(device, stop), name=f"fastforks-warmup-{device}", daemon=True)
        _threads[device.index] = th, stop
    th.start()


def wait(device: torch.device) -> None:
    """End the warm-up of `device` (if any): its remaining steps are skipped, the running one finishes."""
    entry = _threads.get(device.index)
    if entry is not None and entry[0].is_alive():
        entry[1].set()
        entry[0].join()


@atexit.register  # runs before finalization: a daemon thread taking the GIL after it exits through C++ frames
def _end_all() -> None:
    for index in list(_threads):
        wait(torch.device("cuda", index))
