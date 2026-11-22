"""Small host values onto the device without a host sync.

`torch.tensor(values, device="cuda")` copies from pageable memory, which waits for the copy and so empties the
launch queue; under Windows' WDDM, where each launch costs more, a few of these per call cost milliseconds.
`device_tensor` stages the values in pinned memory and copies them asynchronously instead. PyTorch's caching host
allocator keeps the staging buffer alive until the copy has run, so the values can't be overwritten early.
"""

from __future__ import annotations

from typing import Any

import torch


def device_tensor(data: Any, *, device: torch.device | str, dtype: torch.dtype | None = None) -> torch.Tensor:  # noqa: ANN401 - whatever torch.tensor takes
    """`torch.tensor(data, dtype=dtype, device=device)`, without a host sync on CUDA."""
    device = torch.device(device)
    if isinstance(data, torch.Tensor):  # torch.tensor would copy it (and warn); a CUDA tensor cannot be pinned
        return data.to(device, dtype).clone()
    if device.type != "cuda" or torch.compiler.is_compiling() or torch.jit.is_tracing():  # traced: a constant
        return torch.tensor(data, dtype=dtype, device=device)
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def to_device(t: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """`t.to(device)` for a CPU tensor, without a host sync on CUDA (the values as made on the CPU, bit for bit)."""
    device = torch.device(device)
    if device.type != "cuda" or t.device.type != "cpu" or torch.compiler.is_compiling() or torch.jit.is_tracing():
        return t.to(device)
    return t.pin_memory().to(device, non_blocking=True)
