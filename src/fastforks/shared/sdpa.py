"""Which `F.scaled_dot_product_attention` backends fastforks enables on CUDA, in which priority (logged, overridable).

PyTorch picks the first enabled backend, in priority order, that supports a call (dtype, head dim, mask, ...). Its
default order is flash, memory-efficient, math, cuDNN. PyTorch's Windows CUDA wheels are built without flash, so
there half-precision calls run the memory-efficient backend, about twice as slow as flash (fp16 `[1, 6, 4096, 64]`
on an RTX 5060 Ti: 1.18 ms efficient, 0.62 ms flash, 0.61 ms cuDNN).

Policy (`FASTFORKS_SDPA`, read at each call):

- `auto` (default): PyTorch's own order where the build has flash (Linux wheels: unchanged, nothing logged).
  Without flash: memory-efficient, then math, then cuDNN (PyTorch's order minus flash), logged. cuDNN first was
  measured and is opt-in, not the default: it is as accurate and 15-25 % faster for SAM 2 and Depth Anything V2
  portable on Windows (RF-DETR: slower), but it fails the `rf-detr-s-coco@1` portable golden that passes with
  memory-efficient (and fixes two others: rounding, not accuracy; docs/plans/windows-kernels.md, decision 9).
- `torch`: PyTorch's own order everywhere (fastforks changes nothing).
- a comma-separated priority list of `flash`, `cudnn`, `efficient`, `math`, e.g. `cudnn,efficient,math` (cuDNN
  where it supports the call, else memory-efficient, else math): only those backends, in that order. PyTorch
  picks per call from the call's shape, dtype and mask, the same for every run of a build (no retry after an
  error); a call none of them supports raises PyTorch's "No available kernel" error.

Every model call on a CUDA device (`Model._session`, which holds the process-wide GPU call lock the global switches
need) runs under `policy()`; a runner that pins its own backends (`sdpa_kernel(...)` inside, e.g. FoundationStereo's
math attention) keeps them, and C-FoundationStereo's kernels take theirs from `backends()`. The resolved choice is
logged once per process and value (logger `fastforks`, INFO).
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
from typing import TYPE_CHECKING

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

__all__ = ["ENV", "backends", "policy"]

ENV = "FASTFORKS_SDPA"
NAMES = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
}
NO_FLASH = (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION)  # `auto` without flash


def backends() -> tuple[SDPBackend, ...] | None:
    """The enabled SDPA backends in priority order under `FASTFORKS_SDPA`; `None`: PyTorch's own choice.

    Raises:
        ValueError: `FASTFORKS_SDPA` names an unknown backend or is empty.
    """
    return _resolve(os.environ.get(ENV, "auto").strip().lower() or "auto")


@functools.cache
def _resolve(value: str) -> tuple[SDPBackend, ...] | None:
    log = logging.getLogger("fastforks")
    if value == "torch":
        log.info("SDPA: PyTorch's own backend order (%s=torch)", ENV)
        return None
    if value == "auto":
        if torch.backends.cuda.is_flash_attention_available():
            return None  # PyTorch's order, flash first: nothing to log, nothing changes
        log.info(
            "SDPA: this PyTorch build has no flash attention; CUDA attention uses memory-efficient, then math, then "
            "cuDNN (%s=auto; %s=cudnn,efficient,math prefers cuDNN: faster, other rounding)",
            ENV,
            ENV,
        )
        return NO_FLASH
    names = [n.strip() for n in value.split(",")]
    unknown = [n for n in names if n not in NAMES]
    if unknown:
        msg = f"{ENV}={value!r}: unknown SDPA backend {unknown}; use auto, torch or a list of {', '.join(NAMES)}"
        raise ValueError(msg)
    chosen = tuple(dict.fromkeys(NAMES[n] for n in names))
    log.info("SDPA: CUDA attention uses %s, in that order (%s)", ", ".join(dict.fromkeys(names)), ENV)
    return chosen


def policy(device: torch.device) -> AbstractContextManager[object]:
    """`sdpa_kernel(backends(), set_priority=True)` on a CUDA `device`; a no-op elsewhere or for PyTorch's choice.

    The switches are process-global: callers hold `fastforks._core.model._GPU_CALLS`.
    """
    chosen = backends() if device.type == "cuda" else None
    if chosen is None:
        return contextlib.nullcontext()
    return sdpa_kernel(list(chosen), set_priority=True)
