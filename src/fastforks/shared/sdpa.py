"""Which `F.scaled_dot_product_attention` backends fastforks enables on CUDA, in which priority (logged, overridable).

PyTorch picks the first enabled backend, in priority order, that supports a call (dtype, head dim, mask, ...). Its
default order is flash, memory-efficient, math, cuDNN. PyTorch's Windows CUDA wheels are built without flash, so
there half-precision calls run the memory-efficient backend, about twice as slow as flash (fp16 `[1, 6, 4096, 64]`
on an RTX 5060 Ti: 1.18 ms efficient, 0.62 ms flash, 0.61 ms cuDNN).

Policy (`FASTFORKS_SDPA`, read at each call):

- `auto` (default): PyTorch's own order where the build has flash (Linux wheels: unchanged, nothing logged).
  Without flash: cuDNN, then memory-efficient, then math, logged. cuDNN is as accurate as memory-efficient and
  15-25 % faster for SAM 2 and Depth Anything V2 portable on Windows (RF-DETR: about 10 % slower); the golden
  bounds of the attention references cover flash, cuDNN, memory-efficient and math (docs/plans/windows-kernels.md,
  decisions 6 and 9).
  A runner whose attention cuDNN serves badly sets `Runner.sdpa_auto`, its own priority list for this case (also
  logged): BiRefNet's and InSPyReNet's Swin window attention (`shared.swin_v1`: 144 tokens, a bias mask) takes
  `efficient,cudnn,math`, since cuDNN builds a graph of about 40 ms for each new shape (birefnet's first call at
  1080p 0.93 -> 0.34 s) and is no faster there (birefnet 126.6 -> 124.2 ms, inspyrenet-fast 18.5 -> 16.0 ms; RTX
  5060 Ti, Windows; docs/plans/windows-kernels.md, decision 9). OneFormer's Swin backbone (`shared.maskformer`) does
  the same (first 1024 x 2048 call 0.95 -> 0.45 s).
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

from fastforks._core.lazy import lazy

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    import torch
    from torch.nn.attention import SDPBackend
else:
    torch = lazy("torch")


__all__ = ["ENV", "backends", "policy"]

ENV = "FASTFORKS_SDPA"
NAMES = {"flash": "FLASH_ATTENTION", "cudnn": "CUDNN_ATTENTION", "efficient": "EFFICIENT_ATTENTION", "math": "MATH"}
NO_FLASH = ("cudnn", "efficient", "math")  # `auto` without flash


def _backend(name: str) -> SDPBackend:
    """The `SDPBackend` of a policy name (torch's enum, looked up at first use)."""
    from torch.nn.attention import SDPBackend  # noqa: PLC0415 - torch at first use

    return getattr(SDPBackend, NAMES[name])


def backends(auto: str | None = None) -> tuple[SDPBackend, ...] | None:
    """The enabled SDPA backends in priority order under `FASTFORKS_SDPA`; `None`: PyTorch's own choice.

    Args:
        auto: A runner's priority list (`Runner.sdpa_auto`, e.g. `efficient,cudnn,math`) that replaces cuDNN-first
            under `auto` where the build has no flash; an explicit `FASTFORKS_SDPA` list or `torch` wins over it.

    Raises:
        ValueError: `FASTFORKS_SDPA` names an unknown backend or is empty.
    """
    value = os.environ.get(ENV, "auto").strip().lower() or "auto"
    if auto is not None and value == "auto" and not torch.backends.cuda.is_flash_attention_available():
        return _runner_order(auto)
    return _resolve(value)


@functools.cache
def _runner_order(value: str) -> tuple[SDPBackend, ...]:
    chosen = _parse(value, "Runner.sdpa_auto")
    logging.getLogger("fastforks").info(
        "SDPA: this model's attention uses %s, in that order, where PyTorch has no flash (%s=auto)", value, ENV
    )
    return chosen


def _parse(value: str, source: str) -> tuple[SDPBackend, ...]:
    names = [n.strip() for n in value.split(",")]
    unknown = [n for n in names if n not in NAMES]
    if unknown:
        msg = f"{source}={value!r}: unknown SDPA backend {unknown}; use auto, torch or a list of {', '.join(NAMES)}"
        raise ValueError(msg)
    return tuple(dict.fromkeys(_backend(n) for n in names))


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
            "SDPA: this PyTorch build has no flash attention; CUDA attention uses cuDNN, then memory-efficient, then "
            "math (%s=auto; %s=efficient,math,cudnn is PyTorch's order without flash)",
            ENV,
            ENV,
        )
        return tuple(map(_backend, NO_FLASH))
    chosen = _parse(value, ENV)
    names = dict.fromkeys(n.strip() for n in value.split(","))
    log.info("SDPA: CUDA attention uses %s, in that order (%s)", ", ".join(names), ENV)
    return chosen


def policy(device: torch.device, auto: str | None = None) -> AbstractContextManager[object]:
    """`sdpa_kernel(backends(auto), set_priority=True)` on a CUDA `device`; a no-op elsewhere or for PyTorch's choice.

    The switches are process-global: callers hold `fastforks._core.model._GPU_CALLS`.
    """
    chosen = backends(auto) if device.type == "cuda" else None
    if chosen is None:
        return contextlib.nullcontext()
    from torch.nn.attention import sdpa_kernel  # noqa: PLC0415 - torch at first use

    return sdpa_kernel(list(chosen), set_priority=True)
