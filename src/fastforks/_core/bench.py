"""Task-generic benchmarking (`fastforks bench`): synthetic inputs, the per-task input hook and wall-time medians.

Every task's `_api` module defines `bench_inputs(model, size) -> (args, kwargs)`: the positional and keyword arguments
of one representative call of `model` on synthetic data of `size` (a pair for stereo, an image for depth, a centre
point prompt for promptable segmentation, `size.views` views for multi-view). The CLI times `model(*args, **kwargs)`.
"""

from __future__ import annotations

import importlib
import statistics
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import torch
import torch.nn.functional as F

from fastforks._core.errors import FastforksError

WARMUP = (
    4  # calls before timing: `kernels` builds a new size's cudnn_hacc plans on its 1st call, CUDA graphs on its 4th
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks._core.model import Model

BenchInputs: TypeAlias = tuple[tuple[Any, ...], dict[str, Any]]
"""`(args, kwargs)` of one model call."""


@dataclass(frozen=True)
class BenchSize:
    """What `fastforks bench` asks for; each task's `bench_inputs` uses the fields that apply to it.

    Attributes:
        height: Image height in pixels.
        width: Image width in pixels.
        views: Views per scene (multi-view tasks).
    """

    height: int
    width: int
    views: int = 8


def bench_inputs(model: Model, size: BenchSize) -> BenchInputs:
    """The task's `bench_inputs(model, size)` (from `fastforks.<task>._api`).

    Raises:
        FastforksError: The task defines no `bench_inputs`.
    """
    task = model.card.task
    hook: Callable[[Model, BenchSize], BenchInputs] | None = getattr(
        importlib.import_module(f"fastforks.{task}._api"), "bench_inputs", None
    )
    if hook is None:
        msg = f"bench is not implemented for {task} models (fastforks.{task}._api defines no bench_inputs)"
        raise FastforksError(msg)
    return hook(model, size)


def synthetic_pair(height: int, width: int, device: torch.device, *, shift: int = 8) -> tuple[torch.Tensor, ...]:
    """A textured RGB pair `[1, 3, H, W]` in `[0, 1]` whose right view is the left shifted by `shift` px (checks)."""
    gen = torch.Generator().manual_seed(0)
    coarse = torch.rand(1, 3, height // 16 + 2, (width + shift) // 16 + 2, generator=gen)
    img = F.interpolate(coarse, size=(height, width + shift), mode="bicubic", align_corners=False)
    img = (0.7 * img + 0.3 * torch.rand(img.shape, generator=gen)).clamp(0, 1).to(device)
    return img[..., :width].contiguous(), img[..., shift:].contiguous()


def synthetic_image(height: int, width: int, device: torch.device) -> torch.Tensor:
    """A textured RGB image `[1, 3, H, W]` in `[0, 1]` (the left view of `synthetic_pair`)."""
    return synthetic_pair(height, width, device)[0]


def median_ms(fn: Any, *, device: str | None, runs: int = 10, warmup: int = WARMUP) -> float:  # noqa: ANN401 - any callable
    """Median wall time of `fn()` in ms after `warmup` calls (on a GPU `device`, synchronised before and after each)."""
    sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(device or "", lambda: None)
    times = []
    for i in range(warmup + runs):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        if i >= warmup:
            times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times)
