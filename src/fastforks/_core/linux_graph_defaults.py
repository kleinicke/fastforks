"""Exact-workload admission for enrolled Linux CUDA graph routes."""

from __future__ import annotations

import functools
import os
import sys
from typing import TYPE_CHECKING

import torch

from fastforks._core._measured_graphs import APPROVED, ENROLLED

if TYPE_CHECKING:
    from fastforks._core.model import Model

_CHW = 3
_BCHW = 4
_RGB = 3


def enrolled(model: Model) -> bool:
    return (
        sys.platform == "linux"
        and model._device.type == "cuda"  # noqa: SLF001 - the device of the runner
        and model.backend.mode == "kernels"
        and (model.card.task, model.card.family) in ENROLLED
    )


@functools.cache
def _gpu_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device)


def _image_layout(image: torch.Tensor) -> str | None:
    """Identify the measured layout of an image before public input conversion."""
    if (
        image.dtype == torch.float32
        and image.ndim == _BCHW
        and image.shape[:2] == (1, _RGB)
        and image.is_contiguous()
        and image.storage_offset() == 0
    ):
        return "f32-bchw-contiguous"
    if (
        image.dtype == torch.uint8
        and image.ndim == _CHW
        and image.shape[0] == _RGB
        and image.stride() == (1, image.shape[-1] * _RGB, _RGB)
        and image.storage_offset() == 0
    ):
        return "u8-chw-rgb-interleaved"
    return None


def allows(model: Model, image: object = None) -> bool:
    """An original public CUDA image matching a graph-on/eager measured pair."""
    if not enrolled(model):
        return True
    options = model.options
    if (
        model.weights is not None
        or options.get("cuda_graphs") is False
        or (options and (len(options) != 1 or "cuda_graphs" not in options))
    ):
        return False
    if not isinstance(image, torch.Tensor) or image.device != model._device:  # noqa: SLF001 - the runner's device
        return False
    layout = _image_layout(image)
    if layout is None:
        return False
    profiles = APPROVED.get((model.variant.ref, model.backend.dtype, layout, (image.shape[-2], image.shape[-1])))
    if profiles is None or os.getenv("FASTFORKS_ARENA") == "1":
        return False
    allocator = torch.cuda.get_allocator_backend()
    settings = (os.getenv("PYTORCH_ALLOC_CONF"), os.getenv("PYTORCH_CUDA_ALLOC_CONF"))
    backend_only = f"backend:{allocator}"
    if all(setting in (None, backend_only) for setting in settings):
        settings = (None, None)
    runtime = (
        str(torch.__version__),
        _gpu_name(image.device),
        allocator,
        *settings,
        os.getenv("TRITON_DEFAULT_FP_FUSION"),
        os.getenv("OMP_NUM_THREADS"),
        torch.get_num_threads(),
        tuple(sorted(model.backend.kernels)),
        model.backend.weights,
    )
    return runtime in profiles
