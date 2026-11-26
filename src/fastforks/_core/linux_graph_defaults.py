"""Exact-workload admission for Linux CUDA graph routes."""

from __future__ import annotations

import functools
import os
import sys
from typing import TYPE_CHECKING, NamedTuple

import torch

from fastforks._core._measured_graphs import APPROVED

if TYPE_CHECKING:
    from fastforks._core._measured_graphs import MeasuredKey, TensorSignature
    from fastforks._core.model import Model

_CHW = 3
_BCHW = 4
_RGB = 3


class ImagePair(NamedTuple):
    """Original arguments to an ordinary ordered two-image public request."""

    first: object
    second: object


class VideoQueries(NamedTuple):
    """Original arguments to the public tracking request."""

    video: object
    queries: object


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


def _image_tensor(value: object, device: torch.device) -> TensorSignature | None:
    if not isinstance(value, torch.Tensor) or value.device != device:
        return None
    layout = _image_layout(value)
    return None if layout is None else (layout, tuple(value.shape))


def _tracking_inputs(call: VideoQueries, device: torch.device) -> tuple[TensorSignature, TensorSignature] | None:
    video, queries = call
    if (
        not isinstance(video, torch.Tensor)
        or video.device != device
        or video.dtype != torch.float32
        or video.ndim != _BCHW
        or video.shape[1] != _RGB
        or not video.is_contiguous()
        or video.storage_offset() != 0
    ):
        return None
    if (
        not isinstance(queries, torch.Tensor)
        or (queries.device.type != "cpu" and queries.device != device)
        or queries.dtype != torch.float32
        or queries.ndim != 2  # noqa: PLR2004 - [Q, (t, x, y)]
        or queries.shape[1] != 3  # noqa: PLR2004 - (t, x, y)
        or not queries.is_contiguous()
        or queries.storage_offset() != 0
    ):
        return None
    return ("f32-tchw-cuda-contiguous", tuple(video.shape)), (
        f"f32-q3-{queries.device.type}-contiguous",
        tuple(queries.shape),
    )


def _workload_key(model: Model, call: object, device: torch.device) -> MeasuredKey | None:
    ref, mode, dtype, task = model.variant.ref, model.backend.mode, model.backend.dtype, model.card.task
    if isinstance(call, ImagePair) and task in {"flow", "stereo"}:
        first, second = (_image_tensor(x, device) for x in call)
        if first is not None and second is not None and first[1][-2:] == second[1][-2:]:
            return ref, mode, dtype, f"{task}-pair", (first, second)
        return None
    if isinstance(call, VideoQueries) and task == "tracking":
        args = _tracking_inputs(call, device)
        return None if args is None else (ref, mode, dtype, "tracking-video-queries", args)
    if task in {"flow", "stereo", "tracking"}:
        return None
    image = _image_tensor(call, device)
    if image is None:
        return None
    layout, shape = image
    return ref, mode, dtype, layout, (shape[-2], shape[-1])


def allows(model: Model, call: object = None) -> bool:
    """The original public arguments match a graph-on/eager measured pair."""
    if sys.platform != "linux":
        return True
    if call is None:
        return False
    options = model.options
    if (
        model.weights is not None
        or options.get("cuda_graphs") is False
        or (options and (len(options) != 1 or "cuda_graphs" not in options))
    ):
        return False
    device = model._device  # noqa: SLF001 - the runner's device
    if device.type != "cuda":
        return False
    key = _workload_key(model, call, device)
    profiles = APPROVED.get(key) if key is not None else None
    if profiles is None or hasattr(model._runner, "_ff_eager") or os.getenv("FASTFORKS_ARENA") == "1":  # noqa: SLF001
        return False
    allocator = torch.cuda.get_allocator_backend()
    settings = (os.getenv("PYTORCH_ALLOC_CONF"), os.getenv("PYTORCH_CUDA_ALLOC_CONF"))
    backend_only = f"backend:{allocator}"
    if all(setting in (None, backend_only) for setting in settings):
        settings = (None, None)
    runtime = (
        str(torch.__version__),
        _gpu_name(device),
        allocator,
        *settings,
        os.getenv("TRITON_DEFAULT_FP_FUSION"),
        os.getenv("OMP_NUM_THREADS"),
        torch.get_num_threads(),
        tuple(sorted(model.backend.kernels)),
        model.backend.weights,
    )
    return runtime in profiles
