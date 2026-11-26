"""Exact-workload admission for Linux CUDA graph routes."""

from __future__ import annotations

import functools
import os
import sys
from typing import TYPE_CHECKING, NamedTuple

import torch

from fastforks._core._measured_graphs import APPROVED
from fastforks.shared import sdpa

if TYPE_CHECKING:
    from fastforks._core._measured_graphs import MeasuredKey, TensorSignature
    from fastforks._core.model import Model

_CHW = 3
_BCHW = 4
_RGB = 3
_BOX_COORDINATES = 4
_PAIR_TASKS = {"flow", "interpolation", "stereo"}
_MEASURED_VRAM_CLASS = 15 << 30  # visible capacity of the measured 16 GiB board is slightly lower than 16 GiB


class ImagePair(NamedTuple):
    """Original arguments to an ordinary ordered two-image public request."""

    first: object
    second: object


class VideoQueries(NamedTuple):
    """Original arguments to the public tracking request."""

    video: object
    queries: object


class PoseBoxes(NamedTuple):
    """Original arguments to a top-down pose request for one image."""

    image: object
    boxes: object


@functools.cache
def _gpu_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device)


@functools.cache
def _gpu_memory(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).total_memory


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


def _adjacent_rgb_pair(call: ImagePair, device: torch.device) -> tuple[TensorSignature, TensorSignature] | None:
    """Adjacent ordered BCHW views of one exactly sized RGB-interleaved frame batch."""
    first, second = call
    if (
        not isinstance(first, torch.Tensor)
        or not isinstance(second, torch.Tensor)
        or first.device != device
        or second.device != device
        or first.layout != torch.strided
        or second.layout != torch.strided
        or first.dtype != torch.float32
        or second.dtype != torch.float32
        or first.ndim != _BCHW
        or first.shape[1] != _RGB
        or second.shape != first.shape
    ):
        return None
    batch, _, height, width = first.shape
    if batch == 0 or height == 0 or width == 0:
        return None
    frame = _RGB * height * width
    stride = (frame, 1, _RGB * width, _RGB)
    if (
        first.stride() != stride
        or second.stride() != stride
        or first.storage_offset() != 0
        or second.storage_offset() != frame
    ):
        return None
    storage = first.untyped_storage()
    if storage.nbytes() != (batch + 1) * frame * first.element_size():
        return None
    if storage.data_ptr() != second.untyped_storage().data_ptr():
        return None
    sig = ("f32-bchw-adjacent-rgb-pair", tuple(first.shape))
    return sig, sig


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


def _pose_inputs(call: PoseBoxes, device: torch.device) -> tuple[TensorSignature, TensorSignature] | None:
    image, boxes = call
    if (
        not isinstance(image, torch.Tensor)
        or image.device != device
        or image.dtype != torch.float32
        or image.ndim != _CHW
        or image.shape[0] != _RGB
        or not image.is_contiguous()
        or image.storage_offset() != 0
    ):
        return None
    if (
        not isinstance(boxes, torch.Tensor)
        or boxes.device != device
        or boxes.dtype != torch.float32
        or boxes.ndim != 2  # noqa: PLR2004 - one row per person
        or boxes.shape[1] != _BOX_COORDINATES
        or not boxes.is_contiguous()
        or boxes.storage_offset() != 0
    ):
        return None
    return ("f32-chw-contiguous", tuple(image.shape)), ("f32-p4-cuda-contiguous", tuple(boxes.shape))


def _workload_key(model: Model, call: object, device: torch.device) -> MeasuredKey | None:
    ref, mode, dtype, task = model.variant.ref, model.backend.mode, model.backend.dtype, model.card.task
    if isinstance(call, ImagePair) and task in _PAIR_TASKS:
        first, second = (_image_tensor(x, device) for x in call)
        if first is not None and second is not None and first[1][-2:] == second[1][-2:]:
            operands = first, second
        else:
            operands = _adjacent_rgb_pair(call, device) if task == "interpolation" else None
        route = "interpolation-midpoint-pair" if task == "interpolation" else f"{task}-pair"
        return None if operands is None else (ref, mode, dtype, route, operands)
    if isinstance(call, VideoQueries) and task == "tracking":
        args = _tracking_inputs(call, device)
        return None if args is None else (ref, mode, dtype, "tracking-video-queries", args)
    if isinstance(call, PoseBoxes) and task == "pose":
        args = _pose_inputs(call, device)
        return None if args is None else (ref, mode, dtype, "pose-boxes", args)
    if task in _PAIR_TASKS or task == "tracking":
        return None
    image = _image_tensor(call, device)
    if image is None:
        return None
    layout, shape = image
    return ref, mode, dtype, layout, (shape[-2], shape[-1])


def _standard_torch_settings(model: Model) -> bool:
    """Keep the measured CUDA backend and precision settings for capture and replay."""
    cudnn = torch.backends.cudnn
    if (
        torch.are_deterministic_algorithms_enabled()
        or not cudnn.enabled
        or cudnn.benchmark
        or (cudnn.deterministic and not model._runner.deterministic_cudnn)  # noqa: SLF001 - runner policy
    ):
        return False
    cuda = torch.backends.cuda
    if cuda.matmul.allow_fp16_accumulation or not (
        cuda.flash_sdp_enabled()
        and cuda.mem_efficient_sdp_enabled()
        and cuda.math_sdp_enabled()
        and cuda.cudnn_sdp_enabled()
    ):
        return False
    if os.getenv(sdpa.ENV, "auto").strip().lower() not in ("", "auto", "torch"):
        return False
    try:
        return cudnn.allow_tf32 and torch.get_float32_matmul_precision() == "highest"
    except RuntimeError:
        # PyTorch can disable its legacy getters when backend-specific FP32 modes differ.
        return False


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
    return runtime in profiles and _standard_torch_settings(model) and _gpu_memory(device) >= _MEASURED_VRAM_CLASS
