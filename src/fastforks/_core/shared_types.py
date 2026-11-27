"""The shared types the task packages exported in 0.1.1 (`fastforks.depth.ModelCard`), now only `fastforks.<Name>`."""

from __future__ import annotations

from fastforks._core.arguments import Later
from fastforks._core.cards import ModelCard
from fastforks._core.image import ImageLike
from fastforks._core.model import Backend
from fastforks._core.sizes import SizeRange

__all__ = ["OLD_TASK_NAMES"]


def _tensorrt_engine() -> object:
    from fastforks._core.export import TensorRTEngine  # noqa: PLC0415 - export imports torch: at first access

    return TensorRTEngine


# old name in a task package -> (the new name, the object), for `renamed_attributes`
OLD_TASK_NAMES: dict[str, tuple[str, object]] = {
    "Backend": ("fastforks.Backend", Backend),
    "ImageLike": ("fastforks.ImageLike", ImageLike),
    "ModelCard": ("fastforks.ModelCard", ModelCard),
    "SizeRange": ("fastforks.SizeRange", SizeRange),
    "TensorRTEngine": ("fastforks.TensorRTEngine", Later(_tensorrt_engine)),
}
