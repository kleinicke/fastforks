"""Single-image camera calibration (`fastforks.calibration`): intrinsics, and gravity in the camera frame."""

from fastforks._core.tasks._base import VALUE_KEYS, TaskSpec

SPEC = TaskSpec(
    standard=("intrinsics", "gravity", "roll", "pitch", "vfov"),
    extras=True,
    required=frozenset({"intrinsics"}),
    output_keys={},
    io_keys=VALUE_KEYS,
    order=67,
    inputs=("image",),
)
