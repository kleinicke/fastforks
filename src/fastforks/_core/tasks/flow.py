"""Optical flow (`fastforks.flow`): per-pixel motion from one image to the next, in input pixels."""

from fastforks._core.tasks._base import VALUE_KEYS, TaskSpec

SPEC = TaskSpec(
    standard=("flow",),
    extras=False,
    required=frozenset({"flow"}),
    output_keys={},
    io_keys=VALUE_KEYS | {"pad"},
    order=75,
    inputs=("image1", "image2"),
)
