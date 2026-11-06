"""Stereo matching (`fastforks.stereo`): disparity of a rectified pair."""

from fastforks._core.tasks._base import VALUE_KEYS, TaskSpec

SPEC = TaskSpec(
    standard=("disparity", "confidence"),
    extras=True,
    required=frozenset({"disparity"}),
    output_keys={},
    io_keys=VALUE_KEYS | {"pad"},
    order=0,
    inputs=("left", "right"),
)
