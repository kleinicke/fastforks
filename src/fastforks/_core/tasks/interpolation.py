"""Frame interpolation (`fastforks.interpolation`): the frame between two images at a time in [0, 1]."""

from fastforks._core.tasks._base import VALUE_KEYS, TaskSpec

SPEC = TaskSpec(
    standard=("frame",),
    extras=False,
    required=frozenset({"frame"}),
    output_keys={},
    io_keys=VALUE_KEYS | {"pad"},
    order=78,
    inputs=("image1", "image2"),
)
