"""Dense matching (`fastforks.matching`): for every pixel of one image, where it lies in another (any viewpoint)."""

from fastforks._core.tasks._base import VALUE_KEYS, TaskSpec

SPEC = TaskSpec(
    standard=("warp", "certainty"),
    extras=False,
    required=frozenset({"warp"}),
    output_keys={},
    io_keys=VALUE_KEYS,
    order=76,
    inputs=("image1", "image2"),
)
