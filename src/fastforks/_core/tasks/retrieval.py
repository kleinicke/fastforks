"""Image retrieval / place recognition (`fastforks.retrieval`): one unit-length global descriptor per image."""

from fastforks._core.tasks._base import VALUE_KEYS, TaskSpec

SPEC = TaskSpec(
    standard=("descriptor",),
    extras=False,
    required=frozenset({"descriptor"}),
    output_keys={},
    io_keys=VALUE_KEYS,
    order=65,
    inputs=("image",),
)
