"""Matting / background removal (`fastforks.matting`): a per-pixel foreground alpha in [0, 1]."""

from fastforks._core.tasks._base import VALUE_KEYS, TaskSpec

SPEC = TaskSpec(
    standard=("alpha",),
    extras=False,
    required=frozenset({"alpha"}),
    output_keys={},
    io_keys=VALUE_KEYS,
    order=35,
    inputs=("image", "trimap", "mask"),  # trimap / mask: only where the runner requires it (`input_use`)
)
