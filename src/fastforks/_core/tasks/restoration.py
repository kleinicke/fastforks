"""Image restoration (`fastforks.restoration`): super-resolution and other image-to-image models (`scale` x size)."""

from fastforks._core.tasks._base import VALUE_KEYS, TaskSpec

SPEC = TaskSpec(
    standard=("image",),
    extras=False,
    required=frozenset({"image"}),
    output_keys={},
    io_keys=VALUE_KEYS | {"pad"},
    order=79,
    inputs=("image", "mask"),  # mask: only inpainting runners require it (`input_use`)
)
