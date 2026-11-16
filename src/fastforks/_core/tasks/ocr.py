"""Text recognition (`fastforks.ocr`): text lines found in an image, each a quad, its text and a confidence."""

from fastforks._core.tasks._base import VALUE_KEYS, TaskSpec

SPEC = TaskSpec(
    standard=("boxes", "texts", "scores"),
    extras=False,
    required=frozenset({"scores"}),
    output_keys={},
    io_keys=VALUE_KEYS,
    order=52,
    inputs=("image",),  # module(): the text detector, RGB in [0, 1] at the network size (multiples of 32)
)
