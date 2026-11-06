"""Image classification (`fastforks.classification`): class logits of a timm-style network."""

from fastforks._core.tasks._base import LABELS, VALUE_KEYS, TaskSpec

SPEC = TaskSpec(
    standard=("logits",),
    extras=False,
    required=frozenset({"logits"}),
    output_keys=LABELS,
    io_keys=VALUE_KEYS | {"resize", "crop", "interpolation"},
    order=20,
    inputs=("image",),
    export_size=(224, 224),  # used only if the card has no crop / resize
)
