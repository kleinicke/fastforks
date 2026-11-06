"""Backbone features (`fastforks.features`): HF-named token features of an image (a batch = independent images)."""

from fastforks._core.tasks._base import VALUE_KEYS, TaskSpec

SPEC = TaskSpec(  # [io] pad.multiple = patch stride the core resizes to (never pads); interpolation of it
    standard=("last_hidden_state", "pooler_output", "hidden_states"),
    extras=False,
    required=frozenset({"last_hidden_state"}),
    output_keys={},
    io_keys=VALUE_KEYS | {"pad", "interpolation"},
    order=60,
    inputs=("image",),
    export_size=(224, 224),  # a multiple of every patch stride (DINOv2 14, DINOv3 ViT 16, ConvNeXt 32): no resize
)
