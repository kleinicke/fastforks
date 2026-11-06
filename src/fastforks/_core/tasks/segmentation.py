"""Semantic segmentation (`fastforks.segmentation`): per-pixel class labels (logits on request)."""

from fastforks._core.tasks._base import LABELS, VALUE_KEYS, TaskSpec

SPEC = TaskSpec(
    standard=("logits",),
    extras=False,
    required=frozenset({"logits"}),
    output_keys=LABELS,
    io_keys=VALUE_KEYS | {"pad"},
    order=30,
    inputs=("image",),
)
