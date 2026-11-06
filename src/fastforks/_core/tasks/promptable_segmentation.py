"""Promptable segmentation (`fastforks.promptable_segmentation`): masks for point / box prompts.

SAM-style models: per-prompt low-res mask logits + predicted IoU; concept models (text / exemplar prompts): a variable
set of instances. The core upsamples and thresholds the masks.
"""

from fastforks._core.tasks._base import VALUE_KEYS, TaskSpec

SPEC = TaskSpec(
    standard=("pred_masks", "iou_scores", "boxes", "scores", "masks"),
    extras=False,
    required=frozenset({"pred_masks", "masks"}),
    output_keys={},
    io_keys=VALUE_KEYS,
    order=40,
    inputs=("image",),  # module(): the image encoder only
    export_size=(1024, 1024),  # fallback; the card's `params.image_size` (SAM 3: 1008) wins
)
