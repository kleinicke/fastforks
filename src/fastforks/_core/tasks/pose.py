"""2D human pose estimation (`fastforks.pose`): keypoints of every person, in input pixels.

Top-down models take one box per person (from a detector, or ground truth) and return the keypoints inside it; the
keypoint names come from the card's label set (`[output] label_set`, e.g. `coco-keypoints-17`).
"""

from fastforks._core.tasks._base import LABELS, VALUE_KEYS, TaskSpec

SPEC = TaskSpec(
    standard=("keypoints", "keypoint_scores", "boxes", "scores"),
    extras=False,
    required=frozenset({"keypoints"}),
    output_keys=LABELS,
    io_keys=VALUE_KEYS,
    order=55,
    inputs=("image",),  # module(): person crops RGB in [0, 1] at the network size -> heatmaps
    export_size=(256, 192),  # the crop size of the top-down models
)
