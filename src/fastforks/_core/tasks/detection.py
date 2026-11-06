"""Object detection (`fastforks.detection`): a variable number of instances per image.

Boxes in input pixels, class labels (a label set), optional instance masks.
"""

from fastforks._core.tasks._base import LABELS, VALUE_KEYS, TaskSpec

SPEC = TaskSpec(
    standard=("boxes", "scores", "labels", "masks"),
    extras=False,
    required=frozenset({"boxes"}),
    output_keys=LABELS,
    io_keys=VALUE_KEYS,
    order=50,
    bool_outputs=frozenset({"masks"}),  # instance masks pasted into the image
    inputs=("image",),  # module(): RGB in [0, 1] at the network size (normalised and padded inside)
)
