"""Point tracking (`fastforks.tracking`): where query points of a video are in every frame, and whether visible."""

from fastforks._core.tasks._base import VALUE_KEYS, TaskSpec

SPEC = TaskSpec(
    standard=("tracks", "visibility"),
    extras=False,
    required=frozenset({"tracks"}),
    output_keys={},
    io_keys=VALUE_KEYS,
    order=77,
    inputs=("video", "queries"),  # module(): [T, 3, H, W] frames in [0, 1], [Q, 3] (t, x, y) queries
    export_size=(256, 256),
)
