"""3D volumes (`fastforks.volume`): voxel labels (nnU-Net models), encoder features, or a promptable mask."""

from fastforks._core.tasks._base import VALUE_KEYS, TaskSpec

SPEC = TaskSpec(
    standard=("labels", "features", "mask"),
    extras=False,
    required=frozenset({"labels", "features", "mask"}),
    bool_outputs=frozenset({"mask"}),  # promptable models: the object mask
    output_keys={},
    io_keys=VALUE_KEYS,
    order=85,
    inputs=("volume",),
)
