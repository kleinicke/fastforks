"""3D volume segmentation (`fastforks.volume`): a label per voxel of a CT / MR volume (nnU-Net based models)."""

from fastforks._core.tasks._base import VALUE_KEYS, TaskSpec

SPEC = TaskSpec(
    standard=("labels",),
    extras=False,
    required=frozenset({"labels"}),
    output_keys={},
    io_keys=VALUE_KEYS,
    order=85,
    inputs=("volume",),
)
