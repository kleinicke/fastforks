"""Matching (`fastforks.matching`): where the content of one image lies in another (any viewpoint).

Dense models return a warp of every pixel (`warp`, `certainty`); sparse ones detect keypoints in both images and
return the index pairs of the matched ones (`keypoints1`, `keypoints2`, `pairs`, `scores`).
"""

from fastforks._core.tasks._base import VALUE_KEYS, TaskSpec

SPEC = TaskSpec(
    standard=("warp", "certainty", "keypoints1", "keypoints2", "pairs", "scores"),
    extras=False,
    required=frozenset({"warp", "pairs"}),
    output_keys={},
    io_keys=VALUE_KEYS,
    order=76,
    inputs=("image1", "image2"),
)
