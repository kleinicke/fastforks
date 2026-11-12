"""Monocular depth (`fastforks.depth`): depth, inverse depth or points of one image (a batch = independent images)."""

from typing import Literal, get_args

from fastforks._core.tasks._base import VALUE_KEYS, TaskSpec

DepthKind = Literal["metric", "scale_invariant", "affine_invariant", "affine_invariant_inverse", "none"]
"""What depth / inverse depth / points mean: metres, up to scale, up to scale and shift, inverse up to both, or none
(a normals-only model)."""

SPEC = TaskSpec(
    standard=("depth", "inverse_depth", "points", "intrinsics", "confidence", "mask", "normals"),
    extras=True,
    required=frozenset({"depth", "inverse_depth", "points", "normals"}),
    output_keys={"kind": get_args(DepthKind)},
    io_keys=VALUE_KEYS | {"pad"},
    order=10,
    bool_outputs=frozenset({"mask"}),
    inputs=("image",),
)
