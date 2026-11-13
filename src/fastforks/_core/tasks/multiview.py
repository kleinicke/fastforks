"""Multi-view depth (`fastforks.multiview`): depth / points / cameras of the views of one scene (a call = one scene)."""

from typing import Literal, get_args

from fastforks._core.tasks._base import VALUE_KEYS, TaskSpec

MultiViewKind = Literal["metric", "scale_invariant"]
"""What multi-view depth / points / camera translations mean: `"metric"`: absolute units (metres for models that
estimate the cameras; for calibrated MVS the units of the given `cam_to_world`). `"scale_invariant"`: one unknown
scale shared by every view, depth and translation alike."""

SPEC = TaskSpec(
    standard=("depth", "world_points", "confidence", "mask", "intrinsics", "cam_to_world", "gaussians"),
    extras=True,
    required=frozenset({"depth", "world_points"}),
    output_keys={"kind": get_args(MultiViewKind)},
    io_keys=VALUE_KEYS | {"pad"},
    order=70,
    bool_outputs=frozenset({"mask"}),
    inputs=("images", "intrinsics", "cam_to_world", "depth_range"),
)
