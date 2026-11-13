"""AnySplat's copy of VGGT (InternRobotics/AnySplat 5f5e208, `src/model/encoder/vggt/`; VGGT License, see ../LICENSE).

The files are those of `../../vggt/net` (facebookresearch/vggt a288dd0, same modules and lib changes), with AnySplat's
one functional change: `utils/pose_enc.py` builds the intrinsics with `tan(fov / 2) + 1e-3` (and in the pose encoding's
dtype), as AnySplat's copy does. AnySplat's other edits (gradient checkpointing, selecting the intermediate layers by
list position) do not change the eval outputs. A copy, not an import: families never import each other.
"""
