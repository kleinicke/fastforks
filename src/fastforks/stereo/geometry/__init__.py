"""Stereo geometry: camera models, calibration files, rectification, depth and point clouds.

Runtime dependencies are numpy and torch only. Typical use::

    from fastforks.stereo import geometry as g

    rig = g.read_calib("calib_cam_to_cam.txt", "kitti-unrectified")  # or build g.StereoRig(left, right, R, T)
    rect = g.rectify(rig)  # rotations, rectified intrinsics; maps are built on first use and reused
    left_r, right_r = rect.remap(left, right)
    disparity = ...  # any fastforks stereo model on (left_r, right_r)
    cloud = rect.point_cloud(disparity, colors=left_r, frame="camera")

Already rectified data (KITTI `P_rect`, Middlebury, ETH3D) reads as a `RectifiedRig` with the same `depth` and
`point_cloud` methods and needs no remapping.
"""

from fastforks.stereo.geometry.calib import read_calib
from fastforks.stereo.geometry.cameras import (
    EUCM,
    FOV,
    KB4,
    UCM,
    Camera,
    DoubleSphere,
    E57Cylindrical,
    E57Pinhole,
    E57Spherical,
    Fisheye624,
    Pinhole,
    PinholeOpenCV,
    ThinPrismFisheye,
    camera,
)
from fastforks.stereo.geometry.ply import write_las, write_ply
from fastforks.stereo.geometry.rectify import Rectification, rectify
from fastforks.stereo.geometry.rig import PointCloud, RectifiedRig, StereoRig

__all__ = [
    "EUCM",
    "FOV",
    "KB4",
    "UCM",
    "Camera",
    "DoubleSphere",
    "E57Cylindrical",
    "E57Pinhole",
    "E57Spherical",
    "Fisheye624",
    "Pinhole",
    "PinholeOpenCV",
    "PointCloud",
    "Rectification",
    "RectifiedRig",
    "StereoRig",
    "ThinPrismFisheye",
    "camera",
    "read_calib",
    "rectify",
    "write_las",
    "write_ply",
]
