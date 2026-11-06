"""Stereo rigs: `StereoRig` (calibrated, unrectified) and `RectifiedRig` (already rectified data)."""

from __future__ import annotations

import dataclasses
import math
import numbers
from typing import TYPE_CHECKING, Literal, NamedTuple, TypeVar, cast

import numpy as np
import torch

from fastforks._core.errors import InputError
from fastforks.stereo.geometry.cameras import Camera, hw_size

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

ArrayT = TypeVar("ArrayT", np.ndarray, torch.Tensor)
Frame = Literal["rectified", "camera"]
View = Literal["left", "right"]
_ROTATION_TOLERANCE = 1e-4  # files round R to ~1e-6; more is a calibration bug
UNITS = {"m": 1.0, "cm": 0.01, "mm": 0.001}  # metres per unit


def unit_scale(unit: object) -> float:
    """Metres per file unit for `unit=`: "m", "cm", "mm" or a positive finite number (the factor to metres)."""
    if isinstance(unit, str) and unit in UNITS:
        return UNITS[unit]
    if isinstance(unit, (int, float)) and not isinstance(unit, bool) and math.isfinite(unit) and unit > 0:
        return float(unit)
    msg = f"unit must be one of {', '.join(UNITS)} or a positive number (the factor to metres), got {unit!r}"
    raise InputError(msg)


_CAMERA_AXES = {"opencv": (1.0, 1.0, 1.0), "opengl": (1.0, -1.0, -1.0)}  # x right; y down, z forward / y up, z back


def _rigid_pose(pose: object, what: str) -> tuple[np.ndarray, np.ndarray]:
    """`(R, t)` of a finite rigid 4x4 camera-to-world matrix; scale, shear or a mirror raise `InputError`."""
    m = np.asarray(pose, dtype=np.float64) if isinstance(pose, (np.ndarray, list, tuple)) else np.zeros(0)
    if m.shape != (4, 4) or not np.isfinite(m).all() or np.abs(m[3] - (0, 0, 0, 1)).max() > 1e-6:  # noqa: PLR2004
        msg = f"{what} must be a finite 4x4 camera-to-world matrix with the bottom row [0, 0, 0, 1]"
        raise InputError(msg)
    error = np.abs(m[:3, :3] @ m[:3, :3].T - np.eye(3)).max()
    if error > _ROTATION_TOLERANCE or np.linalg.det(m[:3, :3]) < 0:
        msg = (
            f"{what} is not a rigid pose (rotation part off orthonormal by {error:.2g}): it has scale, shear or a"
            " mirror. A camera pose is a rotation plus a translation; put the scale in unit=, and check that the matrix"
            " is camera-to-world (not a projection or world-to-camera)"
        )
        raise InputError(msg)
    return m[:3, :3], m[:3, 3]


class PointCloud(NamedTuple):
    """Per-pixel 3D points of a disparity map.

    Attributes:
        points: `[..., H, W, 3]` float32 points (metres if the baseline is in metres), NaN where not valid.
        valid: `[..., H, W]` bool, True where the disparity gave a finite point in front of the camera (and, for
            `Rectification.point_cloud`, the rectified pixel has a source pixel).
        colors: The `colors` image passed in (`[..., H, W, C]` or `[..., H, W]`), or None; index it with `valid`.
    """

    points: np.ndarray | torch.Tensor
    valid: np.ndarray | torch.Tensor
    colors: np.ndarray | torch.Tensor | None


def as_tensor(a: np.ndarray | torch.Tensor, what: str) -> tuple[torch.Tensor, bool]:
    """`(tensor, came_from_numpy)` of an array input; anything else is an `InputError`.

    Read-only arrays (e.g. `StereoView.numpy()`) are copied: torch has no read-only tensors.
    """
    if isinstance(a, np.ndarray):
        return torch.from_numpy(np.require(a, requirements="CW")), True
    if isinstance(a, torch.Tensor):
        return a, False
    msg = f"{what} must be a numpy array or torch tensor, got {type(a).__name__}"
    raise InputError(msg)


def side_of(view: str) -> int:
    """0 for the left view, 1 for the right view; anything else is an `InputError`."""
    if view not in ("left", "right"):
        msg = f"view must be 'left' or 'right', got {view!r}"
        raise InputError(msg)
    return int(view == "right")


def _back(t: torch.Tensor, numpy: bool) -> np.ndarray | torch.Tensor:  # noqa: FBT001
    return t.cpu().numpy() if numpy else t


def _check_size(t: torch.Tensor, size: tuple[int, int] | None, what: str) -> None:
    if t.ndim < 2:  # noqa: PLR2004
        msg = f"{what} must have shape [..., H, W], got {tuple(t.shape)}"
        raise InputError(msg)
    if size is not None and tuple(t.shape[-2:]) != size:
        msg = f"{what} is {t.shape[-2]}x{t.shape[-1]} (height x width) but the rig is {size[0]}x{size[1]}"
        raise InputError(msg)


@dataclasses.dataclass(frozen=True)
class RectifiedRig:
    """A rectified stereo pair: one focal length, shared principal point row, horizontal baseline.

    Use it for data that is already rectified (KITTI, Middlebury, ETH3D, the output of `Rectification.remap`):
    `depth` and `point_cloud` need no remapping. A left pixel `(u, v)` with disparity `d` lies at depth
    `Z = f * baseline / (d + doffs)`, `X = (u - cx) Z / f`, `Y = (v - cy) Z / f`.

    Attributes:
        f: Focal length in pixels (fx = fy).
        baseline: Distance between the camera centres; depth comes out in this unit (metres from `read_calib`).
        cx: Principal point x of the left image, pixels (OpenCV convention: the first pixel centre is 0); default
            the image centre `(width - 1) / 2` from `size`.
        cy: Principal point y (both images), pixels; default `(height - 1) / 2`.
        doffs: `cx_right - cx_left` in pixels (Middlebury's `doffs`); 0 when both principal points coincide.
        size: `(height, width)` of the rectified images, or None if unknown (then inputs are not size-checked).
    """

    f: float
    baseline: float
    cx: float | None = None
    cy: float | None = None
    doffs: float = 0.0
    size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        """Fill the principal point (image centre) if missing; check finite values, positive focal and baseline."""
        if self.size is not None:
            object.__setattr__(self, "size", hw_size(self.size, what="RectifiedRig.size"))
        for name, axis in (("cx", 1), ("cy", 0)):
            if getattr(self, name) is None:
                if self.size is None:
                    msg = f"RectifiedRig needs {name} or a size to default it to the image centre"
                    raise InputError(msg)
                object.__setattr__(self, name, (self.size[axis] - 1) / 2)
        for name in ("f", "cx", "cy", "baseline", "doffs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value):
                msg = f"RectifiedRig.{name} must be a finite number, got {value!r}"
                raise InputError(msg)
            object.__setattr__(self, name, float(value))
        if self.f <= 0 or self.baseline <= 0:
            msg = f"RectifiedRig needs f > 0 and baseline > 0, got f={self.f}, baseline={self.baseline}"
            raise InputError(msg)

    def resized(self, size: tuple[int, int]) -> RectifiedRig:
        """The same rig for images resized to `size = (height, width)`: `f' = f s`, `c' = (c + 0.5) s - 0.5`.

        A rectified pair has one focal length (square pixels), so the aspect ratio must stay: the height may differ
        from `height * s` by at most one pixel of rounding (`s = new width / width`). Needs `size` on this rig.

        Args:
            size: New `(height, width)`.

        Returns:
            The rig for the resized images (`f`, `cx`, `cy`, `doffs` and `size` change).

        Raises:
            InputError: This rig has no `size`, `size` swaps portrait and landscape, or it scales x and y
                differently.
        """
        if self.size is None:
            msg = "RectifiedRig.resized needs the rig's size"
            raise InputError(msg)
        h, w = hw_size(size, self.size)
        s = w / self.size[1]
        if abs(self.size[0] * s - h) > 1:
            msg = (
                f"RectifiedRig.resized({(h, w)}) scales x by {s:.6g} and y by {h / self.size[0]:.6g}: a rectified rig"
                " has one focal length, so keep the aspect ratio (resize anisotropically scaled images back first)"
            )
            raise InputError(msg)
        return RectifiedRig(
            self.f * s,
            self.baseline,
            (self.cx + 0.5) * s - 0.5,  # ty: ignore[unsupported-operator]
            (self.cy + 0.5) * h / self.size[0] - 0.5,  # ty: ignore[unsupported-operator]
            self.doffs * s,
            (h, w),
        )

    def _cx_of(self, view: View) -> float:
        """Principal point x of a view: the right camera has `cx + doffs` (Middlebury: `doffs = cx_right - cx_left`)."""
        return self.cx + (self.doffs if side_of(view) else 0.0)  # ty: ignore[unsupported-operator]

    def depth(self, disparity: ArrayT, *, euclidean: bool = False, view: View = "left") -> ArrayT:
        """Depth `f * baseline / (disparity + doffs)` along the optical axis of the camera the disparity is of.

        Args:
            disparity: `[..., H, W]` disparity in pixels (numpy or torch, e.g. `StereoResult.disparity`).
            euclidean: Distance along each pixel's ray, `Z sqrt(1 + x^2 + y^2)`, instead of the depth `Z`.
            view: Camera the disparity belongs to; matters for `euclidean` when `doffs != 0` (the right camera's
                principal point is `cx + doffs`).

        Returns:
            Depth of the same shape and kind (float32 for integer and half-precision input), NaN where
            `disparity + doffs <= 0` or the disparity is not finite.
        """
        d, numpy = as_tensor(disparity, "disparity")
        _check_size(d, self.size, "disparity")
        d = _at_least_float32(d)
        denom = d + self.doffs
        ok = torch.isfinite(denom) & (denom > 0)
        z = torch.where(ok, (self.f * self.baseline) / torch.where(ok, denom, 1), torch.nan)
        if euclidean:
            h, w = d.shape[-2:]
            x = (torch.arange(w, device=d.device, dtype=z.dtype) - self._cx_of(view)) / self.f
            y = ((torch.arange(h, device=d.device, dtype=z.dtype) - cast("float", self.cy)) / self.f)[:, None]
            z = z * torch.sqrt(1 + x * x + y * y)
        return _back(z, numpy)  # ty: ignore[invalid-return-type]

    def point_cloud(
        self, disparity: ArrayT, *, colors: ArrayT | None = None, frame: Frame = "rectified", view: View = "left"
    ) -> PointCloud:
        """3D point of every pixel of one view, in that camera's frame.

        Args:
            disparity: `[..., H, W]` disparity of `view` in pixels (numpy or torch).
            view: `"left"` or `"right"` (the right camera's principal point is `cx + doffs`).
            colors: Optional left image `[..., H, W, C]` or `[..., H, W]` (the rectified one), returned as
                `PointCloud.colors` so `colors[valid]` lines up with `points[valid]`.
            frame: `"rectified"` or `"camera"`; identical for already rectified data.

        Returns:
            `PointCloud(points, valid, colors)`, arrays of the same kind as `disparity`.
        """
        if frame not in ("rectified", "camera"):
            msg = f"frame must be 'rectified' or 'camera', got {frame!r}"
            raise InputError(msg)
        points, valid, numpy = self._points(disparity, view)
        return PointCloud(_back(points, numpy), _back(valid, numpy), _check_colors(colors, valid))

    def _points(
        self, disparity: np.ndarray | torch.Tensor, view: View = "left"
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        d, numpy = as_tensor(disparity, "disparity")
        _check_size(d, self.size, "disparity")
        d = _at_least_float32(d)
        h, w = d.shape[-2:]
        denom = d + self.doffs
        valid = torch.isfinite(denom) & (denom > 0)
        z = torch.where(valid, (self.f * self.baseline) / torch.where(valid, denom, 1), torch.nan)
        u = torch.arange(w, device=d.device, dtype=d.dtype)
        v = torch.arange(h, device=d.device, dtype=d.dtype)[:, None]
        cy = cast("float", self.cy)
        points = torch.stack(
            torch.broadcast_tensors((u - self._cx_of(view)) * z / self.f, (v - cy) * z / self.f, z),
            -1,
        )
        return points, valid, numpy


def _at_least_float32(d: torch.Tensor) -> torch.Tensor:
    """float32 for integer and half-precision disparities (bf16 rounds pixel indices above 256), else as is."""
    return d if d.is_floating_point() and d.element_size() >= 4 else d.float()  # noqa: PLR2004


def _check_colors(colors: np.ndarray | torch.Tensor | None, valid: torch.Tensor) -> np.ndarray | torch.Tensor | None:
    if colors is None:
        return None
    shape = tuple(colors.shape)
    if shape[: valid.ndim] != tuple(valid.shape) and shape != tuple(valid.shape):
        msg = f"colors must have shape {tuple(valid.shape)} or {(*valid.shape, 'C')}, got {shape}"
        raise InputError(msg)
    return colors


@dataclasses.dataclass(frozen=True, eq=False)
class StereoRig:
    """Two calibrated cameras and the pose of the right camera relative to the left one.

    A point `X_left` in the left camera frame is `X_right = R @ X_left + T` in the right camera frame (OpenCV's
    `stereoCalibrate` convention). For a usual horizontal rig `T ~ (-baseline, 0, 0)`.

    Attributes:
        left: Left camera (any `Camera` model).
        right: Right camera.
        R: `[3, 3]` rotation, right-from-left; a rounded one (as stored in calibration files, `R R^T` off the
            identity by up to 1e-4) is replaced by the nearest exact rotation, a larger error raises `InputError`.
        T: `[3]` translation, right-from-left, in metres (depth and point clouds come out in this unit).
    """

    left: Camera
    right: Camera
    R: np.ndarray
    T: np.ndarray

    def __post_init__(self) -> None:
        """Check the cameras and the pose; store `R` and `T` as float64 arrays."""
        for name in ("left", "right"):
            if not isinstance(getattr(self, name), Camera):
                msg = f"StereoRig.{name} must be a camera model (e.g. PinholeOpenCV), got {type(getattr(self, name))}"
                raise InputError(msg)
        rot = np.array(self.R, dtype=np.float64)
        t = np.array(self.T, dtype=np.float64).reshape(-1)  # a copy: the caller's array stays theirs
        if rot.shape != (3, 3) or t.shape != (3,) or not (np.isfinite(rot).all() and np.isfinite(t).all()):
            msg = f"StereoRig needs a finite R [3, 3] and T [3], got {rot.shape} and {np.shape(self.T)}"
            raise InputError(msg)
        error = np.abs(rot @ rot.T - np.eye(3)).max()
        if error > _ROTATION_TOLERANCE or np.linalg.det(rot) < 0:
            msg = f"StereoRig.R must be a rotation matrix (orthonormal to 1e-4, det +1), got |R R^T - I| = {error:.2g}"
            raise InputError(msg)
        u, _, vt = np.linalg.svd(rot)  # files store rounded rotations: use the nearest exact one
        rot = u @ vt
        if np.linalg.norm(t) <= 0:
            msg = "StereoRig.T must not be zero (the cameras need a baseline)"
            raise InputError(msg)
        rot.flags.writeable = t.flags.writeable = False  # the rig is frozen: no in-place edits either
        object.__setattr__(self, "R", rot)
        object.__setattr__(self, "T", t)

    @classmethod
    def from_poses(  # noqa: PLR0913 - the two cameras and poses, keyword-only
        cls,
        left: Camera,
        right: Camera,
        *,
        left_to_world: ArrayLike,
        right_to_world: ArrayLike,
        unit: Literal["m", "cm", "mm"] | float = "m",
        axes: Literal["opencv", "opengl"] = "opencv",
    ) -> StereoRig:
        """Build a rig from two cameras and their camera-to-world poses (any world frame), e.g. from a dataset's JSON.

        Args:
            left: Left camera (any `Camera`, e.g. `camera("pinhole", w, h, fx, fy, cx, cy)`).
            right: Right camera; the right one must sit on the +x side of the left one (else `rectify` raises).
            left_to_world: `[4, 4]` rigid camera-to-world matrix of the left camera (what many datasets call
                `T_WorldFromCamera` or `transform_matrix`), the same world and unit as `right_to_world`.
            right_to_world: `[4, 4]` camera-to-world matrix of the right camera.
            unit: Unit of the translations in both matrices (`"mm"` for SHOW3D, or the factor to metres as a
                number); the rig is in metres.
            axes: Camera axes of the matrices: `"opencv"` (x right, y down, z forward) or `"opengl"` (x right, y up,
                z backward: nerfstudio, instant-ngp, Blender).

        Returns:
            The rig with `X_right = R X_left + T`.

        Raises:
            InputError: A matrix that is not a finite rigid 4x4 pose, an unknown `unit` or `axes`, identical positions.
        """
        scale = unit_scale(unit)
        if axes not in _CAMERA_AXES:
            msg = f"axes must be opencv or opengl, got {axes!r}"
            raise InputError(msg)
        flip = np.array(_CAMERA_AXES[axes])  # opengl -> opencv camera axes: negate the y and z columns
        (r_l, t_l), (r_r, t_r) = (
            _rigid_pose(left_to_world, "left_to_world"),
            _rigid_pose(right_to_world, "right_to_world"),
        )
        r_l, r_r = r_l * flip, r_r * flip
        return cls(left, right, r_r.T @ r_l, r_r.T @ (t_l - t_r) * scale)

    @property
    def baseline(self) -> float:
        """Distance between the two camera centres, in the unit of `T`."""
        return float(np.linalg.norm(self.T))

    def resized(self, size: tuple[int, int]) -> StereoRig:
        """The same rig for images resized to `size = (height, width)` (see `Camera.resized`; R and T stay).

        Args:
            size: New `(height, width)` in pixels.

        Returns:
            The rig of the resized cameras.
        """
        return StereoRig(self.left.resized(size), self.right.resized(size), self.R, self.T)
