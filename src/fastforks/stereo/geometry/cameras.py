"""Camera models: one frozen dataclass per model, all following the `Camera` protocol.

Every model maps a 3D ray in its camera frame (x right, y down, z forward) to a pixel (`project`) and a pixel back
to a unit ray (`unproject`). Both take numpy arrays or torch tensors of shape `[..., 3]` / `[..., 2]`, compute in
float64 on the input's device, and return the same kind of array plus a boolean `valid` mask; invalid entries are
NaN. Pixel coordinates follow OpenCV: the centre of the top-left pixel is `(0, 0)`.

The math follows OpenCV (`PinholeOpenCV`, `KB4`) and the published models (`DoubleSphere`: Usenko et al. 2018,
`UCM`: Mei & Rives 2007, `EUCM`: Khomutenko et al. 2016, `FOV`: Devernay & Faugeras 2001, `Fisheye624`: Project
Aria), the E57 image representations (`E57Pinhole`, `E57Spherical`, `E57Cylindrical`) ASTM E2807 as read by
ply-visualizer. Inverse distortion is solved per pixel (Newton with bisection or damping), not by OpenCV's fixed
five iterations, so `project(unproject(p)) == p` to ~1e-9 px wherever the model is invertible.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import numbers
import operator
from typing import TYPE_CHECKING, ClassVar, Protocol, Self, TypeVar, runtime_checkable

import numpy as np
import torch

from fastforks._core.errors import InputError

if TYPE_CHECKING:
    from collections.abc import Callable

ArrayT = TypeVar("ArrayT", np.ndarray, torch.Tensor)
Pair = tuple[torch.Tensor, torch.Tensor]

_EPS = 1e-12
_MAX_THETA = math.pi - 1e-7
_ITERATIONS = 30
_TOLERANCE = 1e-10


@runtime_checkable
class Camera(Protocol):
    """What every camera model provides; `StereoRig` and `rectify` accept any object with these members."""

    model_id: ClassVar[str]

    @property
    def width(self) -> int:
        """Image width in pixels."""
        ...

    @property
    def height(self) -> int:
        """Image height in pixels."""
        ...

    @property
    def fx(self) -> float:
        """Focal length in x, pixels."""
        ...

    @property
    def fy(self) -> float:
        """Focal length in y, pixels."""
        ...

    @property
    def cx(self) -> float:
        """Principal point x, pixels."""
        ...

    @property
    def cy(self) -> float:
        """Principal point y, pixels."""
        ...

    @property
    def size(self) -> tuple[int, int]:
        """Image size `(height, width)` in pixels, the order of tensor shapes."""
        ...

    def resized(self, size: tuple[int, int]) -> Camera:
        """The same camera for images resized to `size = (height, width)` (OpenCV pixel convention).

        Args:
            size: New `(height, width)` in pixels.

        Returns:
            The camera for the resized image.
        """
        ...

    def project(self, points: ArrayT) -> tuple[ArrayT, ArrayT]:
        """Project camera-frame points to pixels.

        Args:
            points: `[..., 3]` points (or rays) in the camera frame, numpy or torch.

        Returns:
            `(pixels [..., 2], valid [...])`, same array kind and device as `points`; pixels are NaN where invalid.
        """
        ...

    def unproject(self, pixels: ArrayT) -> tuple[ArrayT, ArrayT]:
        """Unproject pixels to unit rays in the camera frame.

        Args:
            pixels: `[..., 2]` pixel coordinates `(u, v)`, numpy or torch.

        Returns:
            `(rays [..., 3], valid [...])`, same array kind and device as `pixels`; rays are NaN where invalid.
        """
        ...

    def _project_normalized(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple[Pair, torch.Tensor]:
        """Normalised image coordinates `(x / z, y / z)` after distortion and the validity mask (used by `rectify`)."""
        ...


def _to_tensor(a: np.ndarray | torch.Tensor, last: int, what: str) -> tuple[torch.Tensor, bool]:
    if isinstance(a, np.ndarray):
        t, numpy = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64)), True
    elif isinstance(a, torch.Tensor):
        t, numpy = a.to(torch.float64), False
    else:
        msg = f"{what} must be a numpy array or torch tensor, got {type(a).__name__}"
        raise InputError(msg)
    if t.ndim == 0 or t.shape[-1] != last:
        msg = f"{what} must have shape [..., {last}], got {tuple(t.shape)}"
        raise InputError(msg)
    return t, numpy


def _finish(value: torch.Tensor, valid: torch.Tensor, numpy: bool) -> tuple[ArrayT, ArrayT]:  # noqa: FBT001
    valid = valid & torch.isfinite(value).all(-1)
    value = torch.where(valid[..., None], value, torch.nan)
    if numpy:
        return value.numpy(), valid.numpy()
    return value, valid  # ty: ignore[invalid-return-type]


def _normalize(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    ray = torch.stack([x, y, z], -1)
    return ray / torch.linalg.vector_norm(ray, dim=-1, keepdim=True)


def _invert_2d(
    distort: Callable[[torch.Tensor, torch.Tensor], Pair], tx: torch.Tensor, ty: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve `distort(x, y) == (tx, ty)` per element: damped Newton from `(tx, ty)`, central-difference Jacobian."""
    x, y = tx.clone(), ty.clone()
    scale = _TOLERANCE * (1 + torch.hypot(tx, ty))
    h = 1e-6
    for _ in range(_ITERATIONS):
        dx, dy = distort(x, y)
        rx, ry = dx - tx, dy - ty
        done = ~(torch.hypot(rx, ry) > scale)  # NaN counts as done (and stays invalid)
        if bool(done.all()):
            break
        ax, ay = distort(x + h, y)
        bx, by = distort(x - h, y)
        cx, cy = distort(x, y + h)
        ex, ey = distort(x, y - h)
        j00, j10 = (ax - bx) / (2 * h), (ay - by) / (2 * h)
        j01, j11 = (cx - ex) / (2 * h), (cy - ey) / (2 * h)
        det = j00 * j11 - j01 * j10
        sx = (j11 * rx - j01 * ry) / det
        sy = (-j10 * rx + j00 * ry) / det
        damping = 1 / torch.clamp(torch.hypot(sx, sy), min=1.0)
        x = torch.where(done, x, x - damping * sx)
        y = torch.where(done, y, y - damping * sy)
    dx, dy = distort(x, y)
    ok = torch.hypot(dx - tx, dy - ty) <= scale
    return x, y, ok & (x.abs() < 1e6) & (y.abs() < 1e6)  # noqa: PLR2004


def _check_finite(obj: object, names: tuple[str, ...]) -> None:
    """Every field in `names` is a finite real number (numpy scalars too); stores it as a Python float."""
    for name in names:
        value = getattr(obj, name)
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
            msg = f"{type(obj).__name__}.{name} must be a finite number, got {value!r}"
            raise InputError(msg)
        object.__setattr__(obj, name, float(value))


def hw_size(size: object, like: tuple[int, int] | None = None, what: str = "size") -> tuple[int, int]:
    """`(height, width)` of a size argument: two integers >= 2.

    With `like` (the `(height, width)` of the images or calibration it applies to), a size that turns landscape into
    portrait or back raises: that is almost always `(width, height)` passed where `(height, width)` is expected.

    Raises:
        InputError: Not two integers >= 2, or the orientation differs from `like`.
    """
    h = w = 0
    try:
        h, w = size  # ty: ignore[not-iterable]
        ok = all(isinstance(v, numbers.Real) and not isinstance(v, bool) and int(v) == v >= 2 for v in (h, w))  # noqa: PLR2004
    except (TypeError, ValueError, OverflowError):
        ok = False
    if not ok:
        msg = f"{what} must be (height, width) with integers >= 2, got {size!r}"
        raise InputError(msg)
    h, w = int(h), int(w)
    if like is not None and (h - w) * (like[0] - like[1]) < 0:
        msg = (
            f"{what}={(h, w)} is (height, width) {h}x{w} but the images are {like[0]}x{like[1]} (height x width): "
            f"portrait and landscape are swapped; sizes are (height, width), did you mean {(w, h)}?"
        )
        raise InputError(msg)
    return h, w


def _check_size(obj: object, names: tuple[str, ...]) -> None:
    """Every field in `names` is an integer >= 2 (numpy integers too); stores it as a Python int."""
    for name in names:
        value = getattr(obj, name)
        try:
            index = None if isinstance(value, bool) else operator.index(value)
        except TypeError:
            index = None
        if index is None or index < 2:  # noqa: PLR2004
            msg = f"{type(obj).__name__}.{name} must be an int >= 2, got {value!r}"
            raise InputError(msg)
        object.__setattr__(obj, name, index)


@dataclasses.dataclass(frozen=True)
class _Base:
    """Image size and pinhole intrinsics shared by every model."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    model_id: ClassVar[str] = ""
    # Coefficient fields in the order `camera(model_id, ...)` and the calibration files list them.
    coefficient_names: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        _check_size(self, ("width", "height"))
        _check_finite(self, ("fx", "fy", "cx", "cy", *self.coefficient_names))
        if self.fx <= 0 or self.fy <= 0:
            msg = f"{type(self).__name__}: focal lengths must be positive, got fx={self.fx}, fy={self.fy}"
            raise InputError(msg)

    @property
    def size(self) -> tuple[int, int]:
        """Image size `(height, width)` in pixels, the order of tensor shapes."""
        return self.height, self.width

    @property
    def coefficients(self) -> tuple[float, ...]:
        """Distortion / model coefficients in `coefficient_names` order."""
        return tuple(float(getattr(self, n)) for n in self.coefficient_names)

    def resized(self, size: tuple[int, int]) -> Self:
        """The same camera for images resized to `size = (height, width)` (the whole field of view, no crop).

        Pixel centres sit at integer coordinates (OpenCV), so the image centre is `((W - 1) / 2, (H - 1) / 2)` and
        a pixel coordinate `c` maps to `c' = (c + 0.5) s - 0.5` (not `c s`), with `f' = f s` and `s = W' / W` per axis.
        The distortion coefficients and the other model parameters act on normalized coordinates and stay.

        Args:
            size: New `(height, width)` in pixels.

        Returns:
            The camera of the same model and coefficients for the resized image.

        Raises:
            InputError: `size` is not two integers >= 2, or it swaps portrait and landscape (see `hw_size`).
        """
        h, w = hw_size(size, self.size)
        sx, sy = w / self.width, h / self.height
        return dataclasses.replace(
            self,
            width=int(w),
            height=int(h),
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=(self.cx + 0.5) * sx - 0.5,
            cy=(self.cy + 0.5) * sy - 0.5,
        )

    def project(self, points: ArrayT) -> tuple[ArrayT, ArrayT]:
        """Project camera-frame points to pixels.

        Args:
            points: `[..., 3]` points (or rays) in the camera frame, numpy or torch.

        Returns:
            `(pixels [..., 2], valid [...])`, same array kind and device as `points`; pixels are NaN where invalid.
        """
        t, numpy = _to_tensor(points, 3, "points")
        m, valid = self._project_normalized(t[..., 0], t[..., 1], t[..., 2])
        return _finish(torch.stack([self.fx * m[0] + self.cx, self.fy * m[1] + self.cy], -1), valid, numpy)

    def unproject(self, pixels: ArrayT) -> tuple[ArrayT, ArrayT]:
        """Unproject pixels to unit rays in the camera frame.

        Args:
            pixels: `[..., 2]` pixel coordinates `(u, v)`, numpy or torch.

        Returns:
            `(rays [..., 3], valid [...])`, same array kind and device as `pixels`; rays are NaN where invalid.
        """
        t, numpy = _to_tensor(pixels, 2, "pixels")
        mx = (t[..., 0] - self.cx) / self.fx
        my = (t[..., 1] - self.cy) / self.fy
        ray, valid = self._unproject_normalized(mx, my)
        return _finish(ray, valid, numpy)

    def _project_normalized(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple[Pair, torch.Tensor]:
        raise NotImplementedError

    def _unproject_normalized(self, mx: torch.Tensor, my: torch.Tensor) -> Pair:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class Pinhole(_Base):
    """Ideal pinhole camera, no distortion (`"pinhole"`)."""

    model_id: ClassVar[str] = "pinhole"

    def _project_normalized(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple[Pair, torch.Tensor]:
        return (x / z, y / z), z > _EPS

    def _unproject_normalized(self, mx: torch.Tensor, my: torch.Tensor) -> Pair:
        return _normalize(mx, my, torch.ones_like(mx)), torch.isfinite(mx) & torch.isfinite(my)


@dataclasses.dataclass(frozen=True)
class PinholeOpenCV(_Base):
    """OpenCV pinhole model with the full 14-coefficient distortion (`"pinhole-opencv"`).

    Radial `k1..k6` (rational for `k4..k6`), tangential `p1, p2`, thin prism `s1..s4` and sensor tilt `tx, ty`
    (radians), exactly as `cv2.projectPoints`. Coefficients default to 0, so `PinholeOpenCV(w, h, fx, fy, cx, cy,
    k1=..., k2=..., p1=..., p2=..., k3=...)` is the common 5-coefficient model.
    """

    _: dataclasses.KW_ONLY
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0
    s1: float = 0.0
    s2: float = 0.0
    s3: float = 0.0
    s4: float = 0.0
    tx: float = 0.0
    ty: float = 0.0

    model_id: ClassVar[str] = "pinhole-opencv"
    coefficient_names: ClassVar[tuple[str, ...]] = (
        "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "tx", "ty",
    )  # fmt: skip

    def _tilt(self) -> tuple[np.ndarray, np.ndarray]:
        """OpenCV's `computeTiltProjectionMatrix` and its inverse."""
        sx, cx, sy, cy = math.sin(self.tx), math.cos(self.tx), math.sin(self.ty), math.cos(self.ty)
        rot_x = np.array([[1, 0, 0], [0, cx, sx], [0, -sx, cx]])
        rot_y = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]])
        rot = rot_y @ rot_x
        proj_z = np.array([[rot[2, 2], 0, -rot[0, 2]], [0, rot[2, 2], -rot[1, 2]], [0, 0, 1]])
        tilt = proj_z @ rot
        return tilt, np.linalg.inv(tilt)

    @property
    def _fold2(self) -> float:
        """Squared normalised radius of the distortion fold; rays beyond it are invalid."""
        return _fold_radius2((self.k1, self.k2, self.k3, self.k4, self.k5, self.k6))

    def _distort_untilted(self, x: torch.Tensor, y: torch.Tensor) -> Pair:
        # Terms with zero coefficients are skipped: most files use 4-5 of the 14 (the map build runs this per pixel).
        r2 = x * x + y * y
        radial = r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3)) + 1
        if self.k4 or self.k5 or self.k6:
            radial = radial / (r2 * (self.k4 + r2 * (self.k5 + r2 * self.k6)) + 1)
        xd, yd = x * radial, y * radial
        if self.p1 or self.p2:
            xy2 = 2 * x * y
            xd = xd + self.p1 * xy2 + self.p2 * (r2 + 2 * x * x)
            yd = yd + self.p1 * (r2 + 2 * y * y) + self.p2 * xy2
        if self.s1 or self.s2 or self.s3 or self.s4:
            xd = xd + r2 * (self.s1 + self.s2 * r2)
            yd = yd + r2 * (self.s3 + self.s4 * r2)
        return xd, yd

    @staticmethod
    def _homography(m: np.ndarray, x: torch.Tensor, y: torch.Tensor) -> Pair:
        w = m[2, 0] * x + m[2, 1] * y + m[2, 2]
        return (m[0, 0] * x + m[0, 1] * y + m[0, 2]) / w, (m[1, 0] * x + m[1, 1] * y + m[1, 2]) / w

    def _project_normalized(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple[Pair, torch.Tensor]:
        ok = z > _EPS
        x, y = x / torch.where(ok, z, 1.0), y / torch.where(ok, z, 1.0)
        xd, yd = self._distort_untilted(x, y)
        if self.tx or self.ty:
            xd, yd = self._homography(self._tilt()[0], xd, yd)
        # Past the radial fold, distortion bends rays back towards the centre: mirrored "ghost" pixels, not valid ones.
        return (xd, yd), ok & (x * x + y * y <= self._fold2)

    def _unproject_normalized(self, mx: torch.Tensor, my: torch.Tensor) -> Pair:
        if self.tx or self.ty:
            mx, my = self._homography(self._tilt()[1], mx, my)
        if not any(self.coefficients[:12]):
            return _normalize(mx, my, torch.ones_like(mx)), torch.isfinite(mx) & torch.isfinite(my)
        x, y, ok = _invert_2d(self._distort_untilted, mx, my)
        ok = ok & (x * x + y * y <= self._fold2)
        return _normalize(x, y, torch.ones_like(x)), ok


@functools.lru_cache(maxsize=64)
def _fold_radius2(radial: tuple[float, ...]) -> float:
    """Squared normalised radius where `r * radial(r^2)` of the OpenCV model stops increasing (inf: never folds).

    `radial` is `(k1, k2, k3, k4, k5, k6)`.

    Scans `r = tan(theta)` up to 90 degrees, then bisects the first sign change of the slope.
    """
    k1, k2, k3, k4, k5, k6 = radial

    def slope(r: float) -> float:
        r2 = r * r
        num, den = 1 + r2 * (k1 + r2 * (k2 + r2 * k3)), 1 + r2 * (k4 + r2 * (k5 + r2 * k6))
        dnum, dden = k1 + r2 * (2 * k2 + 3 * r2 * k3), k4 + r2 * (2 * k5 + 3 * r2 * k6)  # d/d(r^2)
        if den <= 0:
            return -1.0
        return num / den + 2 * r2 * (dnum * den - num * dden) / (den * den)

    previous = 0.0
    for step in range(1, 4097):
        r = math.tan(0.5 * math.pi * step / 4097)
        if slope(r) <= 0:
            low, high = previous, r
            for _ in range(60):
                mid = 0.5 * (low + high)
                low, high = (mid, high) if slope(mid) > 0 else (low, mid)
            return low * low
        previous = r
    return math.inf


@functools.lru_cache(maxsize=64)
def _radial_domain(coefficients: tuple[float, ...]) -> tuple[float, float]:
    """First monotonic lobe `[0, theta_max]` of `theta * (1 + k1 theta^2 + ...)` and the radius it reaches."""

    def slope(theta: float) -> float:
        return 1 + sum((2 * i + 3) * k * theta ** (2 * i + 2) for i, k in enumerate(coefficients))

    high = _MAX_THETA
    previous = 0.0
    for step in range(1, 513):
        theta = _MAX_THETA * step / 512
        if slope(theta) <= _EPS:
            high = previous
            break
        previous = theta
    return high, float(_radial(high, coefficients))


def _radial(theta: float | torch.Tensor, coefficients: tuple[float, ...]) -> float | torch.Tensor:
    theta2 = theta * theta
    poly = 0.0
    for k in reversed(coefficients):
        poly = (poly + k) * theta2
    return theta * (1 + poly)


def _invert_radial(rd: torch.Tensor, coefficients: tuple[float, ...]) -> Pair:
    """Solve `_radial(theta) == rd` on the first monotonic lobe: Newton steps kept inside a bisection bracket."""
    high, max_radius = _radial_domain(coefficients)
    if not any(coefficients):
        return rd, (rd < _MAX_THETA) & torch.isfinite(rd)
    lo = torch.zeros_like(rd)
    hi = torch.full_like(rd, high)
    theta = torch.clamp(rd, max=0.5 * high)
    tolerance = _TOLERANCE * (1 + rd)
    for _ in range(_ITERATIONS):
        res = torch.as_tensor(_radial(theta, coefficients)) - rd
        done = ~(res.abs() > tolerance)  # converged (or NaN): frozen, so bisection cannot undo it
        if bool(done.all()):
            break
        lo = torch.where(res <= 0, theta, lo)
        hi = torch.where(res > 0, theta, hi)
        h = 1e-7
        slope = (_radial(theta + h, coefficients) - _radial(theta - h, coefficients)) / (2 * h)
        step = theta - res / slope
        inside = (step > lo) & (step < hi) & (slope > _EPS)
        theta = torch.where(done, theta, torch.where(inside, step, 0.5 * (lo + hi)))
    res = torch.as_tensor(_radial(theta, coefficients)) - rd
    ok = (res.abs() <= tolerance) & (rd <= max_radius)
    return torch.where(rd <= _EPS, 0.0, theta), ok


def _fisheye_angles(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, Pair]:
    """Angle from the optical axis and the unit direction in the image plane (`(1, 0)` on the axis)."""
    r = torch.hypot(x, y)
    on_axis = r <= _EPS
    safe = torch.where(on_axis, 1.0, r)
    return torch.atan2(r, z), (torch.where(on_axis, 1.0, x / safe), torch.where(on_axis, 0.0, y / safe))


def _ray_from_angle(theta: torch.Tensor, mx: torch.Tensor, my: torch.Tensor) -> torch.Tensor:
    r = torch.hypot(mx, my)
    scale = torch.where(r <= _EPS, 0.0, torch.sin(theta) / torch.where(r <= _EPS, 1.0, r))
    return torch.stack([mx * scale, my * scale, torch.cos(theta)], -1)


@dataclasses.dataclass(frozen=True)
class KB4(_Base):
    """Kannala-Brandt fisheye with four radial terms, as `cv2.fisheye` (`"kb4"`, alias `"fisheye-opencv"`).

    `theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)`, with `theta` the angle to the
    optical axis; rays beyond 90 degrees project too (fields of view above 180 degrees). All-zero coefficients
    give the equidistant model (`"fisheye-equidistant"`); `"kb3"` (Project Aria `KannalaBrandtK3`, terms named
    `k0..k3`) is the same polynomial.
    """

    _: dataclasses.KW_ONLY
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0

    model_id: ClassVar[str] = "kb4"
    coefficient_names: ClassVar[tuple[str, ...]] = ("k1", "k2", "k3", "k4")

    def _project_normalized(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple[Pair, torch.Tensor]:
        theta, (ux, uy) = _fisheye_angles(x, y, z)
        rd = _radial(theta, self.coefficients)
        return (rd * ux, rd * uy), theta <= _radial_domain(self.coefficients)[0]

    def _unproject_normalized(self, mx: torch.Tensor, my: torch.Tensor) -> Pair:
        theta, ok = _invert_radial(torch.hypot(mx, my), self.coefficients)
        return _ray_from_angle(theta, mx, my), ok


@dataclasses.dataclass(frozen=True)
class Fisheye624(_Base):
    """Project Aria `Fisheye624`: six radial terms `k0..k5`, tangential `p0, p1`, thin prism `s0..s3`."""

    _: dataclasses.KW_ONLY
    k0: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    p0: float = 0.0
    p1: float = 0.0
    s0: float = 0.0
    s1: float = 0.0
    s2: float = 0.0
    s3: float = 0.0

    model_id: ClassVar[str] = "fisheye624"
    coefficient_names: ClassVar[tuple[str, ...]] = (
        "k0", "k1", "k2", "k3", "k4", "k5", "p0", "p1", "s0", "s1", "s2", "s3",
    )  # fmt: skip

    def _tangential(self, x: torch.Tensor, y: torch.Tensor) -> Pair:
        r2 = x * x + y * y
        r4 = r2 * r2
        xd = x + self.p0 * (2 * x * x + r2) + 2 * self.p1 * x * y + self.s0 * r2 + self.s1 * r4
        yd = y + self.p1 * (2 * y * y + r2) + 2 * self.p0 * x * y + self.s2 * r2 + self.s3 * r4
        return xd, yd

    def _project_normalized(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple[Pair, torch.Tensor]:
        radial = self.coefficients[:6]
        theta, (ux, uy) = _fisheye_angles(x, y, z)
        rd = _radial(theta, radial)
        return self._tangential(rd * ux, rd * uy), theta <= _radial_domain(radial)[0]

    def _unproject_normalized(self, mx: torch.Tensor, my: torch.Tensor) -> Pair:
        ok = torch.isfinite(mx) & torch.isfinite(my)
        if any(self.coefficients[6:]):
            mx, my, ok = _invert_2d(self._tangential, mx, my)
        theta, ok_radial = _invert_radial(torch.hypot(mx, my), self.coefficients[:6])
        return _ray_from_angle(theta, mx, my), ok & ok_radial


def _eucm_project(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, alpha: float, beta: float
) -> tuple[Pair, torch.Tensor]:
    rho = torch.sqrt(beta * (x * x + y * y) + z * z)
    norm = alpha * rho + (1 - alpha) * z
    w = (1 - alpha) / alpha if alpha > 0.5 else alpha / (1 - alpha)  # noqa: PLR2004
    return (x / norm, y / norm), (z > -w * rho) & (norm > _EPS)


def _eucm_unproject(mx: torch.Tensor, my: torch.Tensor, alpha: float, beta: float) -> Pair:
    r2 = mx * mx + my * my
    gamma = 1 - alpha
    ok = torch.isfinite(r2)
    if alpha > 0.5:  # noqa: PLR2004
        ok &= r2 < 1 / ((alpha - gamma) * beta)
    root = torch.sqrt(torch.clamp(1 - (alpha - gamma) * beta * r2, min=0.0))
    k = (1 - alpha * alpha * beta * r2) / (alpha * root + gamma)
    return _normalize(mx, my, k), ok


@dataclasses.dataclass(frozen=True)
class ThinPrismFisheye(_Base):
    """COLMAP `THIN_PRISM_FISHEYE` (ETH3D DSLR cameras), coefficients in COLMAP order `k1 k2 p1 p2 k3 k4 sx1 sy1`.

    Equidistant `(u, v) = theta (x, y) / r`, then `u (1 + k1 r^2 + .. + k4 r^8) + 2 p1 u v + p2 (r^2 + 2 u^2) +
    sx1 r^2` (and `v` likewise) with `r^2 = u^2 + v^2`: unlike `Fisheye624` the tangential and prism terms act on the
    undistorted angles.
    """

    _: dataclasses.KW_ONLY
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    sx1: float = 0.0
    sy1: float = 0.0

    model_id: ClassVar[str] = "thin-prism-fisheye"
    coefficient_names: ClassVar[tuple[str, ...]] = ("k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1")

    def _distort(self, u: torch.Tensor, v: torch.Tensor) -> Pair:
        r2 = u * u + v * v
        radial = 1 + r2 * (self.k1 + r2 * (self.k2 + r2 * (self.k3 + r2 * self.k4)))
        uv = u * v
        du = u * radial + 2 * self.p1 * uv + self.p2 * (r2 + 2 * u * u) + self.sx1 * r2
        dv = v * radial + 2 * self.p2 * uv + self.p1 * (r2 + 2 * v * v) + self.sy1 * r2
        return du, dv

    def _project_normalized(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple[Pair, torch.Tensor]:
        radial = (self.k1, self.k2, self.k3, self.k4)
        theta, (ux, uy) = _fisheye_angles(x, y, z)
        return self._distort(theta * ux, theta * uy), theta <= _radial_domain(radial)[0]

    def _unproject_normalized(self, mx: torch.Tensor, my: torch.Tensor) -> Pair:
        u, v, ok = _invert_2d(self._distort, mx, my)
        theta = torch.hypot(u, v)
        return _ray_from_angle(theta, u, v), ok & (theta <= _radial_domain((self.k1, self.k2, self.k3, self.k4))[0])


@dataclasses.dataclass(frozen=True)
class EUCM(_Base):
    """Enhanced unified camera model (`"eucm"`), parameters `alpha` in [0, 1] and `beta` > 0 (Kalibr `eucm`)."""

    _: dataclasses.KW_ONLY
    alpha: float
    beta: float

    model_id: ClassVar[str] = "eucm"
    coefficient_names: ClassVar[tuple[str, ...]] = ("alpha", "beta")

    def __post_init__(self) -> None:
        """Check the model parameters' ranges."""
        super().__post_init__()
        if not (0 <= self.alpha <= 1 and self.beta > 0):
            msg = f"EUCM needs 0 <= alpha <= 1 and beta > 0, got alpha={self.alpha}, beta={self.beta}"
            raise InputError(msg)

    def _project_normalized(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple[Pair, torch.Tensor]:
        return _eucm_project(x, y, z, self.alpha, self.beta)

    def _unproject_normalized(self, mx: torch.Tensor, my: torch.Tensor) -> Pair:
        return _eucm_unproject(mx, my, self.alpha, self.beta)


@dataclasses.dataclass(frozen=True)
class UCM(_Base):
    """Unified camera model of Mei & Rives (`"ucm"`, Kalibr `omni` without distortion), mirror parameter `xi`.

    `u = fx x / (z + xi |p|) + cx`; `xi = 0` is a pinhole camera.
    """

    _: dataclasses.KW_ONLY
    xi: float

    model_id: ClassVar[str] = "ucm"
    coefficient_names: ClassVar[tuple[str, ...]] = ("xi",)

    def __post_init__(self) -> None:
        """Check the model parameters' ranges."""
        super().__post_init__()
        if self.xi < 0:
            msg = f"UCM needs xi >= 0, got {self.xi}"
            raise InputError(msg)

    # The same model as EUCM with beta = 1, alpha = xi / (1 + xi) and the normalized plane scaled by 1 + xi.
    def _project_normalized(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple[Pair, torch.Tensor]:
        (mx, my), ok = _eucm_project(x, y, z, self.xi / (1 + self.xi), 1.0)
        return (mx / (1 + self.xi), my / (1 + self.xi)), ok

    def _unproject_normalized(self, mx: torch.Tensor, my: torch.Tensor) -> Pair:
        return _eucm_unproject(mx * (1 + self.xi), my * (1 + self.xi), self.xi / (1 + self.xi), 1.0)


@dataclasses.dataclass(frozen=True)
class DoubleSphere(_Base):
    """Double sphere model of Usenko et al. 2018 (`"double-sphere"`, Kalibr `ds`), parameters `xi` and `alpha`."""

    _: dataclasses.KW_ONLY
    xi: float
    alpha: float

    model_id: ClassVar[str] = "double-sphere"
    coefficient_names: ClassVar[tuple[str, ...]] = ("xi", "alpha")

    def __post_init__(self) -> None:
        """Check the model parameters' ranges."""
        super().__post_init__()
        if not 0 <= self.alpha <= 1:
            msg = f"DoubleSphere needs 0 <= alpha <= 1, got {self.alpha}"
            raise InputError(msg)

    def _project_normalized(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple[Pair, torch.Tensor]:
        xi, alpha = self.xi, self.alpha
        d1 = torch.sqrt(x * x + y * y + z * z)
        k = xi * d1 + z
        d2 = torch.sqrt(x * x + y * y + k * k)
        norm = alpha * d2 + (1 - alpha) * k
        w1 = (1 - alpha) / alpha if alpha > 0.5 else alpha / (1 - alpha)  # noqa: PLR2004
        w2 = (w1 + xi) / math.sqrt(2 * w1 * xi + xi * xi + 1)
        return (x / norm, y / norm), (z > -w2 * d1) & (norm > _EPS)

    def _unproject_normalized(self, mx: torch.Tensor, my: torch.Tensor) -> Pair:
        xi, alpha = self.xi, self.alpha
        r2 = mx * mx + my * my
        ok = torch.isfinite(r2)
        if alpha > 0.5:  # noqa: PLR2004
            ok &= r2 <= 1 / (2 * alpha - 1)
        mz = (1 - alpha * alpha * r2) / (alpha * torch.sqrt(torch.clamp(1 - (2 * alpha - 1) * r2, min=0.0)) + 1 - alpha)
        k = (mz * xi + torch.sqrt(torch.clamp(mz * mz + (1 - xi * xi) * r2, min=0.0))) / (mz * mz + r2)
        return _normalize(k * mx, k * my, k * mz - xi), ok


@dataclasses.dataclass(frozen=True)
class FOV(_Base):
    """Field-of-view model of Devernay & Faugeras (`"fov"`, Kalibr `pinhole` + `fov`), parameter `w` (radians).

    `r_d = atan2(2 r tan(w / 2), z) / w`; valid up to `r_d w < pi`, so it covers fields of view above 180 degrees.
    """

    _: dataclasses.KW_ONLY
    w: float

    model_id: ClassVar[str] = "fov"
    coefficient_names: ClassVar[tuple[str, ...]] = ("w",)

    def __post_init__(self) -> None:
        """Check the model parameters' ranges."""
        super().__post_init__()
        if not 0 < self.w < math.pi:
            msg = f"FOV needs 0 < w < pi, got {self.w}"
            raise InputError(msg)

    def _project_normalized(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple[Pair, torch.Tensor]:
        r = torch.hypot(x, y)
        _, (ux, uy) = _fisheye_angles(x, y, z)
        rd = torch.atan2(2 * math.tan(self.w / 2) * r, z) / self.w
        return (rd * ux, rd * uy), rd * self.w < _MAX_THETA

    def _unproject_normalized(self, mx: torch.Tensor, my: torch.Tensor) -> Pair:
        rd = torch.hypot(mx, my)
        angle = rd * self.w
        scale = torch.where(rd <= _EPS, self.w, torch.sin(angle) / torch.where(rd <= _EPS, 1.0, rd))
        z = 2 * math.tan(self.w / 2) * torch.cos(angle)
        return _normalize(mx * scale, my * scale, z), angle < _MAX_THETA


# E57 image representations (ASTM E2807; ply-visualizer `e57-*`): fixed axes, no coefficients. They project and
# unproject like the other models but are not perspective cameras looking along +z, so `rectify` rejects them.
@dataclasses.dataclass(frozen=True)
class E57Pinhole(_Base):
    """E57 pinhole image (`"e57-pinhole"`): +x right, +y down, the camera looks along -z.

    `u = cx + fx x / (-z)`, `v = cy + fy y / (-z)`; points with `z >= 0` are behind it.
    """

    model_id: ClassVar[str] = "e57-pinhole"

    def _project_normalized(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple[Pair, torch.Tensor]:
        return (x / -z, y / -z), z < -_EPS

    def _unproject_normalized(self, mx: torch.Tensor, my: torch.Tensor) -> Pair:
        return _normalize(mx, my, -torch.ones_like(mx)), torch.isfinite(mx) & torch.isfinite(my)


@dataclasses.dataclass(frozen=True)
class E57Spherical(_Base):
    """E57 spherical panorama (`"e57-spherical"`): `u = cx - fx azimuth`, `v = cy - fy elevation` (radians).

    Azimuth `atan2(y, x)` around +z (up), elevation `atan2(z, hypot(x, y))`; points on the z axis are invalid.
    """

    model_id: ClassVar[str] = "e57-spherical"

    def _project_normalized(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple[Pair, torch.Tensor]:
        rho = torch.hypot(x, y)
        return (-torch.atan2(y, x), -torch.atan2(z, rho)), rho > _EPS

    def _unproject_normalized(self, mx: torch.Tensor, my: torch.Tensor) -> Pair:
        horizontal = torch.cos(my)
        ray = torch.stack([horizontal * torch.cos(mx), -horizontal * torch.sin(mx), -torch.sin(my)], -1)
        return ray, torch.isfinite(mx) & torch.isfinite(my)


@dataclasses.dataclass(frozen=True)
class E57Cylindrical(_Base):
    """E57 cylindrical panorama (`"e57-cylindrical"`): `u = cx - fx azimuth`, `v = cy - fy z / hypot(x, y)`."""

    model_id: ClassVar[str] = "e57-cylindrical"

    def _project_normalized(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple[Pair, torch.Tensor]:
        rho = torch.hypot(x, y)
        ok = rho > _EPS
        return (-torch.atan2(y, x), -z / torch.where(ok, rho, 1.0)), ok

    def _unproject_normalized(self, mx: torch.Tensor, my: torch.Tensor) -> Pair:
        return _normalize(torch.cos(mx), -torch.sin(mx), -my), torch.isfinite(mx) & torch.isfinite(my)


MODELS: dict[str, type[_Base]] = {
    cls.model_id: cls
    for cls in (
        Pinhole, PinholeOpenCV, KB4, Fisheye624, ThinPrismFisheye, DoubleSphere, UCM, EUCM, FOV, E57Pinhole,
        E57Spherical, E57Cylindrical,
    )
}  # fmt: skip
_ALIASES = {
    "pinhole-ideal": "pinhole",
    "fisheye-opencv": "kb4",
    "fisheye-equidistant": "kb4",
    "kb3": "kb4",
    "fisheye-kb3": "kb4",
    "ds": "double-sphere",
    "omni": "ucm",
}
# Coefficient counts other than `len(coefficient_names)`: equidistant = KB4 without terms; KB3 (Project Aria
# `KannalaBrandtK3`, ply-visualizer `fisheye-kb3`) = KB4 with the terms named k0..k3, 3 terms padded with 0.
_COUNTS = {"fisheye-equidistant": (0,), "kb3": (3, 4), "fisheye-kb3": (3, 4)}
_PINHOLE_OPENCV_COUNTS = (4, 5, 8, 12, 14)


def camera(  # noqa: PLR0913, PLR0917 - the positional order of calibration files: size, fx fy cx cy, coefficients
    model_id: str, width: int, height: int, fx: float, fy: float, cx: float, cy: float, *coefficients: float
) -> Camera:
    """Build a camera from a model id and its coefficients in the usual file order.

    Args:
        model_id: One of `"pinhole"`, `"pinhole-opencv"` (4, 5, 8, 12 or 14 coefficients in OpenCV order
            `k1 k2 p1 p2 [k3 [k4 k5 k6 [s1 s2 s3 s4 [tx ty]]]]`), `"kb4"` (`k1..k4`), `"fisheye624"` (12),
            `"thin-prism-fisheye"` (8, COLMAP order),
            `"double-sphere"` (`xi alpha`), `"ucm"` (`xi`), `"eucm"` (`alpha beta`), `"fov"` (`w`), `"e57-pinhole"`,
            `"e57-spherical"`, `"e57-cylindrical"` (none), or an alias (`"pinhole-ideal"`, `"fisheye-opencv"`,
            `"fisheye-equidistant"` (none), `"kb3"` / `"fisheye-kb3"` (`k0..k3`, or 3 terms), `"ds"`, `"omni"`).
        width: Image width in pixels.
        height: Image height in pixels.
        fx: Focal length in x, pixels.
        fy: Focal length in y, pixels.
        cx: Principal point x, pixels.
        cy: Principal point y, pixels.
        *coefficients: Model coefficients.

    Returns:
        The camera dataclass of that model.

    Raises:
        InputError: Unknown model id, wrong number of coefficients or invalid values.
    """
    key = _ALIASES.get(model_id, model_id)
    cls = MODELS.get(key)
    if cls is None:
        msg = f"unknown camera model {model_id!r}; known: {sorted([*MODELS, *_ALIASES])}"
        raise InputError(msg)
    names = cls.coefficient_names
    expected = _PINHOLE_OPENCV_COUNTS if cls is PinholeOpenCV else _COUNTS.get(model_id, (len(names),))
    if len(coefficients) not in expected:
        msg = f"camera model {model_id!r} takes {' or '.join(map(str, expected))} coefficients, got {len(coefficients)}"
        raise InputError(msg)
    kwargs = {n: float(c) for n, c in zip(names, coefficients, strict=False)}
    return cls(int(width), int(height), float(fx), float(fy), float(cx), float(cy), **kwargs)
