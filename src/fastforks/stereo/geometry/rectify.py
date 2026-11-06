"""Stereo rectification for any camera model: Bouguet rotations, rectified pinhole intrinsics, remap maps.

`rectify(rig)` computes everything that depends only on the calibration (rotations `R1`/`R2`, projections
`P1`/`P2`, `Q`, regions of interest). The per-pixel maps are built on first use, once per device, and reused by
every `remap` / `unrectify` call (`Rectification.prepare` builds them ahead of time). `rectify` of a rig equal to one
whose `Rectification` is still alive returns that object, so its maps are reused too.

The math follows `cv2.stereoRectify` (Bouguet's algorithm, `CALIB_ZERO_DISPARITY`) but never inverts a
distortion model by hand: every rectified pixel is turned into a ray `R_i^T K_new^-1 p` and projected with the
camera's own `project`, and the image borders are sampled with the camera's `unproject`. That makes one code path
work for every model, fisheyes included; rays more than `fov_max / 2` off the rectified axis are clamped when the
new focal length and principal point are chosen, so a 200-degree fisheye still gets a finite pinhole view.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import math
import numbers
import threading
import weakref
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

from fastforks._core.errors import InputError
from fastforks._core.sampling import grid_sample
from fastforks.stereo.geometry.cameras import hw_size
from fastforks.stereo.geometry.rig import (
    ArrayT,
    Frame,
    PointCloud,
    RectifiedRig,
    StereoRig,
    View,
    _back,
    _check_colors,
    _check_size,
    as_tensor,
    side_of,
)

if TYPE_CHECKING:
    from fastforks.stereo.geometry.cameras import Camera

_GRID = 9  # border samples per side, as cv::getRectangles (exact OpenCV parity for pinhole models)
_GRID_WIDE = 33  # non-pinhole models: curved borders need denser samples for a tight inner rectangle
_MIN_GAP = 1e-9  # px: the valid area must extend past the principal point on every side
_CHUNK = 1 << 20  # pixels per map-building step (bounds the temporaries: ~100 MB per step)
_COARSE = 8  # px between the float64 samples of the unrectify map (refined in float32)
_UNRECTIFY_TOL = 1e-2  # px: float32-refined unrectify pixels farther off are solved in float64


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _rotation_from_vector(v: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(v))
    if angle < 1e-15:  # noqa: PLR2004
        return np.eye(3) + _skew(v)
    k = _skew(v / angle)
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def _vector_from_rotation(rot: np.ndarray) -> np.ndarray:
    angle = math.acos(max(-1.0, min(1.0, (np.trace(rot) - 1) / 2)))
    vee = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    if angle < 1e-6:  # noqa: PLR2004
        return vee / 2
    if math.pi - angle > 1e-6:  # noqa: PLR2004
        return vee * angle / (2 * math.sin(angle))
    sym = (rot + np.eye(3)) / 2  # near 180 degrees: the axis is the dominant column of (R + I) / 2
    axis = sym[:, int(np.argmax(np.diag(sym)))]
    return axis / np.linalg.norm(axis) * angle


def bouguet(rot: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rectifying rotations `(R1, R2)` and the rectified translation `R2 @ T` of a right-from-left pose.

    Each camera turns half of the relative rotation, then both turn so the baseline lies along x (OpenCV's
    `stereoRectify` for horizontal rigs). Depends on the extrinsics only.
    """
    r_half = _rotation_from_vector(_vector_from_rotation(rot) * -0.5)
    tt = r_half @ t
    uu = np.array([1.0 if tt[0] > 0 else -1.0, 0.0, 0.0])
    ww = np.cross(tt, uu)
    nw = np.linalg.norm(ww)
    if nw > 0:
        ww *= math.acos(abs(tt[0]) / np.linalg.norm(tt)) / nw
    w_rot = _rotation_from_vector(ww)
    r1 = w_rot @ r_half.T
    r2 = w_rot @ r_half
    return r1, r2, r2 @ t


def _clamped_projection(rays: torch.Tensor, tan_max: float) -> torch.Tensor:
    """`(x / z, y / z)` with both clamped to `[-tan_max, tan_max]`; rays behind the plane go to the clamp."""
    x, y, z = rays.unbind(-1)
    front = z > 0
    zs = torch.where(front, z, 1.0)
    u = torch.where(front, (x / zs).clamp(-tan_max, tan_max), torch.sign(x) * tan_max)
    v = torch.where(front, (y / zs).clamp(-tan_max, tan_max), torch.sign(y) * tan_max)
    return torch.stack([u, v], -1)


def _border_rays(cam: Camera, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit rays `[n, n, 3]` and valid mask of an `n x n` grid spanning the image (corners included)."""
    h, w = cam.size
    xs = torch.linspace(0, w - 1, n, dtype=torch.float64)
    ys = torch.linspace(0, h - 1, n, dtype=torch.float64)
    grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1)
    rays, valid = cam.unproject(grid)
    return rays, valid


def _rectangles(
    rays: torch.Tensor, valid: torch.Tensor, tan_max: float, fc: float, cc0: np.ndarray
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Inner and outer rectangles `(x, y, w, h)` of the border samples in rectified pixels (`cv::getRectangles`).

    Generalized to wide-angle cameras: per row and column the outermost valid sample stands in for the border one
    (outside a fisheye's image circle), and a sample clamped to the `fov_max` cone on an axis does not shrink the
    inner rectangle on that axis; the cone's edge bounds it instead (rays behind the camera may even change side
    under the rectifying rotation).
    """
    front = rays[..., 2] > 0
    free = front[..., None] & ((rays[..., :2] / torch.where(front, rays[..., 2], 1.0)[..., None]).abs() <= tan_max)
    pts = _clamped_projection(rays, tan_max) * fc + torch.from_numpy(cc0)
    lo, hi = (cc0 - tan_max * fc).tolist(), (cc0 + tan_max * fc).tolist()
    x, y = pts[..., 0], pts[..., 1]
    rows, cols = valid.any(1), valid.any(0)
    first_x = valid.int().argmax(1)
    last_x = valid.shape[1] - 1 - valid.flip(1).int().argmax(1)
    first_y = valid.int().argmax(0)
    last_y = valid.shape[0] - 1 - valid.flip(0).int().argmax(0)
    ar_r, ar_c = torch.arange(valid.shape[0]), torch.arange(valid.shape[1])

    def edge(idx: tuple[torch.Tensor, torch.Tensor], axis: int, keep: torch.Tensor, bound: float) -> torch.Tensor:
        return torch.where(free[idx[0], idx[1], axis], pts[idx[0], idx[1], axis], bound)[keep]

    ix0 = float(edge((ar_r, first_x), 0, rows, lo[0]).max())
    ix1 = float(edge((ar_r, last_x), 0, rows, hi[0]).min())
    iy0 = float(edge((first_y, ar_c), 1, cols, lo[1]).max())
    iy1 = float(edge((last_y, ar_c), 1, cols, hi[1]).min())
    inf = torch.tensor(math.inf, dtype=x.dtype)
    ox0 = float(torch.where(valid, x, inf).min())
    ox1 = float(torch.where(valid, x, -inf).max())
    oy0 = float(torch.where(valid, y, inf).min())
    oy1 = float(torch.where(valid, y, -inf).max())
    return (ix0, iy0, ix1 - ix0, iy1 - iy0), (ox0, oy0, ox1 - ox0, oy1 - oy0)


def _intersect(rect: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0 = max(rect[0], 0), max(rect[1], 0)
    x1, y1 = min(rect[0] + rect[2], w), min(rect[1] + rect[3], h)
    return (x0, y0, x1 - x0, y1 - y0) if x1 > x0 and y1 > y0 else (0, 0, 0, 0)


def _check_arguments(
    rig: StereoRig, alpha: float | None, size: tuple[int, int] | None, fov_max: float, focal: float | None
) -> tuple[int, int]:
    """Validate the `rectify` arguments; returns the rectified `(height, width)`."""
    if not isinstance(rig, StereoRig):
        msg = f"rectify needs a StereoRig, got {type(rig).__name__} (already rectified data needs no rectify())"
        raise InputError(msg)
    for cam in (rig.left, rig.right):
        if type(cam).model_id.startswith("e57-"):
            msg = f"{type(cam).model_id} images do not look along +z and cannot be rectified"
            raise InputError(msg)
    if rig.left.size != rig.right.size:
        msg = f"both cameras must have the same image size, got {rig.left.size} and {rig.right.size}"
        raise InputError(msg)
    if alpha is not None and not (isinstance(alpha, numbers.Real) and (0 <= alpha <= 1 or alpha == -1)):
        msg = f"alpha must be in [0, 1], -1 or None, got {alpha!r}"
        raise InputError(msg)
    if alpha is not None and focal is not None:
        msg = "give alpha or focal, not both (focal fixes the focal length that alpha would choose)"
        raise InputError(msg)
    if focal is not None and not (isinstance(focal, numbers.Real) and 0 < focal < math.inf):
        msg = f"focal must be a positive finite number, got {focal!r}"
        raise InputError(msg)
    if not 0 < fov_max < 180:  # noqa: PLR2004
        msg = f"fov_max must be in (0, 180) degrees, got {fov_max}"
        raise InputError(msg)
    return rig.left.size if size is None else hw_size(size, rig.left.size)


def _rotated_border(cam: Camera, rot: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Border sample rays of a camera in its rectified frame, and their valid mask."""
    pinhole = type(cam).model_id in ("pinhole", "pinhole-opencv")
    rays, valid = _border_rays(cam, _GRID if pinhole else _GRID_WIDE)
    if pinhole and not bool(valid[[0, -1]].all() and valid[:, [0, -1]].all()):
        rays, valid = _border_rays(cam, _GRID_WIDE)  # border past the lens model's fold: sample it as a wide lens
    if not bool(valid.any()):
        msg = f"the {type(cam).model_id} camera has no valid pixels (check its intrinsics and distortion)"
        raise InputError(msg)
    return torch.where(valid[..., None], rays, 0.0) @ torch.from_numpy(rot).T, valid


def _centre(rays: torch.Tensor, valid: torch.Tensor, fc: float, tan_max: float, size: tuple[int, int]) -> np.ndarray:
    """Principal point that puts the mean of the four rectified image corners on the image centre (OpenCV).

    Corners outside a fisheye's image circle (or past a lens model's distortion fold) are replaced by all valid
    samples of the image border, or by all valid samples if no border sample is valid.
    """
    corners = rays[[0, 0, -1, -1], [0, -1, 0, -1]]
    if not bool(valid[[0, 0, -1, -1], [0, -1, 0, -1]].all()):
        ring = torch.ones_like(valid)
        ring[1:-1, 1:-1] = False
        corners = rays[ring & valid] if bool((ring & valid).any()) else rays[valid]
    mean = (_clamped_projection(corners, tan_max) * fc).mean(0).numpy()
    return (np.array(size) - 1) / 2 - mean


def _scale(
    rects: list[tuple[tuple[float, ...], tuple[float, ...]]],
    cc0: np.ndarray,
    cc: np.ndarray,
    size: tuple[int, int],
    alpha: float,
) -> float:
    """Focal length scale: 0 = only valid pixels (inner rectangles), 1 = all source pixels (outer rectangles)."""
    (cx0, cy0), (cx, cy), (nw, nh) = cc0, cc, size

    def bound(rect: tuple[float, ...]) -> tuple[float, float, float, float]:
        gaps = (cx0 - rect[0], cy0 - rect[1], rect[0] + rect[2] - cx0, rect[1] + rect[3] - cy0)
        if min(gaps) <= _MIN_GAP:
            msg = (
                "the valid image area does not surround the rectified principal point (it only touches the image"
                " edges); use alpha=1 or the default focal"
            )
            raise InputError(msg)
        return (cx / gaps[0], cy / gaps[1], (nw - 1 - cx) / gaps[2], (nh - 1 - cy) / gaps[3])

    s0 = max(max(bound(inner)) for inner, _ in rects)
    s1 = min(min(bound(outer)) for _, outer in rects)
    return s0 * (1 - alpha) + s1 * alpha


def rectify(
    rig: StereoRig,
    *,
    size: tuple[int, int] | None = None,
    focal: float | None = None,
    alpha: float | None = None,
    fov_max: float = 120.0,
) -> Rectification:
    """Compute the rectification of a calibrated stereo rig (no image is touched yet).

    Args:
        rig: The calibrated rig; the right camera must sit to the right of the left one.
        size: `(height, width)` of the rectified images; default: the source image size. Swapping portrait and
            landscape against the cameras raises (a `(width, height)` mix-up).
        focal: Focal length of the rectified pair in pixels of the rectified size, with the principal point at the
            image centre. Default (neither `focal` nor `alpha`): the left camera's `fx` at the rectified size, the
            same rectification as `model(left, right, rig=rig)`.
        alpha: Instead of `focal`, choose the focal length and principal point as OpenCV's `alpha`: 0 keeps only
            valid pixels in both rectified images, 1 keeps every source pixel (black borders), values in between
            interpolate; -1 keeps the mean focal length of the source cameras (OpenCV's `alpha=-1`).
        fov_max: Largest field of view in degrees (horizontal and vertical) that border rays are clamped to when the
            new focal length and principal point are chosen; only wide-angle cameras or strongly rotated rigs reach
            it (border rays turned past `fov_max / 2`). There `P1`/`P2` differ from OpenCV's by design (OpenCV's
            focal length shrinks towards 0); raise `fov_max` to follow OpenCV further.

    Returns:
        A `Rectification` holding `R1, R2, P1, P2, Q`, the regions of interest and the lazily built maps. While one
        is alive, a call with the same calibration and arguments returns that same object (and its maps).

    Raises:
        InputError: Bad arguments, a vertical rig, a forward-moving rig (baseline near an optical axis), swapped
            cameras, an E57 camera, or a camera without valid border pixels.
    """
    nh, nw = _check_arguments(rig, alpha, size, fov_max, focal)
    focal = None if focal is None else float(focal)
    alpha = None if alpha is None else float(alpha)
    key = (
        rig.left,
        rig.right,
        np.asarray(rig.R).tobytes(),
        np.asarray(rig.T).tobytes(),
        alpha,
        (nh, nw),
        fov_max,
        focal,
    )
    with _ALIVE_LOCK:
        found = _ALIVE.get(key)
    if found is not None:
        return found
    rect = _rectify(rig, alpha, (nh, nw), fov_max, focal)
    with _ALIVE_LOCK:
        return _ALIVE.setdefault(key, rect)


_ALIVE: weakref.WeakValueDictionary[tuple, Rectification] = weakref.WeakValueDictionary()  # by calibration content
_ALIVE_LOCK = threading.Lock()


_log = logging.getLogger(__name__)
_MAX_TURN = 20.0  # degrees between the optical axes; garden pairs at 25 and 35 degrees gave wrong depth
_NARROW = {"pinhole", "pinhole-opencv", "e57-pinhole"}  # the turn check's models: fisheye / 360 views turn freely
_MIN_VALID = 0.1  # fraction of the rectified image that must be valid


def _check_horizontal(rig: StereoRig) -> None:
    """Raise `InputError` for a baseline that is not mostly along x (vertical or forward), with its direction."""
    if abs(rig.T[1]) > abs(rig.T[0]):
        t_dir = np.asarray(rig.T) / np.linalg.norm(rig.T)
        nums = f"baseline direction (x, y, z) = ({t_dir[0]:.2f}, {t_dir[1]:.2f}, {t_dir[2]:.2f})"
        if abs(rig.T[2]) > abs(rig.T[1]):
            msg = f"the baseline is mostly along the optical axis (forward motion; {nums}): pick another camera pair"
        else:
            msg = (
                f"vertical stereo rigs are not supported ({nums}): rotate both images by 90 degrees and the"
                " calibration with them"
            )
        raise InputError(msg)


def _warn_if_poor(rig: StereoRig, rois: list, size: tuple[int, int]) -> None:
    """Warn when pinhole cameras are turned far against each other or little of the rectified image is valid.

    A fisheye or 360 image can be viewed in any direction its field of view covers; for those only the valid fraction
    counts.
    """
    angle = math.degrees(math.acos(min(1.0, max(-1.0, (float(np.trace(np.asarray(rig.R))) - 1) / 2))))
    narrow = all(getattr(cam, "model_id", "pinhole") in _NARROW for cam in (rig.left, rig.right))
    valid = min(r[2] * r[3] for r in rois) / (size[0] * size[1])
    if (narrow and angle > _MAX_TURN) or valid < _MIN_VALID:
        _log.warning(
            "poor rectification: the cameras are turned %.0f degrees against each other and %.0f%% of the rectified"
            " image is valid; disparities may exceed the model's range and the depth be wrong: use a closer pair, a"
            " smaller size or a smaller focal",
            angle,
            100 * valid,
        )


def _rectify(
    rig: StereoRig, alpha: float | None, size: tuple[int, int], fov_max: float, focal: float | None
) -> Rectification:
    nh, nw = size
    _check_horizontal(rig)
    # Planar rectification looks perpendicular to the baseline, at most fov_max / 2 off that plane: a baseline closer
    # than 90 - fov_max / 2 degrees to an optical axis (forward motion, the epipole inside the image) leaves no
    # rectified pixel with a source.
    t_norm = float(np.linalg.norm(rig.T))
    tilt = max(abs(float(rig.T[2])), abs(float((np.asarray(rig.R).T @ rig.T)[2]))) / t_norm if t_norm else 0.0
    if tilt > math.sin(math.radians(fov_max) / 2):
        angle = math.degrees(math.acos(min(tilt, 1.0)))
        msg = (
            f"the baseline is only {angle:.0f} degrees off an optical axis (forward motion: the epipole lies in the"
            f" image); planar rectification needs more than {90 - fov_max / 2:g} degrees (fov_max={fov_max:g})"
        )
        raise InputError(msg)
    r1, r2, t = bouguet(rig.R, rig.T)
    if t[0] > 0:
        msg = "the right camera is left of the left camera (T_x > 0 after rectification); swap the two cameras"
        raise InputError(msg)
    source = rig
    if rig.left.size != (nh, nw):  # everything below runs on the output grid, so no scale is applied twice
        rig = rig.resized((nh, nw))
    w, h = nw, nh
    tan_max = math.tan(math.radians(fov_max) / 2)
    if focal is None and alpha is None:  # the default: keep the left camera's focal length
        focal = rig.left.fx
    fc = (rig.left.fy + rig.right.fy) / 2 if focal is None else focal
    samples = [_rotated_border(cam, rot) for cam, rot in ((rig.left, r1), (rig.right, r2))]
    for _, valid in samples:
        if not bool(valid.any()):
            msg = "no border sample of the image unprojects to a valid ray; check the camera model and size"
            raise InputError(msg)
    (rays_l, valid_l), (rays_r, valid_r) = samples
    cc0 = (_centre(rays_l, valid_l, fc, tan_max, (w, h)) + _centre(rays_r, valid_r, fc, tan_max, (w, h))) / 2
    rects = [_rectangles(rays, valid, tan_max, fc, cc0) for rays, valid in samples]
    # CALIB_ZERO_DISPARITY: one principal point for both images. Pixel centres sit at integer coordinates, so a
    # resize maps c to (c + 0.5) s - 0.5 (cv2.stereoRectify uses c s, which is off by 0.5 (s - 1) px); the resized
    # cameras above already follow that rule.
    cc = cc0
    if focal is not None:  # a chosen focal length: the principal point is the canvas centre, whatever the rig's borders
        cc = np.array([(nw - 1) / 2, (nh - 1) / 2])
    s = 1.0 if alpha is None or alpha == -1 else _scale(rects, cc0, cc, (nw, nh), alpha)
    fc *= s
    rois = []
    for inner, _ in rects:
        x, y = np.ceil((np.array(inner[:2]) - cc0) * s + cc).astype(int).tolist()
        rois.append(_intersect((x, y, math.floor(inner[2] * s), math.floor(inner[3] * s)), nw, nh))
    cx, cy = float(cc[0]), float(cc[1])
    p1 = np.array([[fc, 0, cx, 0], [0, fc, cy, 0], [0, 0, 1, 0]])
    p2 = p1.copy()
    p2[0, 3] = t[0] * fc
    q = np.array([[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, fc], [0, 0, -1 / t[0], 0]])
    _warn_if_poor(rig, rois, (nh, nw))
    return Rectification(source, (nh, nw), r1, r2, p1, p2, q, rois[0], rois[1])


_NHWC = {2: ((1,), (1,)), 3: ((1,), ()), 4: ((), ())}  # numpy image rank -> padding to [N, H, W, C]


def _device(device: str | torch.device) -> torch.device:
    """`device` with an explicit index (`"cuda"` -> `cuda:<current>`), so the caches see one key per device."""
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def _f64_device(device: torch.device) -> torch.device:
    """Where float64 work for `device` runs: MPS has no float64, so its float64 inverse is done on the CPU."""
    return torch.device("cpu") if device.type == "mps" else device


def _image_nchw(img: np.ndarray | torch.Tensor, what: str) -> tuple[torch.Tensor, Any]:
    """`[N, C, H, W]` view of an image and a function that restores its layout, kind and dtype at the output's size."""
    t, numpy = as_tensor(img, what)
    if t.dtype == torch.bool or t.is_complex():
        msg = f"{what} has unsupported dtype {t.dtype}"
        raise InputError(msg)
    shape, dtype = t.shape, t.dtype
    if numpy and t.ndim in _NHWC:  # [H, W], [H, W, C] or [B, H, W, C]
        lead, tail = _NHWC[t.ndim]
        x = t.reshape(*lead, *t.shape, *tail).permute(0, 3, 1, 2)
    elif not numpy and t.ndim in (2, 3, 4):
        x = t.reshape(-1, *t.shape[-3:]) if t.ndim == 4 else t.reshape(1, -1, *t.shape[-2:])  # noqa: PLR2004
    else:
        layout = "[H, W], [H, W, C] or [B, H, W, C]" if numpy else "[H, W], [C, H, W] or [B, C, H, W]"
        msg = f"{what} must be {layout}, got shape {tuple(shape)}"
        raise InputError(msg)

    def restore(out: torch.Tensor) -> np.ndarray | torch.Tensor:
        if not dtype.is_floating_point:
            info = torch.iinfo(dtype)
            out = out.round_().clamp_(info.min, info.max)
        out = out.to(dtype)
        if numpy:  # the input's layout at the output's size (a `size=` rectification changes H and W)
            h = 1 if len(shape) == 4 else 0  # noqa: PLR2004 - [B, H, W, C]
            out = out.permute(0, 2, 3, 1).reshape(*shape[:h], *out.shape[-2:], *shape[h + 2 :])
            if out.device.type == "cuda":  # via pinned memory: ~8x faster than a pageable copy, the buffer is reused
                host = torch.empty(out.shape, dtype=out.dtype, pin_memory=True)
                return host.copy_(out).numpy().copy()
            return out.contiguous().cpu().numpy()
        return out.reshape(*shape[:-2], *out.shape[-2:])

    return x, restore


def _freeze(rect: Rectification) -> None:
    """Store `R1`, `R2`, `P1`, `P2`, `Q` as read-only float64 arrays (copies if they are not already that)."""
    for name in ("R1", "R2", "P1", "P2", "Q"):
        a = getattr(rect, name)
        if not (isinstance(a, np.ndarray) and a.dtype == np.float64 and not a.flags.writeable):
            a = np.array(a, dtype=np.float64)  # a copy: a caller's array stays theirs
            a.flags.writeable = False
            object.__setattr__(rect, name, a)


@dataclasses.dataclass(frozen=True, eq=False)
class Rectification:
    """The rectification of a `StereoRig`, as returned by `rectify`.

    Attributes:
        rig: The source rig.
        size: `(height, width)` of the rectified images.
        R1: `[3, 3]` rotation from the left camera frame to the rectified left frame.
        R2: `[3, 3]` rotation from the right camera frame to the rectified right frame.
        P1: `[3, 4]` projection of the rectified left camera.
        P2: `[3, 4]` projection of the rectified right camera (`P2[0, 3] = -f * baseline`).
        Q: `[4, 4]` disparity-to-depth matrix (OpenCV convention).
        roi_left: `(x, y, w, h)` of the rectified left image where every pixel is valid (with `alpha=0` close to
            the full image).
        roi_right: The same for the right image.

    The matrices are read-only float64 arrays (the object is shared by every call with the same calibration);
    `.copy()` one to edit it.
    """

    rig: StereoRig
    size: tuple[int, int]
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    roi_left: tuple[int, int, int, int]
    roi_right: tuple[int, int, int, int]
    _cache: dict[tuple[str, str], Any] = dataclasses.field(default_factory=dict, repr=False)
    _lock: threading.RLock = dataclasses.field(default_factory=threading.RLock, repr=False)  # guards `_cache` fills

    def __post_init__(self) -> None:
        """Make the matrices read-only: one `Rectification` is shared by every call with the same calibration."""
        _freeze(self)

    def __getstate__(self) -> dict[str, Any]:
        """Pickle / deepcopy without the lock and the maps (rebuilt on first use)."""
        state = self.__dict__.copy()
        del state["_lock"]
        state["_cache"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a pickled `Rectification` with a fresh lock."""
        self.__dict__.update(state, _lock=threading.RLock())
        _freeze(self)  # unpickled arrays are writable again

    @property
    def rectified(self) -> RectifiedRig:
        """The rectified pair as a `RectifiedRig` (focal length, principal point, baseline)."""
        f = float(self.P1[0, 0])
        return RectifiedRig(f, -float(self.P2[0, 3]) / f, float(self.P1[0, 2]), float(self.P1[1, 2]), 0.0, self.size)

    @property
    def f(self) -> float:
        """Focal length of the rectified cameras, pixels."""
        return float(self.P1[0, 0])

    @property
    def baseline(self) -> float:
        """Baseline of the rectified pair, in the unit of `rig.T`."""
        return -float(self.P2[0, 3]) / float(self.P1[0, 0])

    def _build(self, device: torch.device, *, grid: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Source pixel of every rectified pixel in one pass: `(maps or grid [2, H, W, 2] float32, valid [2, H, W])`.

        `grid=False` gives source pixels (-1 where invalid), `grid=True` the `grid_sample` coordinates in [-1, 1]
        (-2 where invalid). The rectified rays `R^T ((u - cx) / f, (v - cy) / f, 1)` are float32 sums of per-row and
        per-column terms computed in float64, so the maps differ from a float64 build by <= 3e-4 px at 1080p and
        <= 8e-4 px at 12 MP (the float32 storage alone is ~1e-4 px at 4000 px), at a fraction of the float64 cost.
        """
        nh, nw = self.size
        out = torch.empty(2, nh, nw, 2, dtype=torch.float32, device=device)
        valid = torch.empty(2, nh, nw, dtype=torch.bool, device=device)
        rows = max(1, _CHUNK // nw)
        f, cx, cy = self.f, float(self.P1[0, 2]), float(self.P1[1, 2])
        cols = torch.from_numpy((np.arange(nw) - cx) / f).float().to(device)
        for i, (cam, rot) in enumerate(zip((self.rig.left, self.rig.right), (self.R1, self.R2), strict=True)):
            h, w = cam.size
            for v0 in range(0, nh, rows):
                v1 = min(v0 + rows, nh)
                row = torch.from_numpy((np.arange(v0, v1) - cy) / f).float().to(device)[:, None]
                # ray = R^T (col, row, 1): component k is col R[0, k] + row R[1, k] + R[2, k]
                x, y, z = (cols * float(rot[0, k]) + (row * float(rot[1, k]) + float(rot[2, k])) for k in range(3))
                (mx, my), ok = cam._project_normalized(x, y, z)  # noqa: SLF001
                src = out[i, v0:v1]
                sx, sy = src.unbind(-1)
                sx.copy_(mx).mul_(cam.fx).add_(cam.cx)
                sy.copy_(my).mul_(cam.fy).add_(cam.cy)
                ok = ok & (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)  # also drops NaN
                if grid:  # grid_sample coordinates in [-1, 1] (align_corners=True)
                    sx.mul_(2 / (w - 1)).sub_(1)
                    sy.mul_(2 / (h - 1)).sub_(1)
                src.masked_fill_(~ok[..., None], -2.0 if grid else -1.0)
                valid[i, v0:v1] = ok
        self._cache["valid", str(device)] = valid  # kept when the maps are dropped
        return out, valid

    def _maps(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = ("maps", str(device))
        with self._lock:
            if key not in self._cache:
                self._cache[key] = self._build(device, grid=False)
            return self._cache[key]

    def _valid(self, device: torch.device) -> torch.Tensor:
        """`[2, H, W]` mask of the rectified pixels with a source pixel (built with the maps, kept without them)."""
        key = ("valid", str(device))
        with self._lock:
            if key not in self._cache:
                self._maps(device)
                del self._cache["maps", str(device)]
            return self._cache[key]

    def maps(self, device: str | torch.device = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """Per-pixel source coordinates of both rectified images, built once per device and cached.

        Args:
            device: Where to build and keep the maps.

        Returns:
            `(maps [2, H, W, 2], valid [2, H, W])`: float32 source `(x, y)` pixel for every rectified pixel of the
            left (index 0) and right (1) image, -1 where not valid. `cv2.remap(img, maps[0, ..., 0].numpy(),
            maps[0, ..., 1].numpy(), cv2.INTER_LINEAR)` gives the same image as `remap`.
        """
        return self._maps(_device(device))

    def _grid(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = ("grid", f"{device}/{dtype}")
        with self._lock:
            if key not in self._cache:
                grid, valid = self._build(device, grid=True)
                self._cache[key] = (grid.to(dtype), valid)
            return self._cache[key]

    def prepare(self, device: str | torch.device = "cuda", *, background: bool = False) -> None:
        """Build the maps of `remap` (float32 images), `depth` and `unrectify` on `device` now instead of on first use.

        Args:
            device: Where the maps will be used.
            background: Build them in a thread and return at once; a `remap` / `unrectify` meanwhile waits for
                the map it needs (the maps are built under a lock), so the first pair can be decoded in parallel.
        """
        dev = _device(device)

        def build() -> None:
            with torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext():
                self._grid(dev, torch.float32)
                self._valid(dev)
                self._unrectify_map(dev)

        if background:
            # not a daemon: one still inside a torch op at interpreter exit ends in std::terminate (exit code 134)
            threading.Thread(target=build, name="fastforks-rectify-maps").start()
        else:
            build()

    def remap(
        self,
        left: ArrayT,
        right: ArrayT,
        *,
        device: str | torch.device | None = None,
        return_valid: bool = False,
    ) -> tuple[ArrayT, ...]:
        """Rectify a stereo pair (bilinear, both images in one `grid_sample` call).

        Matches `cv2.remap(..., cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)` with the maps of
        `cv2.initUndistortRectifyMap` to within 1 grey level for 8-bit images: pixel centres at integer
        coordinates, pixels mapped outside the source image are 0.

        Args:
            left: Left image, numpy `[H, W]` / `[H, W, C]` or torch `[H, W]` / `[C, H, W]` / `[B, C, H, W]`.
            right: Right image of the same kind and shape.
            device: Where to compute (maps are cached per device). Default: the tensors' device, CPU for numpy.
            return_valid: Also return the `[H, W]` bool masks of rectified pixels with a source pixel.

        Returns:
            `(left, right)` or `(left, right, valid_left, valid_right)`: same kind, layout and dtype as the inputs
            (integer images are rounded); numpy in, numpy out; tensors stay on `device`.
        """
        x_left, restore_left = _image_nchw(left, "left")
        x_right, restore_right = _image_nchw(right, "right")
        for name, x, cam in (("left", x_left, self.rig.left), ("right", x_right, self.rig.right)):
            if tuple(x.shape[-2:]) != cam.size:
                msg = (
                    f"{name} image is {x.shape[-2]}x{x.shape[-1]} (height x width) but the camera is"
                    f" {cam.size[0]}x{cam.size[1]}"
                )
                raise InputError(msg)
        if x_left.shape != x_right.shape:
            msg = f"left and right images differ: {tuple(x_left.shape)} vs {tuple(x_right.shape)}"
            raise InputError(msg)
        dev = _device(device if device is not None else x_left.device)
        dtype = torch.float64 if x_left.dtype == torch.float64 else torch.float32
        grid, valid = self._grid(dev, dtype)
        n = x_left.shape[0]
        x = torch.cat([x_left.to(dev), x_right.to(dev)]).to(dtype)  # transfer before the dtype conversion
        grid = grid if n == 1 else grid.repeat_interleave(n, 0)
        out = grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        result = (restore_left(out[:n]), restore_right(out[n:]))
        if return_valid:
            numpy = isinstance(left, np.ndarray)
            result += (_back(valid[0], numpy), _back(valid[1], numpy))
        return result  # ty: ignore[invalid-return-type]

    def depth(self, disparity: ArrayT, *, euclidean: bool = False, view: View = "left") -> ArrayT:
        """Depth in the rectified frame, `f * baseline / disparity` (see `RectifiedRig.depth`).

        Args:
            disparity: `[..., H, W]` disparity of the rectified pair in pixels.
            euclidean: Distance along each pixel's ray instead of the depth along the optical axis.
            view: Which rectified image the disparity belongs to; selects the validity mask (`"right"`: the
                disparity of the right view, positive as well).

        Returns:
            Depth of the same shape and kind, NaN where the disparity is not positive or the rectified pixel has no
            source pixel (the black border with `alpha > 0`, outside a fisheye's image circle).
        """
        d, numpy = as_tensor(disparity, "disparity")
        z = self.rectified.depth(d, euclidean=euclidean)
        valid = self._valid(_device(z.device))[side_of(view)]
        return _back(z.masked_fill(~valid, torch.nan), numpy)  # ty: ignore[invalid-return-type]

    def point_cloud(
        self, disparity: ArrayT, *, colors: ArrayT | None = None, frame: Frame = "rectified", view: View = "left"
    ) -> PointCloud:
        """3D point of every rectified pixel of one view, in the frame of that view's camera.

        Args:
            disparity: `[..., H, W]` disparity of the rectified pair in pixels.
            colors: Optional rectified image `[..., H, W, C]` or `[..., H, W]` of that view, returned as `colors`.
            frame: `"rectified"` (the rectified camera) or `"camera"` (the original camera of that view: `R^T X`).
            view: `"left"` or `"right"`: the camera the disparity map belongs to. Each cloud is in its own camera
                frame (no merging into a common one).

        Returns:
            `PointCloud(points, valid, colors)` of the same kind as `disparity`; `valid` is also False (and the
            point NaN) where the rectified pixel has no source pixel.
        """
        if frame not in ("rectified", "camera"):
            msg = f"frame must be 'rectified' or 'camera', got {frame!r}"
            raise InputError(msg)
        points, valid, numpy = self.rectified._points(disparity)  # noqa: SLF001
        side = side_of(view)
        valid = valid & self._valid(_device(valid.device))[side]
        points = points.masked_fill(~valid[..., None], torch.nan)
        if frame == "camera":
            points = points @ torch.tensor((self.R1, self.R2)[side], dtype=points.dtype, device=points.device)
        return PointCloud(_back(points, numpy), _back(valid, numpy), _check_colors(colors, valid))

    def _unrectify_map(self, device: torch.device, side: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        key = ("unrectify" if side == 0 else "unrectify-right", str(device))
        with self._lock:
            return self._unrectify_map_locked(key, device, side)

    def _unrectify_map_locked(
        self, key: tuple[str, str], device: torch.device, side: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key not in self._cache:
            h, w = (self.rig.left, self.rig.right)[side].size
            grid = torch.empty(1, h, w, 2, dtype=torch.float32, device=device)
            factor = torch.empty(h, w, dtype=torch.float32, device=device)
            coarse = self._unrectify_coarse(device, side)
            rows = max(1, _CHUNK // w)
            for v0 in range(0, h, rows):
                v1 = min(v0 + rows, h)
                self._unrectify_rows(coarse, v0, v1, grid[0, v0:v1], factor[v0:v1], side)
            self._cache[key] = (grid, factor)
        return self._cache[key]

    def _to_rectified(self, pix: torch.Tensor, side: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Float64 inverse: rectified pixel `[..., 2]`, `z_src / z_rect` and validity of original pixels of a camera."""
        ray, ok = (self.rig.left, self.rig.right)[side].unproject(pix)
        k = self.P1[:, :3] @ (self.R1, self.R2)[side]
        # (z u_rect, z v_rect, z) up to the ray length; elementwise, as a matmul would start cuBLAS on a first call
        x, y, z_rect = (ray[..., 0] * k[i, 0] + ray[..., 1] * k[i, 1] + ray[..., 2] * k[i, 2] for i in range(3))
        ok &= z_rect > 0
        z_safe = torch.where(ok, z_rect, 1.0)
        return torch.stack([x, y], -1) / z_safe[..., None], torch.where(ok, ray[..., 2] / z_safe, 0.0), ok

    def _unrectify_coarse(self, device: torch.device, side: int = 0) -> torch.Tensor:
        """Float64 inverse every `_COARSE` px (one sample past the last pixel): rectified pixels, NaN where invalid."""
        h, w = (self.rig.left, self.rig.right)[side].size
        dev64 = _f64_device(device)
        u = torch.arange(0, w - 1 + 2 * _COARSE, _COARSE, dtype=torch.float64, device=dev64)
        v = torch.arange(0, h - 1 + 2 * _COARSE, _COARSE, dtype=torch.float64, device=dev64)
        uv, _, ok = self._to_rectified(torch.stack(torch.broadcast_tensors(u, v[:, None]), -1), side)
        return torch.where(ok[..., None], uv, torch.nan).float().to(device)

    def _unrectify_rows(  # noqa: PLR0913, PLR0917
        self, coarse: torch.Tensor, v0: int, v1: int, grid: torch.Tensor, factor: torch.Tensor, side: int = 0
    ) -> None:
        """Fill rows `v0:v1` of the unrectify grid and depth factor.

        The coarse float64 inverse is interpolated bilinearly, then refined by two Newton steps on the closed-form
        forward map (rectified pixel -> source pixel) in float32, with the coarse cell's Jacobian; pixels of cells
        with an invalid corner or that do not converge to 1e-2 px are solved in float64 (`_to_rectified`). The grid
        differs from a float64 build by ~1e-4 px at 1080p (float32 storage), well below the nearest-pixel sampling.
        """
        cam = (self.rig.left, self.rig.right)[side]
        (_, w), (nh, nw) = cam.size, self.size
        dev = coarse.device
        u = torch.arange(w, dtype=torch.float32, device=dev)
        v = torch.arange(v0, v1, dtype=torch.float32, device=dev)[:, None]
        ju, jv = (
            torch.div(u, _COARSE, rounding_mode="floor").long(),
            torch.div(v, _COARSE, rounding_mode="floor").long(),
        )
        a, b = (u / _COARSE - ju)[..., None], (v / _COARSE - jv)[..., None]
        q00, q01, q10, q11 = coarse[jv, ju], coarse[jv, ju + 1], coarse[jv + 1, ju], coarse[jv + 1, ju + 1]
        q = (1 - b) * ((1 - a) * q00 + a * q01) + b * ((1 - a) * q10 + a * q11)
        du = ((1 - b) * (q01 - q00) + b * (q11 - q10)) / _COARSE  # d(rectified) / d(source u), per cell
        dv = ((1 - a) * (q10 - q00) + a * (q11 - q01)) / _COARSE
        f, cx, cy = self.f, float(self.P1[0, 2]), float(self.P1[1, 2])
        rot = (self.R1, self.R2)[side]
        r = [[float(rot[i, k]) for k in range(3)] for i in range(3)]
        for _ in range(2):
            x, y = (q[..., 0] - cx) / f, (q[..., 1] - cy) / f
            # source ray R1^T (x, y, 1), elementwise (no float32 matmul: TF32 would cost precision)
            sx, sy, sz = (x * r[0][k] + (y * r[1][k] + r[2][k]) for k in range(3))
            (mx, my), _ = cam._project_normalized(sx, sy, sz)  # noqa: SLF001
            ex, ey = (mx * cam.fx + cam.cx - u), (my * cam.fy + cam.cy - v)  # source-pixel residual
            q = q - du * ex[..., None] - dv * ey[..., None]
        x, y = (q[..., 0] - cx) / f, (q[..., 1] - cy) / f
        sx, sy, sz = (x * r[0][k] + (y * r[1][k] + r[2][k]) for k in range(3))
        (mx, my), ok = cam._project_normalized(sx, sy, sz)  # noqa: SLF001
        residual = torch.hypot(mx * cam.fx + cam.cx - u, my * cam.fy + cam.cy - v)
        slow = ~(residual <= _UNRECTIFY_TOL) | ~ok  # also NaN: a cell with an invalid corner
        fac = sz  # z_source / z_rectified of the point on the rectified ray (x, y, 1)
        uv = q
        idx = slow.nonzero()
        if len(idx):  # float64 for these pixels (borders of the valid region)
            dev64 = _f64_device(dev)
            pix = torch.stack([idx[:, 1].to(dev64).double(), (idx[:, 0] + v0).to(dev64).double()], -1)
            uv64, fac64, ok64 = self._to_rectified(pix, side)
            uv = uv.index_put((idx[:, 0], idx[:, 1]), uv64.float().to(dev))
            fac = fac.index_put((idx[:, 0], idx[:, 1]), fac64.float().to(dev))
            ok = ok.index_put((idx[:, 0], idx[:, 1]), ok64.to(dev))
        ok &= (uv[..., 0] >= 0) & (uv[..., 0] <= nw - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= nh - 1)
        scale = torch.tensor([2 / (nw - 1), 2 / (nh - 1)], dtype=torch.float32, device=dev)
        grid.copy_(torch.where(ok[..., None], uv * scale - 1, -2.0))
        factor.copy_(torch.where(ok, fac, torch.nan))

    def unrectify(
        self,
        depth: ArrayT,
        *,
        device: str | torch.device | None = None,
        euclidean: bool = False,
        view: View = "left",
    ) -> ArrayT:
        """Resample a rectified depth map into the original image of one camera (nearest neighbour, no blending).

        Args:
            depth: `[..., H, W]` depth in the rectified frame (e.g. from `depth`), NaN (or <= 0) where unknown.
            device: Where to compute; default: the tensor's device, CPU for numpy.
            euclidean: `depth` holds distances along the pixel rays (`depth(..., euclidean=True)`); they are the
                same in the rectified and the original frame, so only the pixels are resampled and no z-factor
                is applied. Default: depth along the optical axis, converted to the original camera's axis.
            view: `"left"` (default) or `"right"`: which original camera the map is resampled into; the depth must
                be of that view (right-view disparity from the swapped-and-flipped pass).

        Returns:
            `[..., h, w]` depth along the original camera's optical axis at the original pixels, NaN where
            unknown; same kind as `depth`.
        """
        d, numpy = as_tensor(depth, "depth")
        _check_size(d, self.size, "depth")
        side = side_of(view)
        dev = _device(device if device is not None else d.device)
        grid, factor = self._unrectify_map(dev, side)
        x = d.to(dev, torch.float32).reshape(-1, 1, *d.shape[-2:])
        # rectified pixels without a source pixel carry no depth; holes are NaN
        x = torch.where(self._valid(dev)[side] & (x > 0), x, torch.nan)
        if euclidean:
            factor = torch.where(torch.isnan(factor), torch.nan, 1.0)
        # nearest sampling copies NaN holes; pixels outside the grid read 0 and the NaN factor turns them into holes
        z = F.grid_sample(x, grid.expand(x.shape[0], -1, -1, -1), mode="nearest", align_corners=True)[:, 0] * factor
        z = z.reshape(*d.shape[:-2], *factor.shape)
        return _back(z, numpy)  # ty: ignore[invalid-return-type]
