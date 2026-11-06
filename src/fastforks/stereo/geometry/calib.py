"""`read_calib`: stereo calibration files (KITTI, Middlebury / ETH3D, OpenCV YAML / XML, Kalibr camchain, ZED)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal, overload

import numpy as np

from fastforks._core.errors import InputError
from fastforks.stereo.geometry import _yaml
from fastforks.stereo.geometry.cameras import Camera, camera, hw_size
from fastforks.stereo.geometry.rectify import _rotation_from_vector
from fastforks.stereo.geometry.rig import RectifiedRig, StereoRig, unit_scale

CalibFormat = Literal[
    "kitti", "kitti-unrectified", "kitti-odometry", "middlebury", "opencv", "kalibr", "tum", "zed",
    "oak", "realsense", "ros", "colmap", "euroc", "aria", "nerfstudio",
]  # fmt: skip
FORMATS: tuple[str, ...] = (
    "kitti", "kitti-unrectified", "kitti-odometry", "middlebury", "opencv", "kalibr", "tum", "zed",
    "oak", "realsense", "ros", "colmap", "euroc", "aria", "nerfstudio",
)  # fmt: skip
_DEFAULT_UNIT = {"middlebury": "mm", "zed": "mm", "oak": "cm"}
_DEFAULT_CAMERAS = {
    "kitti": (2, 3),
    "kitti-unrectified": (2, 3),
    "kitti-odometry": (2, 3),
    "kalibr": (0, 1),
    "tum": (0, 1),
    "oak": (1, 2),
}
_KITTI_ROW = r":[ \t]*[-+]?[\d.]"  # KITTI rows are numbers on the key's line (an OpenCV `P1:` holds a matrix node)
_DETECT = (  # (pattern, format), first match wins; OpenCV FileStorage signatures first (its files often hold P1 / P2)
    (re.compile(r"\A\s*<"), "opencv"),
    (re.compile(r'"cameraData"'), "oak"),
    (re.compile(r'"ppx"'), "realsense"),
    (re.compile(r'"CameraCalibrations"'), "aria"),
    (re.compile(r'"transform_matrix"'), "nerfstudio"),
    (re.compile(r"^projection_matrix:", re.MULTILINE), "ros"),
    (re.compile(r"^T_BS:", re.MULTILINE), "euroc"),
    (re.compile(r"\A# Camera list with one line of data per camera"), "colmap"),
    (re.compile(r"\A\s*%YAML|!!opencv-matrix"), "opencv"),
    (re.compile(rf"^P_rect_\d\d{_KITTI_ROW}", re.MULTILINE), "kitti"),
    (re.compile(rf"^K_\d\d{_KITTI_ROW}", re.MULTILINE), "kitti-unrectified"),
    (re.compile(rf"^P[0-3]{_KITTI_ROW}", re.MULTILINE), "kitti-odometry"),
    (re.compile(r"^cam0\s*=", re.MULTILINE), "middlebury"),
    (re.compile(r"^cam0:(?s:.*)camera_model", re.MULTILINE), "kalibr"),
    (re.compile(r"^\[LEFT_CAM_\w+\]", re.MULTILINE), "zed"),
)
# ZED SN<serial>.conf sections: resolution name -> image size (ZED, ZED 2, ZED Mini, ZED X)
_ZED_SIZES = {
    "2K": (2208, 1242), "FHD": (1920, 1080), "FHD1200": (1920, 1200), "HD": (1280, 720), "SVGA": (960, 600),
    "VGA": (672, 376),
}  # fmt: skip
_ZED_DISTORTION = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")  # OpenCV order; missing terms are 0
_OPENCV_KEYS = (  # (left K, left D, right K, right D, R, T) as written by common stereo calibration scripts
    ("M1", "D1", "M2", "D2", "R", "T"),
    ("K1", "D1", "K2", "D2", "R", "T"),
    ("cameraMatrix1", "distCoeffs1", "cameraMatrix2", "distCoeffs2", "R", "T"),
    ("left_camera_matrix", "left_distortion", "right_camera_matrix", "right_distortion", "rotation_matrix",
     "translation_vector"),
)  # fmt: skip
_KALIBR_MODELS = {"pinhole": ("pinhole", 4), "omni": ("ucm", 5), "ds": ("double-sphere", 6), "eucm": ("eucm", 6)}
_KALIBR_DISTORTIONS = {
    "radtan": ("pinhole-opencv", 4),
    "equidistant": ("kb4", 4),
    "fov": ("fov", 1),
    "none": ("pinhole", 0),
}
_OPENCV_SIZE_KEYS = (("image_width", "image_height"), ("width", "height"), ("imageSize",), ("image_size",), ("size",))


def _fail(path: Path, why: str) -> InputError:
    return InputError(f"{path}: {why}")


def _matrix(value: object, shape: tuple[int, ...], what: str, path: Path) -> np.ndarray:
    """A float64 array of `shape` from a nested list / OpenCV matrix node / flat list."""
    if isinstance(value, dict) and "data" in value:
        value = value["data"]
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise _fail(path, f"{what} is not a numeric matrix") from None
    if arr.size != int(np.prod(shape)) or not np.isfinite(arr).all():
        raise _fail(path, f"{what} needs {int(np.prod(shape))} finite numbers {shape}, got {arr.size}")
    return arr.reshape(shape)


def _vector(value: object, what: str, path: Path) -> np.ndarray:
    if isinstance(value, dict) and "data" in value:
        value = value["data"]
    arr = np.asarray(value if value is not None else [], dtype=np.float64).reshape(-1)
    if not np.isfinite(arr).all():
        raise _fail(path, f"{what} has non-finite values")
    return arr


def _opencv_camera(k: np.ndarray, d: np.ndarray, size: tuple[int, int], what: str, *, fisheye: bool = False) -> Camera:
    """Camera from an OpenCV camera matrix and distortion vector (`cv2.fisheye` coefficients if `fisheye`)."""
    if abs(k[0, 1]) > 1e-9 * abs(k[0, 0]) or not np.allclose(k[2], [0, 0, 1]):
        msg = f"{what} camera matrix must be [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] (no skew)"
        raise InputError(msg)
    intrinsics = (size[0], size[1], k[0, 0], k[1, 1], k[0, 2], k[1, 2])
    counts = (4,) if fisheye else (0, 4, 5, 8, 12, 14)
    if len(d) not in counts:
        msg = f"{what} distortion needs {', '.join(map(str, counts))} coefficients, got {len(d)}"
        raise InputError(msg)
    if fisheye:
        return camera("kb4", *intrinsics, *d)
    return camera("pinhole-opencv", *intrinsics, *d) if d.any() else camera("pinhole", *intrinsics)


def _check_rectified(pa: np.ndarray, pb: np.ndarray, path: Path) -> None:
    f = pa[0, 0]
    if f <= 0 or not np.allclose([pa[1, 1], pb[0, 0], pb[1, 1]], f, rtol=1e-6) or abs(pa[1, 2] - pb[1, 2]) > 1e-6 * f:
        raise _fail(path, "the two cameras are not rectified (different focal lengths or principal point rows)")
    if max(abs(pa[0, 1]), abs(pb[0, 1])) > 1e-9 * f:
        raise _fail(path, "the camera matrices have skew (K[0, 1] != 0), which is not supported: remove it")


def _from_projections(pa: np.ndarray, pb: np.ndarray, size: tuple[int, int] | None, path: Path) -> RectifiedRig:
    """`RectifiedRig` from two rectified `[3, 4]` projections `P = K [I | t]` (KITTI's `P_rect`).

    The baseline is `t_x` of each camera, `(P[0, 3] - cx P[2, 3]) / f`: KITTI's projections carry a small `t_z`, which
    the common shortcut `P[0, 3] / f` mixes in (0.18 % on KITTI odometry 00).
    """
    _check_rectified(pa, pb, path)
    f = float(pa[0, 0])
    tx = lambda p: (float(p[0, 3]) - float(p[0, 2]) * float(p[2, 3])) / f  # noqa: E731
    baseline = tx(pa) - tx(pb)
    if baseline <= 0:
        raise _fail(path, f"the second camera must be right of the first one, got baseline {baseline:.4g}")
    hw = None if size is None else (size[1], size[0])
    return RectifiedRig(f, baseline, float(pa[0, 2]), float(pa[1, 2]), float(pb[0, 2] - pa[0, 2]), hw)


def _key_values(text: str, sep: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        key, found, value = line.partition(sep)
        if found and key.strip() and not key.lstrip().startswith("#"):
            out[key.strip()] = value.strip()
    return out


def _floats(values: dict[str, str], key: str, count: int, path: Path) -> np.ndarray:
    if key not in values:
        raise _fail(path, f"missing {key!r}")
    try:
        arr = np.array([float(v) for v in values[key].replace(";", " ").strip("[]").split()])
    except ValueError:
        raise _fail(path, f"{key!r} is not a list of numbers") from None
    if arr.size != count:
        raise _fail(path, f"{key!r} needs {count} numbers, got {arr.size}")
    return arr


def _size(values: tuple[float, ...] | np.ndarray, path: Path) -> tuple[int, int]:
    w, h = (float(v) for v in values)
    if w != int(w) or h != int(h) or w < 2 or h < 2:  # noqa: PLR2004
        raise _fail(path, f"image size must be two integers >= 2, got {w} x {h}")
    return int(w), int(h)


@overload
def _pick_size(found: tuple[int, int], given: tuple[int, int] | None, path: Path) -> tuple[int, int]: ...
@overload
def _pick_size(found: tuple[int, int] | None, given: tuple[int, int] | None, path: Path) -> tuple[int, int] | None: ...
def _pick_size(found: tuple[int, int] | None, given: tuple[int, int] | None, path: Path) -> tuple[int, int] | None:
    """The image size `(width, height)` (the files' order): the file's, checked against `size=` (also `(w, h)`)."""
    if given is not None:
        given = _size(given, path)
        if found is not None and found != given:
            raise _fail(path, _size_mismatch(given, found, "the image size in the file"))
    return found or given


def _size_mismatch(given: tuple[int, int], found: tuple[int, int], where: str) -> str:
    """Error text for a `size=` (internally `(w, h)`) that differs from the file's, shown as `(height, width)`."""
    msg = f"size={given[::-1]} differs from {where} {found[::-1]} (sizes are (height, width))"
    return msg + (f"; did you mean size={found[::-1]}?" if given == found[::-1] else "")


def _read_kitti(
    text: str, fmt: str, cameras: tuple[int, int] | None, size: tuple[int, int] | None, path: Path
) -> StereoRig | RectifiedRig:
    values = _key_values(text, ":")
    if cameras is None:  # the colour pair (2, 3); KITTI-360 `perspective.txt` only has the front pair (0, 1)
        first = "P2" if fmt == "kitti-odometry" else "P_rect_02" if fmt == "kitti" else "K_02"
        cameras = (0, 1) if first not in values and first.replace("2", "0") in values else (2, 3)
    a, b = cameras
    if fmt == "kitti-odometry":
        pa = _floats(values, f"P{a}", 12, path).reshape(3, 4)
        pb = _floats(values, f"P{b}", 12, path).reshape(3, 4)
        return _from_projections(pa, pb, _pick_size(None, size, path), path)
    if fmt == "kitti":
        pa = _floats(values, f"P_rect_{a:02d}", 12, path).reshape(3, 4)
        pb = _floats(values, f"P_rect_{b:02d}", 12, path).reshape(3, 4)
        found = _size(_floats(values, f"S_rect_{a:02d}", 2, path), path) if f"S_rect_{a:02d}" in values else None
        return _from_projections(pa, pb, _pick_size(found, size, path), path)
    cams, poses = [], []
    for i in cameras:
        k = _floats(values, f"K_{i:02d}", 9, path).reshape(3, 3)
        d = _floats(values, f"D_{i:02d}", 5, path)
        rot = _floats(values, f"R_{i:02d}", 9, path).reshape(3, 3)
        t = _floats(values, f"T_{i:02d}", 3, path)
        found = _size(_floats(values, f"S_{i:02d}", 2, path), path) if f"S_{i:02d}" in values else None
        cam_size = _pick_size(found, size, path)
        if cam_size is None:
            raise _fail(path, f"no S_{i:02d} image size in the file; pass size=(height, width)")
        cams.append(_opencv_camera(k, d, cam_size, f"{path}: camera {i}"))
        poses.append((rot, t))
    (ra, ta), (rb, tb) = poses
    rot = rb @ ra.T  # X_i = R_i X_0 + T_i  =>  X_b = R_b R_a^T (X_a - T_a) + T_b
    return StereoRig(cams[0], cams[1], rot, tb - rot @ ta)


def _read_middlebury(text: str, scale: float, size: tuple[int, int] | None, path: Path) -> RectifiedRig:
    values = _key_values(text, "=")
    ka = _floats(values, "cam0", 9, path).reshape(3, 3)
    kb = _floats(values, "cam1", 9, path).reshape(3, 3)
    pa, pb = np.zeros((3, 4)), np.zeros((3, 4))
    pa[:, :3], pb[:, :3] = ka, kb
    _check_rectified(pa, pb, path)
    baseline = float(_floats(values, "baseline", 1, path)[0]) * scale
    shift = float(kb[0, 2] - ka[0, 2])
    doffs = float(_floats(values, "doffs", 1, path)[0]) if "doffs" in values else shift
    if abs(doffs - shift) > 1:  # file rounding is far below a pixel
        raise _fail(path, f"doffs = {doffs:g} contradicts cam1[0, 2] - cam0[0, 2] = {shift:g}: fix the file")
    found = None
    if "width" in values and "height" in values:
        found = _size((_floats(values, "width", 1, path)[0], _floats(values, "height", 1, path)[0]), path)
    f = float(ka[0, 0])
    wh = _pick_size(found, size, path)
    return RectifiedRig(f, baseline, float(ka[0, 2]), float(ka[1, 2]), doffs, None if wh is None else (wh[1], wh[0]))


def _xml_node(element: ET.Element) -> object:
    children = list(element)
    if element.get("type_id") == "opencv-matrix" or {c.tag for c in children} >= {"rows", "cols", "data"}:
        node = {c.tag: (c.text or "").split() for c in children}
        return {"rows": int(node["rows"][0]), "cols": int(node["cols"][0]), "data": [float(v) for v in node["data"]]}
    if children:
        return {c.tag: _xml_node(c) for c in children}
    parts = (element.text or "").split()
    try:
        numbers = [float(v) for v in parts]
    except ValueError:
        return " ".join(parts).strip("\"'")
    return numbers[0] if len(numbers) == 1 else numbers


def _load_opencv(text: str, path: Path) -> dict:
    if text.lstrip().startswith("<"):
        try:
            data = _xml_node(ET.fromstring(text))  # noqa: S314 - see the import
        except ET.ParseError as e:
            raise _fail(path, f"not valid OpenCV XML ({e})") from None
    else:
        data = _yaml.loads(text)
    if not isinstance(data, dict):
        raise _fail(path, "expected an OpenCV FileStorage map at the top level")
    return data


def _opencv_size(data: dict, size: tuple[int, int] | None, path: Path) -> tuple[int, int]:
    found = None
    for keys in _OPENCV_SIZE_KEYS:
        if all(k in data for k in keys):
            raw = [data[k] for k in keys] if len(keys) == 2 else _vector(data[keys[0]], "size", path)  # noqa: PLR2004
            found = _size(tuple(float(np.asarray(v).reshape(-1)[0]) for v in raw), path)
            break
    cam_size = _pick_size(found, size, path)
    if cam_size is None:
        raise _fail(path, "no image size in the file; pass size=(height, width)")
    return cam_size


def _read_opencv(text: str, scale: float, size: tuple[int, int] | None, path: Path) -> StereoRig:
    data = _load_opencv(text, path)
    keys = next((k for k in _OPENCV_KEYS if all(n in data for n in k)), None)
    if keys is None:
        names = ", ".join("/".join(k[:2]) + "/..." for k in _OPENCV_KEYS)
        why = f"no stereo calibration keys found (expected one of: {names}); a single-camera file is not a rig"
        raise _fail(path, why)
    cam_size = _opencv_size(data, size, path)
    fisheye = str(data.get("distortion_model", "")).lower() in ("fisheye", "equidistant", "kannala-brandt", "kb4")
    cams = []
    for k_key, d_key in ((keys[0], keys[1]), (keys[2], keys[3])):
        k = _matrix(data[k_key], (3, 3), k_key, path)
        d = _vector(data[d_key], d_key, path)
        cams.append(_opencv_camera(k, d, cam_size, f"{path}: {k_key}", fisheye=fisheye))
    rot = _matrix(data[keys[4]], (3, 3), keys[4], path)
    t = _matrix(data[keys[5]], (3,), keys[5], path)
    return StereoRig(cams[0], cams[1], rot, t * scale)


def _kalibr_camera(node: object, name: str, path: Path) -> Camera:
    if not isinstance(node, dict):
        raise _fail(path, f"{name} must be a map")
    model = str(node.get("camera_model", ""))
    dist = str(node.get("distortion_model", "none"))
    intr = _vector(node.get("intrinsics"), f"{name}.intrinsics", path)
    d = _vector(node.get("distortion_coeffs"), f"{name}.distortion_coeffs", path)
    res = _vector(node.get("resolution"), f"{name}.resolution", path)
    if len(res) != 2:  # noqa: PLR2004
        raise _fail(path, f"{name}.resolution must be [width, height]")
    if model not in _KALIBR_MODELS:
        raise _fail(path, f"{name}: camera_model {model!r} is not supported ({', '.join(_KALIBR_MODELS)})")
    model_id, count = _KALIBR_MODELS[model]
    if len(intr) != count:
        raise _fail(path, f"{name}: {model} needs {count} intrinsics, got {len(intr)}")
    if model == "pinhole" and dist in _KALIBR_DISTORTIONS:
        model_id, count = _KALIBR_DISTORTIONS[dist]
        if len(d) != count:
            raise _fail(path, f"{name}: {dist} needs {count} distortion coefficients, got {len(d)}")
    elif dist != "none":
        later = " (omni + radtan is planned for a later release)" if model == "omni" else ""
        raise _fail(path, f"{name}: {model} with distortion_model {dist!r} is not supported{later}")
    coefficients = d if model == "pinhole" else intr[:-4]  # omni / ds / eucm: model parameters before fu fv pu pv
    return camera(model_id, *_size(res, path), *intr[-4:], *coefficients)


def _read_kalibr(
    text: str, cameras: tuple[int, int], scale: float, size: tuple[int, int] | None, path: Path
) -> StereoRig:
    data = _yaml.loads(text)
    if not isinstance(data, dict):
        raise _fail(path, "expected a Kalibr camchain map (cam0, cam1, ...)")
    a, b = cameras
    if a == b:
        raise _fail(path, f"kalibr cameras must be two different cameras, got {cameras}")
    transform = np.eye(4)
    for i in range(min(a, b) + 1, max(a, b) + 1):  # T_cn_cnm1 maps cam n-1 coordinates to cam n
        node = data.get(f"cam{i}")
        if not isinstance(node, dict) or "T_cn_cnm1" not in node:
            raise _fail(path, f"cam{i} or its T_cn_cnm1 is missing")
        transform = _matrix(node["T_cn_cnm1"], (4, 4), f"cam{i}.T_cn_cnm1", path) @ transform
    if a > b:  # the left camera comes later in the chain: left -> right is the inverse
        transform = np.linalg.inv(transform)
    cams = []
    for i in cameras:
        if f"cam{i}" not in data:
            raise _fail(path, f"cam{i} is missing")
        cam = _kalibr_camera(data[f"cam{i}"], f"cam{i}", path)
        if size is not None and (cam.width, cam.height) != _size(size, path):
            raise _fail(path, _size_mismatch(_size(size, path), (cam.width, cam.height), f"cam{i}.resolution"))
        cams.append(cam)
    return StereoRig(cams[0], cams[1], transform[:3, :3], transform[:3, 3] * scale)


def _read_zed(text: str, scale: float, size: tuple[int, int] | None, path: Path) -> StereoRig:
    """ZED `SN<serial>.conf` (factory calibration, one section per resolution; translations in mm).

    Follows Stereolabs' OpenCV samples: `R = Rodrigues(RX, CV, RZ)` of the resolution, `T = (-Baseline, TY, TZ)`.
    """
    sections: dict[str, dict[str, str]] = {}
    current = None
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()  # noqa: PLW2901
        if line.startswith("[") and line.endswith("]"):
            current = sections.setdefault(line[1:-1].strip(), {})
        elif current is not None and "=" in line:
            key, _, value = line.partition("=")
            current[key.strip()] = value.strip()
    found = [r for r in _ZED_SIZES if f"LEFT_CAM_{r}" in sections and f"RIGHT_CAM_{r}" in sections]
    if not found or "STEREO" not in sections:
        raise _fail(path, "no [LEFT_CAM_<res>] / [RIGHT_CAM_<res>] / [STEREO] sections of a ZED calibration")
    sizes = ", ".join(f"{r} {_ZED_SIZES[r][::-1]}" for r in found)
    if size is None:
        if len(found) > 1:
            raise _fail(path, f"pass size=(height, width) of the images, one of: {sizes}")
        res = found[0]
    else:
        given = _size(size, path)
        res = next((r for r in found if _ZED_SIZES[r] == given), None)
        if res is None:
            hint = "; did you mean " + str(given) + "?" if given[::-1] in _ZED_SIZES.values() else ""
            raise _fail(path, f"size={given[::-1]} (height, width) is not a resolution in the file ({sizes}){hint}")
    cams = []
    for side in ("LEFT", "RIGHT"):
        values = sections[f"{side}_CAM_{res}"]
        fx, fy, cx, cy = (float(_floats(values, n, 1, path)[0]) for n in ("fx", "fy", "cx", "cy"))
        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        d = np.array([float(_floats(values, n, 1, path)[0]) if n in values else 0.0 for n in _ZED_DISTORTION])
        d = d if d[5:].any() else d[:5]  # k4..k6 (rational model) only in some ZED X files
        cams.append(_opencv_camera(k, d, _ZED_SIZES[res], f"{path}: {side}_CAM_{res}"))
    stereo = sections["STEREO"]

    def number(*keys: str) -> float:  # the first key the file has (per-resolution values first)
        return float(_floats(stereo, next((k for k in keys if k in stereo), keys[-1]), 1, path)[0])

    rot = _rotation_from_vector(np.array([number(f"RX_{res}"), number(f"CV_{res}"), number(f"RZ_{res}")]))
    t = np.array([-number(f"Baseline_{res}", "Baseline"), number(f"TY_{res}", "TY"), number(f"TZ_{res}", "TZ")])
    return StereoRig(cams[0], cams[1], rot, t * scale)


def _detect(path: Path, text: str) -> str:
    fmt = next((fmt for pattern, fmt in _DETECT if pattern.search(text)), None)
    if fmt is None:
        msg = f"{path}: cannot tell the calibration format; pass format= one of {', '.join(FORMATS)}"
        raise InputError(msg)
    return fmt


def _check_options(
    fmt: str, cameras: tuple[int, int] | None, unit: str | float | None
) -> tuple[tuple[int, int] | None, float]:
    """Validate `format` / `cameras` / `unit`; returns the camera pair (None: KITTI picks) and metres per file unit."""
    if fmt not in FORMATS:
        msg = f"unknown calibration format {fmt!r}; use one of {', '.join(FORMATS)}"
        raise InputError(msg)
    unit = unit if unit is not None else _DEFAULT_UNIT.get(fmt, "m")
    scale = unit_scale(unit)
    if fmt.startswith("kitti") and scale != 1.0:
        msg = "KITTI calibration is in metres; unit= does not apply"
        raise InputError(msg)
    if cameras is not None and fmt not in _DEFAULT_CAMERAS:
        msg = f"cameras= applies to multi-camera formats ({', '.join(_DEFAULT_CAMERAS)}), not {fmt!r}"
        raise InputError(msg)
    if cameras is not None and (
        len(cameras) != 2 or any(not isinstance(c, int) or c < 0 for c in cameras) or cameras[0] == cameras[1]  # noqa: PLR2004
    ):
        msg = f"cameras must be two different camera indices (left, right), got {cameras!r}"
        raise InputError(msg)
    return cameras or (None if fmt.startswith("kitti") else _DEFAULT_CAMERAS.get(fmt, (0, 1))), scale


def read_calib(  # noqa: C901, PLR0911, PLR0913
    path: str | Path,
    format: CalibFormat | None = None,  # noqa: A002
    *,
    cameras: tuple[int, int] | None = None,
    size: tuple[int, int] | None = None,
    unit: Literal["m", "cm", "mm"] | float | None = None,
    right: str | Path | None = None,
    names: tuple[str, str] | None = None,
) -> StereoRig | RectifiedRig:
    """Read a stereo calibration file.

    Already rectified data gives a `RectifiedRig`, a calibrated but unrectified rig a `StereoRig` (pass it to
    `rectify`):

    | format | files | result |
    |---|---|---|
    | `kitti` | `calib_cam_to_cam.txt` (raw, stereo 2012/2015), `P_rect_xx` | `RectifiedRig` |
    | `kitti-unrectified` | the same file, `K_xx D_xx R_xx T_xx` | `StereoRig` |
    | `kitti-odometry` | odometry / object `calib.txt`, `P0..P3` | `RectifiedRig` |
    | `middlebury` | Middlebury 2014/2021 and ETH3D `calib.txt` (baseline in mm) | `RectifiedRig` |
    | `opencv` | `cv2.FileStorage` YAML / XML of `stereoCalibrate` results | `StereoRig` |
    | `kalibr` | Kalibr `camchain.yaml` (pinhole / omni / ds / eucm) | `StereoRig` |
    | `tum` | TUM VI `camchain.yaml` (Kalibr format; same as `kalibr`) | `StereoRig` |
    | `zed` | Stereolabs ZED factory calibration `SN<serial>.conf` (pick the resolution with `size=`) | `StereoRig` |
    | `oak` | OAK-D / DepthAI calibration JSON (`eepromToJson`; cm; sockets 1 and 2 by default) | `StereoRig` |
    | `realsense` | RealSense stream profile JSON: two streams with `rs2_intrinsics` + `rs2_extrinsics` | `StereoRig` |
    | `ros` | ROS `camera_info` YAML of the left camera, the right one in `right=` | `StereoRig` |
    | `colmap` | COLMAP sparse model folder or `cameras.txt` / `cameras.bin` (+ `images`), two images | `StereoRig` |
    | `euroc` | EuRoC `cam0/sensor.yaml` (`cam1` found next to it, or `right=`) | `StereoRig` |
    | `aria` | Project Aria device JSON or MPS `online_calibration.jsonl` (the SLAM pair by default) | `StereoRig` |
    | `nerfstudio` | nerfstudio / instant-ngp `transforms.json`: two frames (`names=`), OpenGL poses | `StereoRig` |

    Args:
        path: The calibration file.
        format: One of the formats above; default: detected from the content.
        cameras: `(left, right)` camera indices for multi-camera files; default `(2, 3)` for KITTI (the colour
            pair; `(0, 1)` in files without camera 2 such as KITTI-360 `perspective.txt`), `(0, 1)` for Kalibr (any
            order: `(1, 0)` reads cam1 as the left camera).
        size: `(height, width)` of the images where the file does not store it (KITTI odometry, some OpenCV
            files); must match when the file does (files store `(width, height)`, converted on read). ZED files
            hold every resolution: `size` picks one (required when the file has several).
        unit: Unit of the translations in the file; default metres (`mm` for Middlebury and ZED). Results are in
            metres (OAK: cm; COLMAP: pass a number, the factor to metres, its scale is arbitrary).
        right: The right camera's file for formats with one file per camera (`ros`, `euroc`; EuRoC: default
            `cam1/sensor.yaml` next to `cam0`).
        names: `(left, right)` image names in a COLMAP model with more than two images, or the stream names
            in a RealSense profile (default `infrared1` / `infrared2`, else the two streams in the file), or the
            camera labels in an Aria calibration (default `camera-slam-left` / `camera-slam-right`, Gen2
            `slam-front-left` / `slam-front-right`), or the two frames' `file_path` (or file name) in a nerfstudio
            `transforms.json` (default: the file's two frames).

    Returns:
        A `StereoRig` or `RectifiedRig` in metres.

    Raises:
        InputError: Unknown format, missing or malformed entries, unsupported camera models.
    """
    path = Path(path)
    if size is not None:  # (height, width) here; the readers work in the files' (width, height)
        h, w = hw_size(size)
        size = (w, h)
    if names is not None and (len(names) != 2 or not all(isinstance(n, str) for n in names)):  # noqa: PLR2004
        msg = f"names must be two strings (left, right), got {names!r}"
        raise InputError(msg)
    if format == "colmap" or (format is None and (path.is_dir() or path.suffix == ".bin")):
        from fastforks.stereo.geometry.calib_formats import read_colmap  # noqa: PLC0415

        _, scale = _check_options("colmap", None, unit)
        return read_colmap(path, names, scale, size)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        msg = f"cannot read calibration file {path}: {e.strerror or e}"
        raise InputError(msg) from None
    except UnicodeDecodeError:
        msg = f"{path} is not a text calibration file"
        raise InputError(msg) from None
    fmt = format if format is not None else _detect(path, text)
    cameras, scale = _check_options(fmt, cameras, unit)
    if right is not None and fmt not in ("ros", "euroc"):
        msg = f"right= applies to the one-file-per-camera formats (ros, euroc), not {fmt!r}"
        raise InputError(msg)
    if names is not None and fmt not in ("colmap", "realsense", "aria", "nerfstudio"):
        msg = f"names= applies to colmap, realsense, aria and nerfstudio, not {fmt!r}"
        raise InputError(msg)
    if fmt in ("oak", "realsense", "ros", "euroc", "colmap", "aria", "nerfstudio"):
        return _read_more(fmt, text, path, cameras, right, names, scale, size)
    if fmt.startswith("kitti"):
        return _read_kitti(text, fmt, cameras, size, path)
    if fmt == "middlebury":
        return _read_middlebury(text, scale, size, path)
    if fmt == "opencv":
        return _read_opencv(text, scale, size, path)
    if fmt == "zed":
        return _read_zed(text, scale, size, path)
    return _read_kalibr(text, cameras or (0, 1), scale, size, path)


def _read_more(fmt, text, path, cameras, right, names, scale, size):  # noqa: ANN001, ANN202, PLR0913, PLR0917
    from fastforks.stereo.geometry import calib_formats as cf  # noqa: PLC0415

    if fmt == "colmap":
        return cf.read_colmap(path, names, scale, size)
    if fmt == "oak":
        return cf.read_oak(text, cameras or (1, 2), scale, size, path)
    if fmt == "realsense":
        return cf.read_realsense(text, names, scale, size, path)
    if fmt == "aria":
        return cf.read_aria(text, names, scale, size, path)
    if fmt == "nerfstudio":
        return cf.read_nerfstudio(text, names, scale, size, path)
    if right is None and fmt == "euroc":
        sibling = path.parent.parent / "cam1" / "sensor.yaml"
        right = sibling if path.parent.name == "cam0" and sibling.exists() else None
    if right is None:
        msg = f"{path}: the {fmt} format has one file per camera; pass the right camera's file as right="
        raise InputError(msg)
    right = Path(right)
    try:
        right_text = right.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        msg = f"cannot read calibration file {right}: {getattr(e, 'strerror', None) or e}"
        raise InputError(msg) from None
    reader = cf.read_ros if fmt == "ros" else cf.read_euroc
    return reader(text, right_text, right, scale, size, path)
