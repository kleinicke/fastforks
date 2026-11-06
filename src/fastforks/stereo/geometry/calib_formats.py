# ruff: noqa: PLR0913, PLR0917 - readers share the (text, ..., scale, size, path) call shape
"""More `read_calib` formats: OAK-D / DepthAI, RealSense, ROS camera_info pairs, COLMAP and EuRoC.

All results use the OpenCV pixel convention (pixel centres at integer coordinates, the image centre at
`((W - 1) / 2, (H - 1) / 2)`). Only COLMAP differs in the file (pixel centres at +0.5): its `cx, cy` lose 0.5.
"""

from __future__ import annotations

import json
import struct
import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from fastforks._core.errors import InputError
from fastforks.stereo.geometry import _yaml
from fastforks.stereo.geometry.calib import _fail, _matrix, _opencv_camera, _pick_size, _size, _vector
from fastforks.stereo.geometry.cameras import Camera, camera
from fastforks.stereo.geometry.rig import StereoRig

if TYPE_CHECKING:
    from pathlib import Path

_SOCKETS = {"CAM_A": 0, "CAM_B": 1, "CAM_C": 2, "CAM_D": 3, "CAM_E": 4, "RGB": 0, "LEFT": 1, "RIGHT": 2}


def _to_rig(left: Camera, right: Camera, x_right_from_left: np.ndarray, scale: float) -> StereoRig:
    return StereoRig(left, right, x_right_from_left[:3, :3], x_right_from_left[:3, 3] * scale)


def _device_rig(left: Camera, right: Camera, x: np.ndarray, scale: float, path: Path, what: str) -> StereoRig:
    """`_to_rig` for the device formats whose extrinsics direction the vendors do not pin down.

    A stereo rig has the right camera on the +x side of the left one (`T_x < 0` in `X_right = R X_left + T`). A file
    that gives `T_x > 0` has the two cameras in the other order (or the transform read the other way), which is not
    guessed: it raises with the fix. A baseline outside 5 mm to 1 m (centimetres or millimetres read as metres) warns.
    """
    if x[0, 3] > 0:
        msg = (
            f"{what} extrinsics put the right camera on the left of the left one (T_x > 0): the left and right cameras "
            "are swapped; pass them the other way round (swap the two sockets / streams in the file)"
        )
        raise _fail(path, msg)
    baseline = float(np.linalg.norm(x[:3, 3])) * scale
    if not 0.005 <= baseline <= 1.0:  # noqa: PLR2004 - plausible stereo baselines in metres
        warnings.warn(
            f"{path}: baseline {baseline:g} (after unit={scale:g}) looks like the wrong unit; check unit=",
            stacklevel=3,
        )
    return _to_rig(left, right, x, scale)


def _invert(transform: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = transform[:3, :3].T
    out[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return out


def _cam(size: tuple[int, int], fx: float, fy: float, cx: float, cy: float, d: np.ndarray, what: str, *,
         fisheye: bool = False, path: Path) -> Camera:  # fmt: skip
    try:
        return _opencv_camera(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), d, size, what, fisheye=fisheye)
    except InputError as e:  # re-raised with the file
        raise _fail(path, str(e)) from None


def _json(text: str, path: Path) -> dict:
    try:
        data = json.loads(text)
    except ValueError as e:
        raise _fail(path, f"not valid JSON: {e}") from None
    if not isinstance(data, dict):
        raise _fail(path, "expected a JSON object")
    return data


def _socket(value: object, path: Path) -> int:
    if isinstance(value, str) and value.upper() in _SOCKETS:
        return _SOCKETS[value.upper()]
    try:
        return int(cast("Any", value))
    except (TypeError, ValueError):
        raise _fail(path, f"unknown camera socket {value!r}") from None


def read_oak(text: str, cameras: tuple[int, int], scale: float, size: tuple[int, int] | None, path: Path) -> StereoRig:
    """OAK-D / DepthAI calibration JSON (`device.readCalibration().eepromToJson()`, `calib.json`).

    `cameraData` lists `[socket, {intrinsicMatrix, distortionCoeff, width, height, extrinsics}]`, translations in cm.
    The extrinsics of a camera are its transform to `toCameraSocket` (`X_to = R X + t`); the left camera (CAM_B,
    socket 1) points at the right one (CAM_C, socket 2). The direction follows the DepthAI docs and is not checked
    against a device here; the reverse entry (right to left) is inverted if only that one is present.
    """
    data = _json(text, path)
    entries: dict[int, dict] = {}
    for item in data.get("cameraData", []):
        if isinstance(item, list) and len(item) == 2 and isinstance(item[1], dict):  # noqa: PLR2004
            entries[_socket(item[0], path)] = item[1]
    left_id, right_id = cameras
    for sid in (left_id, right_id):
        if sid not in entries:
            raise _fail(
                path, f"camera socket {sid} not in cameraData (found {sorted(entries)}); pass cameras=(left, right)"
            )
    cams = []
    for sid in (left_id, right_id):
        e = entries[sid]
        k = _matrix(e.get("intrinsicMatrix"), (3, 3), f"socket {sid} intrinsicMatrix", path)
        d = _vector(e.get("distortionCoeff"), f"socket {sid} distortionCoeff", path)
        cam_size = _pick_size((int(e["width"]), int(e["height"])), size, path)
        if int(e.get("cameraType", 0)) in (2, 3):
            raise _fail(
                path,
                f"socket {sid} has cameraType {e['cameraType']} (equirectangular / radial division), not supported",
            )
        fisheye = int(e.get("cameraType", 0)) == 1
        d = d[:4] if fisheye else d[: next((n for n in (14, 12, 8, 5, 4, 0) if len(d) >= n), 0)]
        cams.append(_cam(cam_size, k[0, 0], k[1, 1], k[0, 2], k[1, 2], d, f"socket {sid}", fisheye=fisheye, path=path))
    transform = None
    for src, dst, entry in ((left_id, right_id, entries[left_id]), (right_id, left_id, entries[right_id])):
        ext = entry.get("extrinsics")
        if (
            not isinstance(ext, dict)
            or _socket(ext.get("toCameraSocket", -1), path) != dst
            or "rotationMatrix" not in ext
        ):
            continue
        t = ext.get("translation", {})
        trans = (
            np.array([t["x"], t["y"], t["z"]], dtype=float) if isinstance(t, dict) else _vector(t, "translation", path)
        )
        x = np.eye(4)
        x[:3, :3] = _matrix(ext["rotationMatrix"], (3, 3), f"socket {src} rotationMatrix", path)
        x[:3, 3] = trans
        transform = x if src == left_id else _invert(x)
        break
    if transform is None:
        raise _fail(path, f"no extrinsics between sockets {left_id} and {right_id}")
    return _device_rig(cams[0], cams[1], transform, scale, path, "OAK")


def _realsense_model(model: str, d: np.ndarray, n: str, path: Path) -> str:
    """The checked distortion model name of stream `n` (the model names librealsense writes)."""
    if model not in ("brown_conrady", "inverse_brown_conrady", "modified_brown_conrady", "kannala_brandt4", "none"):
        raise _fail(path, f"{n}: unsupported distortion model {model!r}")
    if model == "inverse_brown_conrady" and d.any():
        raise _fail(path, f"{n}: inverse_brown_conrady distortion is not supported (only with zero coefficients)")
    if model == "modified_brown_conrady" and d[2:4].any():
        msg = f"{n}: modified_brown_conrady with tangential coefficients (p1, p2) is not supported: it applies them"
        raise _fail(path, msg + " to the radially distorted point, unlike brown_conrady")
    if model == "kannala_brandt4" and len(d) == 5 and d[4]:  # noqa: PLR2004
        raise _fail(path, f"{n}: kannala_brandt4 uses 4 coefficients, the fifth must be 0")
    return model


def read_realsense(
    text: str, names: tuple[str, str] | None, scale: float, size: tuple[int, int] | None, path: Path
) -> StereoRig:
    """Intel RealSense stream profile JSON: two stream entries with `rs2_intrinsics` and one `rs2_extrinsics`.

    Each stream holds `width`, `height` and `intrinsics {fx, fy, ppx, ppy, coeffs, model}` (the `rs2_intrinsics`
    fields); `extrinsics {rotation, translation}` is `rs2_extrinsics` from the first (left) to the second (right)
    stream: `X_right = R X_left + t` with `R` stored column-major (9 numbers) and `t` in metres. `model` is
    `brown_conrady` (default; 5 coefficients `k1 k2 p1 p2 k3`), `modified_brown_conrady` (only with p1 = p2 = 0) or
    `kannala_brandt4` (4 coefficients, a zero fifth slot is dropped); `inverse_brown_conrady`
    needs zero coefficients (it is the inverse of what we undistort with). Pixel centres at integer coordinates are
    assumed, like OpenCV (librealsense projects with `x f + ppx`).
    """
    data = _json(text, path)
    streams = {k: v for k, v in data.items() if isinstance(v, dict) and isinstance(v.get("intrinsics"), dict)}
    if names is None:
        pairs = (("infrared1", "infrared2"), ("infrared_1", "infrared_2"), ("left", "right"))
        names = next((p for p in pairs if all(n in streams for n in p)), None)
        if names is None and len(streams) == 2:  # noqa: PLR2004
            names = tuple(streams)  # type: ignore[assignment]
    if names is None or any(n not in streams for n in names):
        raise _fail(path, f"pick the two streams with names=(left, right); streams found: {sorted(streams)}")
    cams = []
    for n in names:
        s = streams[n]
        i = s["intrinsics"]
        model = str(i.get("model", "brown_conrady")).lower().replace(" ", "_")
        d = _vector(i.get("coeffs", []), f"{n}.coeffs", path)
        model = _realsense_model(model, d, n, path)
        d = d[:4] if model == "kannala_brandt4" and len(d) == 5 else d  # noqa: PLR2004 - librealsense's 5 slots
        fisheye = model == "kannala_brandt4"
        cam_size = _pick_size((int(s["width"]), int(s["height"])), size, path)
        cams.append(_cam(cam_size, i["fx"], i["fy"], i["ppx"], i["ppy"], d, n, fisheye=fisheye, path=path))
    ext = data.get("extrinsics")
    if not isinstance(ext, dict):
        raise _fail(path, "extrinsics {rotation, translation} between the two streams is missing")
    x = np.eye(4)
    x[:3, :3] = _matrix(ext.get("rotation"), (9,), "extrinsics.rotation", path).reshape(3, 3, order="F")
    x[:3, 3] = _matrix(ext.get("translation"), (3,), "extrinsics.translation", path)
    return _device_rig(cams[0], cams[1], x, scale, path, "RealSense")


def _ros_camera(text: str, path: Path) -> tuple[Camera, np.ndarray, np.ndarray]:
    data = _yaml.loads(text)
    if not isinstance(data, dict) or "camera_matrix" not in data:
        raise _fail(path, "expected a ROS camera_info YAML (camera_matrix, distortion_coefficients, ...)")
    k = _matrix(data["camera_matrix"], (3, 3), "camera_matrix", path)
    d = _vector(data.get("distortion_coefficients", []), "distortion_coefficients", path)
    model = str(data.get("distortion_model", "plumb_bob")).lower()
    fisheye = model in ("equidistant", "fisheye")
    if model not in ("plumb_bob", "rational_polynomial", "equidistant", "fisheye"):
        raise _fail(path, f"unsupported distortion_model {model!r}")
    rect = (
        _matrix(data["rectification_matrix"], (3, 3), "rectification_matrix", path)
        if "rectification_matrix" in data
        else np.eye(3)
    )
    proj = (
        _matrix(data["projection_matrix"], (3, 4), "projection_matrix", path)
        if "projection_matrix" in data
        else np.c_[k, np.zeros(3)]
    )
    cam_size = (int(cast("int", data["image_width"])), int(cast("int", data["image_height"])))
    return (
        _cam(cam_size, k[0, 0], k[1, 1], k[0, 2], k[1, 2], d, "camera_matrix", fisheye=fisheye, path=path),
        rect,
        proj,
    )


def read_ros(
    text: str, right_text: str, right_path: Path, scale: float, size: tuple[int, int] | None, path: Path
) -> StereoRig:
    """A pair of ROS `sensor_msgs/CameraInfo` YAML files (`camera_calibration` output; left and right).

    `R` is the rectification rotation (unrectified to rectified frame) and `P_right[0, 3] = -f' B` of the rectified
    pair, so `R_rig = R_r^T R_l` and `T = R_r^T (P_r[0, 3] / f', 0, 0)` (the left camera's rectified frame is
    shifted by `P_r[0, 3] / f'` along x to reach the right one). Files of already rectified streams (`D = 0`, `R = I`)
    give the identity rotation and `T = (-B, 0, 0)`.
    """
    left, rect_l, proj_l = _ros_camera(text, path)
    right, rect_r, proj_r = _ros_camera(right_text, right_path)
    for cam in (left, right):
        _pick_size((cam.width, cam.height), size, path)
    fx = float(proj_r[0, 0])
    if fx <= 0:
        raise _fail(right_path, "projection_matrix needs a positive focal length (P[0, 0])")
    shift = np.array([proj_r[0, 3] / fx - proj_l[0, 3] / float(proj_l[0, 0]), 0.0, 0.0])
    x = np.eye(4)
    x[:3, :3] = rect_r.T @ rect_l
    x[:3, 3] = rect_r.T @ shift
    return _to_rig(left, right, x, scale)


def _euroc_camera(text: str, path: Path) -> tuple[Camera, np.ndarray]:
    data = _yaml.loads(text)
    if not isinstance(data, dict) or "T_BS" not in data:
        raise _fail(path, "expected a EuRoC sensor.yaml (T_BS, intrinsics, distortion_coefficients, resolution)")
    model, dist = str(data.get("camera_model", "pinhole")), str(data.get("distortion_model", "radial-tangential"))
    if model != "pinhole" or dist not in (
        "radial-tangential", "radtan",
    ):  # fmt: skip
        raise _fail(path, "only pinhole + radial-tangential EuRoC cameras are supported")
    intr = _vector(data.get("intrinsics"), "intrinsics", path)
    res = _vector(data.get("resolution"), "resolution", path)
    if len(intr) != 4 or len(res) != 2:  # noqa: PLR2004
        raise _fail(path, "intrinsics needs [fu, fv, cu, cv] and resolution [width, height]")
    d = _vector(data.get("distortion_coefficients"), "distortion_coefficients", path)
    t_bs = _matrix(data["T_BS"], (4, 4), "T_BS", path)
    return _cam((int(res[0]), int(res[1])), *intr, d, "intrinsics", path=path), t_bs


def read_euroc(
    text: str, right_text: str, right_path: Path, scale: float, size: tuple[int, int] | None, path: Path
) -> StereoRig:
    """Two EuRoC `sensor.yaml` files (`cam0` left, `cam1` right).

    `T_BS` is sensor to body, so `X_r = T_BS_r^-1 T_BS_l X_l`.
    """
    left, t_l = _euroc_camera(text, path)
    right, t_r = _euroc_camera(right_text, right_path)
    for cam in (left, right):
        _pick_size((cam.width, cam.height), size, path)
    return _to_rig(left, right, _invert(t_r) @ t_l, scale)


# COLMAP camera models: id -> (name, parameter count)
_COLMAP_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5), 4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}  # fmt: skip


def _colmap_camera(model: str, size: tuple[int, int], p: list[float], path: Path) -> Camera:  # noqa: PLR0911
    """COLMAP parameters to a Camera; `cx, cy` move from COLMAP's pixel-corner origin to the OpenCV convention."""
    w, h = size
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = p
        return camera("pinhole", w, h, f, f, cx - 0.5, cy - 0.5)
    if model in ("SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy, *k = p
        return camera("pinhole-opencv", w, h, f, f, cx - 0.5, cy - 0.5, *k, *[0.0] * (4 - len(k)))
    if model in ("SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        f, cx, cy, *k = p
        return camera("kb4", w, h, f, f, cx - 0.5, cy - 0.5, *k, *[0.0] * (4 - len(k)))
    fx, fy, cx, cy, *rest = p
    cx, cy = cx - 0.5, cy - 0.5
    if model == "PINHOLE":
        return camera("pinhole", w, h, fx, fy, cx, cy)
    if model in ("OPENCV", "FULL_OPENCV"):
        return camera("pinhole-opencv", w, h, fx, fy, cx, cy, *rest)  # k1 k2 p1 p2 [k3 k4 k5 k6]: OpenCV order
    if model == "OPENCV_FISHEYE":
        return camera("kb4", w, h, fx, fy, cx, cy, *rest)
    if model == "THIN_PRISM_FISHEYE":
        return camera("thin-prism-fisheye", w, h, fx, fy, cx, cy, *rest)
    if model == "FOV":  # COLMAP's omega = 0 is the plain pinhole
        return camera("fov", w, h, fx, fy, cx, cy, *rest) if rest[0] else camera("pinhole", w, h, fx, fy, cx, cy)
    raise _fail(path, f"COLMAP camera model {model} is not supported")


def _colmap_dir(path: Path) -> tuple[Path, str]:
    if path.is_dir():
        return path, ".bin" if (path / "cameras.bin").exists() else ".txt"
    if path.suffix in (".bin", ".txt"):
        return path.parent, path.suffix
    raise _fail(path, "COLMAP: pass the sparse model folder or its cameras.txt / cameras.bin")


def _colmap_cameras(folder: Path, ext: str) -> dict[int, Camera]:
    path = folder / f"cameras{ext}"
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise _fail(path, f"cannot read: {e.strerror or e}") from None
    out: dict[int, Camera] = {}
    if ext == ".bin":
        (count,), pos = struct.unpack_from("<Q", raw), 8
        for _ in range(count):
            cid, mid, w, h = struct.unpack_from("<iiQQ", raw, pos)
            pos += 24
            if mid not in _COLMAP_MODELS:
                raise _fail(path, f"unknown COLMAP camera model id {mid}")
            name, n = _COLMAP_MODELS[mid]
            params = list(struct.unpack_from(f"<{n}d", raw, pos))
            pos += 8 * n
            out[cid] = _colmap_camera(name, (w, h), params, path)
        return out
    for line in raw.decode().splitlines():
        t = line.split()
        if not t or t[0].startswith("#"):
            continue
        try:
            out[int(t[0])] = _colmap_camera(t[1], (int(t[2]), int(t[3])), [float(v) for v in t[4:]], path)
        except (ValueError, IndexError):
            raise _fail(path, f"malformed camera line {line!r}") from None
    return out


def _quat_to_rot(q: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def _colmap_image_line(t: list[str]) -> bool:
    """`id qw qx qy qz tx ty tz camera name`; an all-numeric line of triples is an observation line (VIK-01)."""
    try:
        int(t[0]), int(t[8]), [float(v) for v in t[1:8]]
    except (ValueError, IndexError):
        return False
    if len(t) < 10:  # noqa: PLR2004
        return False
    try:
        [float(v) for v in t]
    except ValueError:
        return True
    return len(t) % 3 != 0


def _colmap_images_txt(path: Path, text: str) -> list[tuple[str, tuple, tuple, int]]:
    """The image rows of `images.txt`.

    Each image line is followed by its observation line (`x y point3d_id` triples, may be empty or stripped).
    """
    rows, observations = [], False
    for line in text.splitlines():
        t = line.split()
        if t and t[0].startswith("#"):
            continue
        if observations:
            observations = False
            if not _colmap_image_line(t):  # else the observation line was stripped: this is the next image
                continue
        if not t:
            continue
        if not _colmap_image_line(t):
            raise _fail(path, f"not an image line (id qw qx qy qz tx ty tz camera name): {line.strip()[:60]!r}")
        rows.append((" ".join(t[9:]), tuple(map(float, t[1:5])), tuple(map(float, t[5:8])), int(t[8])))
        observations = True
    return rows


def _colmap_images(folder: Path, ext: str) -> dict[str, tuple[np.ndarray, int]]:
    """`name -> (4x4 world-to-camera, camera id)`."""
    path = folder / f"images{ext}"
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise _fail(path, f"cannot read: {e.strerror or e}") from None
    rows: list[tuple[str, tuple, tuple, int]] = []
    if ext == ".bin":
        (count,), pos = struct.unpack_from("<Q", raw), 8
        for _ in range(count):
            _, *rest = struct.unpack_from("<i4d3di", raw, pos)
            pos += 4 + 56 + 4
            end = raw.index(b"\0", pos)
            name = raw[pos:end].decode()
            pos = end + 1
            (points,) = struct.unpack_from("<Q", raw, pos)
            pos += 8 + 24 * points
            rows.append((name, tuple(rest[:4]), tuple(rest[4:7]), rest[7]))
    else:
        rows = _colmap_images_txt(path, raw.decode())
    out = {}
    for name, q, t, cid in rows:
        x = np.eye(4)
        x[:3, :3] = _quat_to_rot(q)  # type: ignore[arg-type]
        x[:3, 3] = t
        out[name] = (x, cid)
    return out


def read_colmap(path: Path, names: tuple[str, str] | None, scale: float, size: tuple[int, int] | None) -> StereoRig:
    """COLMAP sparse model (`cameras` + `images`, text or binary; the folder or either file).

    Poses are world-to-camera (`X_cam = R_i X_world + t_i`), so `X_r = R_r R_l^T (X_l - t_l) + t_r`. Pick the
    two images with `names=(left, right)` (image file names); a model of exactly two images needs none. COLMAP's
    reconstruction is up to scale: `unit=` as a number is the factor to metres of a COLMAP unit.
    COLMAP puts pixel centres at +0.5, so `cx, cy` are read as `c - 0.5`.
    """
    folder, ext = _colmap_dir(path)
    cams, images = _colmap_cameras(folder, ext), _colmap_images(folder, ext)
    if names is None:
        if len(images) != 2:  # noqa: PLR2004
            raise _fail(path, f"the model has {len(images)} images; pick two with names=(left, right)")
        names = cast("tuple[str, str]", tuple(images))
    if any(n not in images for n in names):
        raise _fail(path, f"image names {names} not in the model")
    (x_l, c_l), (x_r, c_r) = images[names[0]], images[names[1]]
    for cid in (c_l, c_r):
        if cid not in cams:
            raise _fail(path, f"camera {cid} is not in cameras{ext}")
        _pick_size((cams[cid].width, cams[cid].height), size, path)
    return _to_rig(cams[c_l], cams[c_r], x_r @ _invert(x_l), scale)


_ARIA_PAIRS = (("camera-slam-left", "camera-slam-right"), ("slam-front-left", "slam-front-right"))  # Gen1, Gen2
# Image sizes (width, height) `projectaria_tools` gives the cameras of a calibration JSON, which stores none
# (`device_calibration_from_json_string`; checked on the Gen1 and Gen2 sample files and the sample VRS SLAM frames).
# Gen2 is told by its `slam-*` labels. The VRS streams of the RGB and eye-tracking cameras are recorded smaller
# (Gen1 RGB 1408x1408, ET 320x240): pass `size=` for those.
_ARIA_SIZES = {
    "gen1": {"camera-slam": (640, 480), "camera-et": (640, 480), "camera-rgb": (2880, 2880)},
    "gen2": {"slam-": (512, 512), "camera-et": (400, 400), "camera-rgb": (4032, 3024)},
}


def _aria_default_size(label: str, labels: list[str]) -> tuple[int, int] | None:
    table = _ARIA_SIZES["gen2" if any(n.startswith("slam-") for n in labels) else "gen1"]
    return next((v for k, v in table.items() if label.startswith(k)), None)


def _aria_pose(entry: dict, label: str, path: Path) -> np.ndarray:
    """`T_Device_Camera` {Translation [x, y, z], UnitQuaternion [w, [x, y, z]]} as a 4x4 camera-to-device matrix."""
    pose = entry.get("T_Device_Camera")
    if not isinstance(pose, dict):
        raise _fail(path, f"{label}: T_Device_Camera is missing")
    q = pose.get("UnitQuaternion")
    if not (isinstance(q, list) and len(q) == 2 and isinstance(q[1], list)):  # noqa: PLR2004
        raise _fail(path, f"{label}: UnitQuaternion must be [w, [x, y, z]], got {q!r}")
    x = np.eye(4)
    x[:3, :3] = _quat_to_rot((float(q[0]), *_matrix(q[1], (3,), f"{label}.UnitQuaternion", path)))
    x[:3, 3] = _matrix(pose.get("Translation"), (3,), f"{label}.Translation", path)
    return x


def _aria_json(text: str, path: Path) -> dict:
    """The calibration object of a device JSON, or of the first line of an MPS `online_calibration.jsonl`."""
    first = text.lstrip().split("\n", 1)[0] if path.suffix == ".jsonl" else text
    return _json(first, path)


def read_aria(
    text: str, names: tuple[str, str] | None, scale: float, size: tuple[int, int] | None, path: Path
) -> StereoRig:
    """Project Aria device calibration JSON (`CameraCalibrations`, as `projectaria_tools` reads and writes it).

    Source: projectaria_tools `core/calibration/loader/DeviceCalibrationJson.cpp` / `SensorCalibrationJson.cpp`
    and the docs pages `tech_insights/camera_intrinsic_models` and `tech_spec/device_calibration`. Each camera
    entry has a `Label`, `Projection {Name: "FisheyeRadTanThinPrism", Params: [f, cx, cy, k0..k5, p0, p1, s0..s3]}`
    (Fisheye624, one focal length, pixel centres at integer coordinates) and `T_Device_Camera {Translation,
    UnitQuaternion: [w, [x, y, z]]}` (camera to device, metres), so `X_right = T_Device_Right^-1 T_Device_Left
    X_left`. The pair is the SLAM pair (`camera-slam-left` / `-right` on Gen1, `slam-front-left` / `-right` on Gen2)
    unless `names=` picks other labels. The image size is `ConfigData {ImageWidth, ImageHeight}` when present, else
    the top-level `ImageSizes` of an MPS online calibration, else the sizes `projectaria_tools` assumes for the
    label (`_ARIA_SIZES`); `size=` overrides the latter two. The Gen1 SLAM cameras are mounted sideways and turned
    away from each other, so that pair is a vertical rig (`rectify` asks to rotate); Gen2's front pair is
    horizontal. A `.jsonl` file (MPS `online_calibration.jsonl`, one calibration per timestamp) reads its first line.
    """
    data = _aria_json(text, path)
    entries = data.get("CameraCalibrations")
    if not isinstance(entries, list):
        raise _fail(path, "no CameraCalibrations list")
    by_label = {e.get("Label"): e for e in entries if isinstance(e, dict)}
    online_sizes = data.get("ImageSizes")
    online_sizes = dict(zip(by_label, online_sizes, strict=False)) if isinstance(online_sizes, list) else {}
    labels = sorted(map(str, by_label))
    names = names or next((p for p in _ARIA_PAIRS if all(n in by_label for n in p)), _ARIA_PAIRS[0])
    missing = [n for n in names if n not in by_label]
    if missing:
        raise _fail(path, f"cameras {missing} not in the file; pick two with names=(left, right) from {labels}")
    cams, poses = [], []
    for label in names:
        entry = by_label[label]
        proj = entry.get("Projection") if isinstance(entry.get("Projection"), dict) else {}
        if proj.get("Name") != "FisheyeRadTanThinPrism":
            raise _fail(path, f"{label}: projection {proj.get('Name')!r} is not supported (FisheyeRadTanThinPrism)")
        p = _matrix(proj.get("Params"), (15,), f"{label}.Projection.Params", path)
        config = entry.get("ConfigData") if isinstance(entry.get("ConfigData"), dict) else {}
        if "ImageWidth" in config:
            cam_size = _pick_size((int(config["ImageWidth"]), int(config["ImageHeight"])), size, path)
        else:  # not stored by the file: size= wins over the sizes the Aria tools assume
            listed = online_sizes.get(label)
            guess = (int(listed[0]), int(listed[1])) if listed else _aria_default_size(label, labels)
            cam_size = _pick_size(None, size, path) or guess
        if cam_size is None:
            raise _fail(path, f"{label}: no image size known for this label; pass size=(height, width)")
        cams.append(camera("fisheye624", *cam_size, p[0], p[0], p[1], p[2], *p[3:]))
        poses.append(_aria_pose(entry, label, path))
    # poses pin down the direction: no swap guess
    return _to_rig(cams[0], cams[1], _invert(poses[1]) @ poses[0], scale)


_NERFSTUDIO_PINHOLE = ("PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV")  # k1 k2 p1 p2 k3, missing = 0


def _ns_value(frame: dict, top: dict, key: str) -> object:
    return frame.get(key, top.get(key))


def _ns_camera(frame: dict, top: dict, size: tuple[int, int] | None, label: str, path: Path) -> Camera:
    """Camera of one `transforms.json` frame: its own intrinsics where it has them, else the file's shared ones."""
    model = str(_ns_value(frame, top, "camera_model") or "OPENCV").upper()  # instant-ngp has no camera_model key
    fisheye = model == "OPENCV_FISHEYE"
    if not (fisheye or model in _NERFSTUDIO_PINHOLE) or _ns_value(frame, top, "is_fisheye"):
        raise _fail(path, f"{label}: camera_model {model!r} is not supported (PINHOLE, OPENCV, OPENCV_FISHEYE)")

    def opt(key: str) -> float | None:
        value = _ns_value(frame, top, key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _fail(path, f"{label}: {key} must be a number, got {value!r}")
        return float(value)

    def num(key: str, default: float) -> float:
        value = opt(key)
        return default if value is None else value

    w, h = opt("w"), opt("h")
    given = _pick_size(None, size, path)
    found = _size((w, h), path) if w is not None and h is not None else None
    cam_size = _pick_size(found, size, path) if found else given
    if cam_size is None:
        raise _fail(path, f"{label}: no w / h in the file; pass size=(height, width)")
    fl_x = opt("fl_x")
    if fl_x is None:  # instant-ngp can give only the field of view
        angle = opt("camera_angle_x")
        if angle is None:
            raise _fail(path, f"{label}: no fl_x (or camera_angle_x)")
        fl_x = cam_size[0] / (2 * np.tan(angle / 2))
    fl_y = num("fl_y", fl_x)
    cx, cy = num("cx", cam_size[0] / 2), num("cy", cam_size[1] / 2)
    keys = ("k1", "k2", "k3", "k4") if fisheye else ("k1", "k2", "p1", "p2", "k3")
    d = np.array([num(k, 0.0) for k in keys])
    # nerfstudio / instant-ngp put the first pixel's centre at +0.5 (as COLMAP); the rig uses integer centres
    return _cam(cam_size, fl_x, fl_y, cx - 0.5, cy - 0.5, d, label, fisheye=fisheye, path=path)


def _ns_frame(frames: list, name: str, path: Path) -> dict:
    exact = [f for f in frames if f.get("file_path") == name]
    hits = exact or [f for f in frames if str(f.get("file_path", "")).rsplit("/", 1)[-1] == name]
    if len(hits) != 1:
        some = [f.get("file_path") for f in frames[:4]]
        why = "not in the file" if not hits else f"matches {len(hits)} frames; give the full file_path"
        raise _fail(path, f"frame {name!r} {why} (e.g. {some})")
    return hits[0]


def read_nerfstudio(
    text: str, names: tuple[str, str] | None, scale: float, size: tuple[int, int] | None, path: Path
) -> StereoRig:
    """Nerfstudio / instant-ngp `transforms.json`: two of its frames as a rig.

    Shared intrinsics at the top level (`fl_x fl_y cx cy w h`, `k1 k2 [p1 p2 k3]`, or `k1..k4` for `OPENCV_FISHEYE`);
    each frame may override them. Poses are `transform_matrix` camera-to-world with OpenGL camera axes
    (x right, y up, z back), read through `StereoRig.from_poses`. The scene scale is arbitrary (as COLMAP):
    `unit=` as a number is the factor to metres. Intrinsics are those of the stored `w x h`; for downscaled images
    (`images_4`) use `rig.resized(...)`. The pair is `names=(left, right)`, each a frame's `file_path` or its file
    name; a file with exactly two frames needs none.
    """
    data = _json(text, path)
    frames = data.get("frames")
    if not isinstance(frames, list) or not all(isinstance(f, dict) for f in frames):
        raise _fail(path, "no frames list")
    if names is None:
        if len(frames) != 2:  # noqa: PLR2004
            raise _fail(
                path, f"the file has {len(frames)} frames; pick two with names=(left, right), a frame's file_path"
            )
        names = (str(frames[0].get("file_path")), str(frames[1].get("file_path")))
    pair = [_ns_frame(frames, n, path) for n in names]
    cams = [_ns_camera(f, data, size, str(n), path) for f, n in zip(pair, names, strict=True)]
    try:
        rig = StereoRig.from_poses(
            cams[0], cams[1], axes="opengl",
            left_to_world=pair[0].get("transform_matrix"), right_to_world=pair[1].get("transform_matrix"),
        )  # fmt: skip
    except InputError as e:
        raise _fail(path, str(e)) from None
    return _to_rig(rig.left, rig.right, np.block([[rig.R, rig.T[:, None]], [np.zeros((1, 3)), 1.0]]), scale)
