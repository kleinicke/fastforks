# ruff: noqa: E501 - inline calibration fixtures
"""`read_calib` formats OAK-D / DepthAI, RealSense, TUM, ROS camera_info pair, COLMAP (txt + bin), EuRoC and Aria.

Layouts follow the format documents: DepthAI `CalibrationHandler.eepromToJson` (`cameraData`, translations in cm),
librealsense `rs2_intrinsics` / `rs2_extrinsics` (column-major rotation), ROS `sensor_msgs/CameraInfo` as written by
`camera_calibration`, COLMAP's `cameras.txt` / `images.txt` and binary `src/colmap/scene/reconstruction_io.cc`
layout, EuRoC `sensor.yaml` (`T_BS` sensor to body), TUM VI = Kalibr camchain.
"""

import json
import struct
from dataclasses import astuple
from pathlib import Path

import numpy as np
import pytest
import torch

from _data import DATA
from fastforks import InputError
from fastforks.stereo import geometry as g
from fastforks.stereo.geometry.calib_formats import _colmap_camera

W, H = 640, 480
FX, FY = 520.0, 521.0
CX, CY = (W - 1) / 2 + 1.3, (H - 1) / 2 - 0.7
DIST = (0.1, -0.05, 0.001, 0.002)


def _rot(rx=0.01, ry=-0.02, rz=0.005):
    return g.calib._rotation_from_vector(np.array([rx, ry, rz]))


def _cam(cx=CX, cy=CY, dist=DIST):
    return g.camera("pinhole-opencv", W, H, FX, FY, cx, cy, *dist)


RIG = g.StereoRig(_cam(), _cam(CX + 1.1, CY - 0.4), _rot(), np.array([-0.12, 0.002, -0.001]))


def _same(rig, ref, atol=1e-9):
    for a, b in ((rig.left, ref.left), (rig.right, ref.right)):
        assert type(a) is type(b)
        np.testing.assert_allclose(astuple(a), astuple(b), atol=atol)
    np.testing.assert_allclose(rig.R, ref.R, atol=atol)
    np.testing.assert_allclose(rig.T, ref.T, atol=atol)


def _mat(m):
    return {"rows": len(m), "cols": len(m[0]), "data": np.asarray(m).ravel().tolist()}


def _flow(m):
    """A ROS matrix node in block style (rows / cols / data), as `camera_calibration` writes it."""
    m = np.asarray(m)
    return f"\n  rows: {m.shape[0]}\n  cols: {m.shape[1]}\n  data: {m.ravel().tolist()}"


def _transform(r, t):
    x = np.eye(4)
    x[:3, :3], x[:3, 3] = r, t
    return x


def test_oak_depthai_json(tmp_path):
    def entry(cam, ext):
        e = {"intrinsicMatrix": [[FX, 0, cam.cx], [0, FY, cam.cy], [0, 0, 1]], "distortionCoeff": [*DIST, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             "width": W, "height": H, "cameraType": 0}  # fmt: skip
        if ext:
            e["extrinsics"] = ext
        return e

    ext = {"rotationMatrix": RIG.R.tolist(), "toCameraSocket": 2,
           "translation": {"x": RIG.T[0] * 100, "y": RIG.T[1] * 100, "z": RIG.T[2] * 100}}  # fmt: skip
    data = {"cameraData": [[0, entry(RIG.left, None)], [1, entry(RIG.left, ext)], [2, entry(RIG.right, None)]]}
    path = tmp_path / "calib.json"
    path.write_text(json.dumps(data))
    ref = g.StereoRig(g.camera("pinhole-opencv", W, H, FX, FY, RIG.left.cx, RIG.left.cy, *DIST, *[0] * 10), g.camera("pinhole-opencv", W, H, FX, FY, RIG.right.cx, RIG.right.cy, *DIST, *[0] * 10), RIG.R, RIG.T)  # fmt: skip
    _same(g.read_calib(path), ref)  # detected, cm to m
    _same(g.read_calib(path, "oak", cameras=(1, 2)), ref)
    # only the reverse entry (right -> left) present: inverted
    inv = _transform(RIG.R, RIG.T)
    inv = np.linalg.inv(inv)
    back = {
        "rotationMatrix": inv[:3, :3].tolist(),
        "toCameraSocket": "CAM_B",
        "translation": dict(zip("xyz", (inv[:3, 3] * 100).tolist(), strict=True)),
    }
    data["cameraData"] = [[1, entry(RIG.left, None)], ["CAM_C", entry(RIG.right, back)]]
    path.write_text(json.dumps(data))
    _same(g.read_calib(path), ref)
    with pytest.raises(InputError, match="socket 3"):
        g.read_calib(path, cameras=(1, 3))


def test_realsense_profile(tmp_path):
    def stream(cam):
        return {"width": W, "height": H, "intrinsics": {"fx": FX, "fy": FY, "ppx": cam.cx, "ppy": cam.cy, "coeffs": [*DIST, 0.0], "model": "brown_conrady"}}  # fmt: skip

    data = {"infrared2": stream(RIG.right), "infrared1": stream(RIG.left),
            "extrinsics": {"rotation": RIG.R.ravel(order="F").tolist(), "translation": RIG.T.tolist()}}  # fmt: skip
    path = tmp_path / "rs.json"
    path.write_text(json.dumps(data))
    ref = g.StereoRig(g.camera("pinhole-opencv", W, H, FX, FY, RIG.left.cx, RIG.left.cy, *DIST, 0.0), g.camera("pinhole-opencv", W, H, FX, FY, RIG.right.cx, RIG.right.cy, *DIST, 0.0), RIG.R, RIG.T)  # fmt: skip
    _same(g.read_calib(path), ref)  # infrared1 -> infrared2 regardless of the key order
    data["color_stream"] = data.pop("infrared1")
    data["depth_stream"] = data.pop("infrared2")
    path.write_text(json.dumps(data))
    _same(g.read_calib(path, names=("color_stream", "depth_stream")), ref)
    with pytest.raises(InputError, match="names="):
        g.read_calib(path, "realsense", names=("a", "b"))
    data["depth_stream"]["intrinsics"].update(model="inverse_brown_conrady")
    path.write_text(json.dumps(data))
    with pytest.raises(InputError, match="inverse_brown_conrady"):
        g.read_calib(path, "realsense", names=("color_stream", "depth_stream"))


def test_realsense_error_names_the_file_once(tmp_path):
    def stream(coeffs):
        return {"width": W, "height": H, "intrinsics": {"fx": FX, "fy": FY, "ppx": W / 2, "ppy": H / 2, "coeffs": coeffs, "model": "brown_conrady"}}  # fmt: skip

    ext = {"rotation": np.eye(3).ravel().tolist(), "translation": [-0.1, 0, 0]}
    path = tmp_path / "rs.json"
    path.write_text(json.dumps({"left": stream([0.0] * 3), "right": stream([0.0] * 5), "extrinsics": ext}))
    with pytest.raises(InputError) as e:
        g.read_calib(path, "realsense")
    assert str(e.value).count(str(path)) == 1


@pytest.mark.parametrize("fmt", ["oak", "ros", "kalibr"])
def test_different_left_right_sizes_read_but_do_not_rectify(tmp_path, fmt):
    """A pair of different sizes reads with both sizes (e.g. Aria SLAM + RGB); `rectify` rejects it."""
    w2 = W + 80
    cx, cy = (W - 1) / 2, (H - 1) / 2
    if fmt == "oak":
        e = {"intrinsicMatrix": [[FX, 0, cx], [0, FX, cy], [0, 0, 1]], "distortionCoeff": [], "width": W, "height": H}
        ext = {"rotationMatrix": np.eye(3).tolist(), "toCameraSocket": 2, "translation": {"x": -10, "y": 0, "z": 0}}
        p = tmp_path / "o.json"
        p.write_text(json.dumps({"cameraData": [[1, {**e, "extrinsics": ext}], [2, {**e, "width": w2}]]}))
        call = (p,)
    elif fmt == "ros":
        k = [[FX, 0, cx], [0, FX, cy], [0, 0, 1]]
        for n, width, tx in (("l", W, 0), ("r", w2, -FX * 0.1)):
            (tmp_path / f"{n}.yaml").write_text(
                f"image_width: {width}\nimage_height: {H}\ncamera_matrix: {_flow(k)}\ndistortion_coefficients: \n  rows: 1\n  cols: 5\n  data: [0, 0, 0, 0, 0]\n"
                f"rectification_matrix: {_flow(np.eye(3))}\nprojection_matrix: {_flow([[FX, 0, cx, tx], [0, FX, cy, 0], [0, 0, 1, 0]])}\n"
            )  # fmt: skip
        call = (tmp_path / "l.yaml",)
    else:
        t = _transform(np.eye(3), [-0.1, 0, 0])
        p = tmp_path / "c.yaml"
        p.write_text("".join(f"cam{i}:\n  camera_model: pinhole\n  intrinsics: [{FX}, {FX}, {cx}, {cy}]\n  distortion_model: none\n  distortion_coeffs: []\n  resolution: [{(W, w2)[i]}, {H}]\n" + ("  T_cn_cnm1:\n" + "".join(f"    - {r.tolist()}\n" for r in t) if i else "") for i in range(2)))  # fmt: skip
        call = (p, fmt)
    kwargs = {"right": tmp_path / "r.yaml"} if fmt == "ros" else {}
    rig = g.read_calib(*call, **kwargs)
    assert (rig.left.size, rig.right.size) == ((H, W), (H, w2))
    with pytest.raises(InputError, match="same image size"):
        g.rectify(rig)


def test_tum_is_the_kalibr_camchain(tmp_path):
    t = _transform(RIG.R, RIG.T)
    text = "".join(
        f"cam{i}:\n  camera_model: pinhole\n  intrinsics: [{FX}, {FY}, {c.cx}, {c.cy}]\n  distortion_model: radtan\n"
        f"  distortion_coeffs: {list(DIST)}\n  resolution: [{W}, {H}]\n"
        + ("  T_cn_cnm1:\n" + "".join(f"    - {row.tolist()}\n" for row in t) if i else "")
        for i, c in enumerate((RIG.left, RIG.right))
    )  # fmt: skip
    path = tmp_path / "camchain.yaml"
    path.write_text(text)
    _same(g.read_calib(path, "tum"), RIG)
    _same(g.read_calib(path, "tum"), g.read_calib(path, "kalibr"))
    _same(g.read_calib(path), RIG)


def _ros_text(cam, rect, proj, name):
    d = [*DIST, 0.0]
    return (f"image_width: {W}\nimage_height: {H}\ncamera_name: {name}\ncamera_matrix: {_flow([[FX, 0, cam.cx], [0, FY, cam.cy], [0, 0, 1]])}\n"
            f"distortion_model: plumb_bob\ndistortion_coefficients: \n  rows: 1\n  cols: 5\n  data: {d}\n"
            f"rectification_matrix: {_flow(rect)}\nprojection_matrix: {_flow(proj)}\n")  # fmt: skip


def test_ros_camera_info_pair_from_stereo_rectify(tmp_path):
    rect = g.rectify(RIG)
    f = rect.P1[0, 0]
    left = tmp_path / "left.yaml"
    right = tmp_path / "right.yaml"
    left.write_text(_ros_text(RIG.left, rect.R1, rect.P1, "left"))
    right.write_text(_ros_text(RIG.right, rect.R2, rect.P2, "right"))
    assert rect.P2[0, 3] == pytest.approx(-f * 0.12, rel=1e-3)
    ref = g.StereoRig(g.camera("pinhole-opencv", W, H, FX, FY, RIG.left.cx, RIG.left.cy, *DIST, 0.0), g.camera("pinhole-opencv", W, H, FX, FY, RIG.right.cx, RIG.right.cy, *DIST, 0.0), RIG.R, RIG.T)  # fmt: skip
    _same(
        g.read_calib(left, right=right), ref, atol=2e-4
    )  # rectification ends at 1e-4: R / T within the rounding of the baseline
    with pytest.raises(InputError, match="right="):
        g.read_calib(left, "ros")
    with pytest.raises(InputError, match="right="):
        g.read_calib(left, "opencv", right=right)


def test_ros_rectified_pair_is_identity_rotation(tmp_path):
    f, b = 525.0, 0.1
    k = [[f, 0, 319.5], [0, f, 239.5], [0, 0, 1]]
    for name, tx in (("l", 0.0), ("r", -f * b)):
        (tmp_path / f"{name}.yaml").write_text(
            f"image_width: {W}\nimage_height: {H}\ncamera_matrix: {_flow(k)}\ndistortion_model: plumb_bob\n"
            f"distortion_coefficients: \n  rows: 1\n  cols: 5\n  data: [0, 0, 0, 0, 0]\nrectification_matrix: {_flow(np.eye(3))}\n"
            f"projection_matrix: {_flow([[f, 0, 319.5, tx], [0, f, 239.5, 0], [0, 0, 1, 0]])}\n"
        )  # fmt: skip
    rig = g.read_calib(tmp_path / "l.yaml", right=tmp_path / "r.yaml")
    np.testing.assert_allclose(rig.R, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(rig.T, [-b, 0, 0], atol=1e-12)
    assert isinstance(rig.left, g.PinholeIdeal) if hasattr(g, "PinholeIdeal") else True


def _euroc_text(cam, t_bs):
    return (f"sensor_type: camera\ncomment: VI-Sensor cam\nT_BS:\n  cols: 4\n  rows: 4\n  data: {np.asarray(t_bs).ravel().tolist()}\n"
            f"rate_hz: 20\nresolution: [{W}, {H}]\ncamera_model: pinhole\nintrinsics: [{FX}, {FY}, {cam.cx}, {cam.cy}]\n"
            f"distortion_model: radial-tangential\ndistortion_coefficients: {list(DIST)}\n")  # fmt: skip


def test_euroc_sensor_yaml_pair(tmp_path):
    t_l = _transform(_rot(0.3, -0.2, 0.1), [0.05, -0.01, 0.02])  # left camera in the body frame (sensor to body)
    t_r = t_l @ np.linalg.inv(_transform(RIG.R, RIG.T))
    for name, cam, t in (("cam0", RIG.left, t_l), ("cam1", RIG.right, t_r)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "sensor.yaml").write_text(_euroc_text(cam, t))
    _same(g.read_calib(tmp_path / "cam0" / "sensor.yaml"), RIG)  # detected; cam1 found next to cam0
    _same(g.read_calib(tmp_path / "cam0" / "sensor.yaml", "euroc", right=tmp_path / "cam1" / "sensor.yaml"), RIG)
    with pytest.raises(InputError, match="right="):
        g.read_calib(tmp_path / "cam1" / "sensor.yaml")


def _quat(r):
    """Unit quaternion (w, x, y, z) of a rotation matrix (Shepperd; the formulas of the COLMAP docs)."""
    w = np.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    return np.array([w, (r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w)])


def _write_colmap(folder, binary):
    """cameras + images of the rig: world = left camera; images a (left), b (right), c (an extra view)."""
    folder.mkdir()
    poses = {
        "a.png": (np.eye(3), np.zeros(3)),
        "b.png": (RIG.R, RIG.T),
        "c.png": (_rot(0.1, 0, 0), np.array([1.0, 2.0, 3.0])),
    }
    cams = {1: ("OPENCV", [FX, FY, CX + 0.5, CY + 0.5, *DIST]), 2: ("PINHOLE", [FX, FY, CX + 1.6, CY + 0.1])}
    ids = {"a.png": 1, "b.png": 2, "c.png": 1}
    if binary:
        ids_model = {"OPENCV": 4, "PINHOLE": 1}
        cb = struct.pack("<Q", len(cams))
        for cid, (model, p) in cams.items():
            cb += struct.pack("<iiQQ", cid, ids_model[model], W, H) + struct.pack(f"<{len(p)}d", *p)
        ib = struct.pack("<Q", len(poses))
        for i, (name, (r, t)) in enumerate(poses.items(), 1):
            ib += struct.pack("<i4d3di", i, *_quat(r), *t, ids[name]) + name.encode() + b"\0" + struct.pack("<Q", 2)
            ib += struct.pack("<ddq", 1.0, 2.0, 7) + struct.pack("<ddq", 3.0, 4.0, -1)
        (folder / "cameras.bin").write_bytes(cb)
        (folder / "images.bin").write_bytes(ib)
        return
    (folder / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        + "".join(f"{c} {m} {W} {H} {' '.join(repr(float(v)) for v in p)}\n" for c, (m, p) in cams.items())
    )
    lines = "# Image list\n"
    for i, (name, (r, t)) in enumerate(poses.items(), 1):
        lines += f"{i} {' '.join(repr(float(v)) for v in _quat(r))} {' '.join(repr(float(v)) for v in t)} {ids[name]} {name}\n1.0 2.0 7 3.0 4.0 -1\n"
    (folder / "images.txt").write_text(lines)


@pytest.mark.parametrize("binary", [False, True])
def test_colmap_txt_and_bin_models(tmp_path, binary):
    folder = tmp_path / "sparse"
    _write_colmap(folder, binary)
    ref = g.StereoRig(g.camera("pinhole-opencv", W, H, FX, FY, CX, CY, *DIST), g.camera("pinhole", W, H, FX, FY, CX + 1.1, CY - 0.4), RIG.R, RIG.T)  # fmt: skip
    with pytest.raises(InputError, match="3 images"):
        g.read_calib(folder)
    for target in (folder, folder / f"cameras.{'bin' if binary else 'txt'}"):
        _same(g.read_calib(target, names=("a.png", "b.png")), ref, atol=1e-7)
    # c as the left view: X_r = R_r R_l^T (X_l - t_l) + t_r, checked on a point
    rig = g.read_calib(folder, "colmap", names=("c.png", "b.png"))
    rc, tc = _rot(0.1, 0, 0), np.array([1.0, 2.0, 3.0])
    x_world = np.array([0.3, -0.2, 2.0])
    x_c = rc @ x_world + tc
    np.testing.assert_allclose(rig.R @ x_c + rig.T, RIG.R @ x_world + RIG.T, atol=1e-7)
    with pytest.raises(InputError, match="not in the model"):
        g.read_calib(folder, names=("a.png", "zzz.png"))


@pytest.mark.skipif(
    not (DATA / "stereo/ETH3D/two_view_training/terrace_1s/cameras.txt").exists(), reason="ETH3D data not present"
)
def test_colmap_real_eth3d_pair():
    """ETH3D two_view_training/terrace_1s: PINHOLE 713 x 434, two images 0.0599 m apart in x (read-only)."""
    rig = g.read_calib(DATA / "stereo/ETH3D/two_view_training/terrace_1s/cameras.txt")
    assert (rig.left.cx, rig.left.cy) == pytest.approx((376.291 - 0.5, 227.53 - 0.5))
    assert (rig.left.fx, rig.left.size) == (711.499, (713, 434))
    np.testing.assert_allclose(rig.R, np.eye(3), atol=1e-12)
    assert rig.T[0] == pytest.approx(1.83617 - 1.89611, abs=1e-9)
    assert abs(rig.T[1]) < 1e-9
    assert abs(rig.T[2]) < 1e-9


@pytest.mark.skipif(
    not (DATA / "multiview/dtu_2dgs/DTU/scan24/sparse/0/cameras.bin").exists(), reason="DTU data not present"
)
def test_colmap_real_binary_model():
    folder = DATA / "multiview/dtu_2dgs/DTU/scan24/sparse/0"
    txt = g.read_calib(folder / "cameras.txt", names=_first_two(folder))
    bin_ = g.read_calib(folder / "cameras.bin", names=_first_two(folder))
    _same(bin_, txt, atol=1e-2)  # cameras.txt of this model is rounded to 2 decimals


def _first_two(folder):
    names = [
        ln.split()[9] for ln in (folder / "images.txt").read_text().splitlines()[4::2] if ln and not ln.startswith("#")
    ]
    return names[0], names[1]


@pytest.mark.parametrize("fmt", ["colmap", "oak", "realsense", "ros", "euroc", "kalibr", "tum", "opencv"])
def test_principal_point_convention_per_format(tmp_path, fmt):
    """A camera whose principal point is the image centre in the file's own convention reads as ((W-1)/2, (H-1)/2)."""
    centre_file = {"colmap": (W / 2, H / 2)}.get(fmt, ((W - 1) / 2, (H - 1) / 2))  # COLMAP: pixel centres at +0.5
    cx, cy = centre_file
    cam = None
    if fmt == "colmap":
        folder = tmp_path / "m"
        folder.mkdir()
        (folder / "cameras.txt").write_text(f"1 PINHOLE {W} {H} {FX} {FX} {cx} {cy}\n")
        (folder / "images.txt").write_text("1 1 0 0 0 0 0 0 1 a\n\n2 1 0 0 0 -0.1 0 0 1 b\n\n")
        rig = g.read_calib(folder)
    elif fmt == "oak":
        e = {"intrinsicMatrix": [[FX, 0, cx], [0, FX, cy], [0, 0, 1]], "distortionCoeff": [], "width": W, "height": H,
             "extrinsics": {"rotationMatrix": np.eye(3).tolist(), "toCameraSocket": 2, "translation": {"x": -10, "y": 0, "z": 0}}}  # fmt: skip
        p = tmp_path / "o.json"
        p.write_text(json.dumps({"cameraData": [[1, e], [2, {**e, "extrinsics": {}}]]}))
        rig = g.read_calib(p)
    elif fmt == "realsense":
        s = {"width": W, "height": H, "intrinsics": {"fx": FX, "fy": FX, "ppx": cx, "ppy": cy, "coeffs": [0] * 5}}
        p = tmp_path / "r.json"
        p.write_text(
            json.dumps(
                {
                    "infrared1": s,
                    "infrared2": s,
                    "extrinsics": {"rotation": np.eye(3).ravel().tolist(), "translation": [-0.1, 0, 0]},
                }
            )
        )
        rig = g.read_calib(p)
    elif fmt == "ros":
        k = [[FX, 0, cx], [0, FX, cy], [0, 0, 1]]
        for n, tx in (("l", 0), ("r", -FX * 0.1)):
            (tmp_path / f"{n}.yaml").write_text(
                f"image_width: {W}\nimage_height: {H}\ncamera_matrix: {_flow(k)}\ndistortion_coefficients: \n  rows: 1\n  cols: 5\n  data: [0, 0, 0, 0, 0]\n"
                f"rectification_matrix: {_flow(np.eye(3))}\nprojection_matrix: {_flow([[FX, 0, cx, tx], [0, FX, cy, 0], [0, 0, 1, 0]])}\n"
            )  # fmt: skip
        rig = g.read_calib(tmp_path / "l.yaml", right=tmp_path / "r.yaml")
    elif fmt == "euroc":
        c = g.camera("pinhole", W, H, FX, FX, cx, cy)
        for n, t in (("cam0", np.eye(4)), ("cam1", _transform(np.eye(3), [0.1, 0, 0]))):
            (tmp_path / n).mkdir()
            (tmp_path / n / "sensor.yaml").write_text(_euroc_text(c, t).replace(str(list(DIST)), "[0, 0, 0, 0]"))
        rig = g.read_calib(tmp_path / "cam0" / "sensor.yaml")
    elif fmt in ("kalibr", "tum"):
        t = _transform(np.eye(3), [-0.1, 0, 0])
        p = tmp_path / "c.yaml"
        p.write_text("".join(f"cam{i}:\n  camera_model: pinhole\n  intrinsics: [{FX}, {FX}, {cx}, {cy}]\n  distortion_model: none\n  distortion_coeffs: []\n  resolution: [{W}, {H}]\n" + ("  T_cn_cnm1:\n" + "".join(f"    - {r.tolist()}\n" for r in t) if i else "") for i in range(2)))  # fmt: skip
        rig = g.read_calib(p, fmt)
    else:
        p = tmp_path / "s.yml"
        p.write_text(f"%YAML:1.0\n---\nM1: !!opencv-matrix\n   rows: 3\n   cols: 3\n   dt: d\n   data: [{FX}, 0, {cx}, 0, {FX}, {cy}, 0, 0, 1]\nD1: !!opencv-matrix\n   rows: 1\n   cols: 5\n   dt: d\n   data: [0, 0, 0, 0, 0]\nM2: !!opencv-matrix\n   rows: 3\n   cols: 3\n   dt: d\n   data: [{FX}, 0, {cx}, 0, {FX}, {cy}, 0, 0, 1]\nD2: !!opencv-matrix\n   rows: 1\n   cols: 5\n   dt: d\n   data: [0, 0, 0, 0, 0]\nR: !!opencv-matrix\n   rows: 3\n   cols: 3\n   dt: d\n   data: [1, 0, 0, 0, 1, 0, 0, 0, 1]\nT: !!opencv-matrix\n   rows: 3\n   cols: 1\n   dt: d\n   data: [-0.1, 0, 0]\nimage_width: {W}\nimage_height: {H}\n")  # fmt: skip
        rig = g.read_calib(p)
    cam = rig.left
    assert (cam.cx, cam.cy) == pytest.approx(((W - 1) / 2, (H - 1) / 2))
    rect = g.rectify(rig)
    np.testing.assert_allclose(
        rect.P1[:2, 2], [(W - 1) / 2, (H - 1) / 2], atol=1e-6
    )  # the rectified pair keeps the centre


def test_device_extrinsics_with_swapped_cameras_raise_and_a_wrong_unit_warns(tmp_path):
    """Not verified on a device: the vendors' docs differ on 'from' and 'to', so a T_x > 0 file raises.

    DepthAI (docs.luxonis.com, calibration `extrinsics`: translation in cm toward `toCameraSocket`) and librealsense
    (`rs2_extrinsics`, metres) both describe a transform between the two sensors; only the sign convention differs.
    """
    inv = np.linalg.inv(_transform(RIG.R, RIG.T))  # the right-to-left reading of the same rig
    streams = {
        n: {
            "width": W,
            "height": H,
            "intrinsics": {"fx": FX, "fy": FY, "ppx": c.cx, "ppy": c.cy, "coeffs": [*DIST, 0.0]},
        }
        for n, c in (("infrared1", RIG.left), ("infrared2", RIG.right))
    }
    data = {
        **streams,
        "extrinsics": {"rotation": inv[:3, :3].ravel(order="F").tolist(), "translation": inv[:3, 3].tolist()},
    }
    path = tmp_path / "rs.json"
    path.write_text(json.dumps(data))
    with pytest.raises(InputError, match="swapped"):
        g.read_calib(path)
    data["extrinsics"] = {"rotation": RIG.R.ravel(order="F").tolist(), "translation": (RIG.T * 1000).tolist()}  # mm
    path.write_text(json.dumps(data))
    with pytest.warns(UserWarning, match="wrong unit"):
        g.read_calib(path)


def test_colmap_images_txt_tolerates_stripped_point_lines_and_spaced_names(tmp_path):
    cams = "# Camera list with one line of data per camera:\n1 PINHOLE 64 48 50 50 32.5 24.5\n2 PINHOLE 64 48 50 50 32.5 24.5\n"
    imgs = "# Image list\n1 1 0 0 0 0 0 0 1 left cam.png\n2 1 0 0 0 -0.1 0 0 2 right cam.png\n"  # no point lines at all
    (tmp_path / "cameras.txt").write_text(cams)
    (tmp_path / "images.txt").write_text(imgs)
    rig = g.read_calib(tmp_path, "colmap", names=("left cam.png", "right cam.png"))
    np.testing.assert_allclose(rig.T, [-0.1, 0, 0], atol=1e-12)
    imgs_pts = "# Image list\n1 1 0 0 0 0 0 0 1 left cam.png\n10 20 1 30 40 2\n2 1 0 0 0 -0.1 0 0 2 right cam.png\n\n"
    (tmp_path / "images.txt").write_text(imgs_pts)
    np.testing.assert_allclose(
        g.read_calib(tmp_path, "colmap", names=("left cam.png", "right cam.png")).T, [-0.1, 0, 0], atol=1e-12
    )


def _aria_project(p, xyz):
    """Fisheye624 as projectaria_tools `FisheyeRadTanThinPrism.h` writes it, in plain numpy (the reference)."""
    f, cx, cy, k, (p0, p1), s = p[0], p[1], p[2], p[3:9], p[9:11], p[11:15]
    a, b = xyz[..., 0] / xyz[..., 2], xyz[..., 1] / xyz[..., 2]
    r = np.hypot(a, b)
    th = np.arctan(r)
    radial = 1 + sum(ki * th ** (2 * (i + 1)) for i, ki in enumerate(k))
    xr, yr = radial * th / r * a, radial * th / r * b
    rd2 = xr * xr + yr * yr
    xt = (2 * xr * xr + rd2) * p0 + 2 * xr * yr * p1
    yt = (2 * yr * yr + rd2) * p1 + 2 * xr * yr * p0
    xp, yp = s[0] * rd2 + s[1] * rd2 * rd2, s[2] * rd2 + s[3] * rd2 * rd2
    return np.stack([f * (xr + xt + xp) + cx, f * (yr + yt + yp) + cy], -1)


ARIA_PARAMS = (
    [241.2, 318.9, 241.7, 0.41, -0.47, -0.12, 1.33, -1.65, 0.62, 1.2e-3, -3e-4, -1.1e-3, 2e-4, 4e-4, -1e-4],
    [242.0, 320.4, 238.1, 0.40, -0.45, -0.10, 1.30, -1.60, 0.60, -8e-4, 5e-4, 9e-4, -2e-4, -3e-4, 1e-4],
)


def _aria_entry(label, params, rot, t, *, config=True):
    w, x, y, z = _quat(rot)
    entry = {
        "Label": label,
        "Projection": {"Name": "FisheyeRadTanThinPrism", "Params": list(params)},
        "T_Device_Camera": {"Translation": list(t), "UnitQuaternion": [w, [x, y, z]]},
        "SerialNumber": "0",
    }
    if config:
        entry["ConfigData"] = {"ImageWidth": W, "ImageHeight": H, "MaxSolidAngle": 1.4}
    return entry


def _quat(r):
    """Unit quaternion (w, x, y, z) of a rotation matrix (w > 0)."""
    w = np.sqrt(max(0.0, 1 + np.trace(r))) / 2
    return w, (r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w)


def test_aria_device_calibration(tmp_path):
    """Synthetic Aria JSON: Fisheye624 projection equals the projectaria_tools formula; poses chain to X_r."""
    t_dl, t_dr = (
        _transform(_rot(0.02, -0.4, 0.01), [-0.06, -0.01, 0.002]),
        _transform(_rot(-0.01, 0.4, 0.0), [0.06, -0.01, 0.0]),
    )
    cams = [
        _aria_entry("camera-et-left", ARIA_PARAMS[0], np.eye(3), [0, 0, 0]),
        _aria_entry("camera-slam-left", ARIA_PARAMS[0], t_dl[:3, :3], t_dl[:3, 3]),
        _aria_entry("camera-slam-right", ARIA_PARAMS[1], t_dr[:3, :3], t_dr[:3, 3]),
    ]
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"DeviceClassInfo": {"DeviceClass": "Aria"}, "CameraCalibrations": cams}))
    rig = g.read_calib(path)
    assert isinstance(rig.left, g.Fisheye624)
    assert rig.left.size == (H, W)
    assert (rig.left.fx, rig.left.fy, rig.left.cx, rig.right.s3) == (241.2, 241.2, 318.9, 1e-4)
    x = np.linalg.inv(t_dr) @ t_dl
    np.testing.assert_allclose(rig.R, x[:3, :3], atol=1e-12)
    np.testing.assert_allclose(rig.T, x[:3, 3], atol=1e-12)
    rng = np.random.default_rng(0)
    xyz = np.concatenate([rng.uniform(-1.5, 1.5, (500, 2)), np.ones((500, 1))], -1)  # up to ~65 degrees off axis
    for cam, params in ((rig.left, ARIA_PARAMS[0]), (rig.right, ARIA_PARAMS[1])):
        pix, ok = cam.project(xyz)
        assert ok.all()
        np.testing.assert_allclose(pix, _aria_project(np.array(params), xyz), atol=1e-9)
        rays, ok = cam.unproject(pix)
        assert ok.all()
        np.testing.assert_allclose(rays, xyz / np.linalg.norm(xyz, axis=-1, keepdims=True), atol=1e-7)
    no_size = tmp_path / "no_size.json"
    no_size.write_text(
        json.dumps(
            {
                "CameraCalibrations": [
                    _aria_entry(c["Label"], c["Projection"]["Params"], np.eye(3), [-0.05 + 0.1 * i, 0, 0], config=False)
                    for i, c in enumerate(cams[1:])
                ]
            }
        )
    )
    assert g.read_calib(no_size).right.size == (480, 640)  # the size projectaria_tools gives camera-slam-*
    assert g.read_calib(no_size, size=(H, W)).right.size == (H, W)
    with pytest.raises(InputError, match="names="):
        g.read_calib(path, names=("camera-rgb", "camera-slam-right"))
    other = g.read_calib(path, names=("camera-slam-left", "camera-et-left"))  # any two labels
    np.testing.assert_allclose(other.T, t_dl[:3, 3], atol=1e-12)


@pytest.mark.parametrize("fmt", ["kitti-unrectified", "opencv", "realsense"])
def test_calibration_files_keep_width_height_and_read_as_height_width(tmp_path, fmt):
    """Files store (width, height); the API reads (height, width): a portrait camera survives the round trip."""
    w, h = 480, 640  # portrait, so a swap would show
    if fmt == "kitti-unrectified":
        path = tmp_path / "calib_cam_to_cam.txt"
        rows = []
        for i, tx in ((2, 0.0), (3, -0.1)):
            rows += [f"S_0{i}: {w} {h}", f"K_0{i}: 400 0 239.5 0 401 319.5 0 0 1", f"D_0{i}: 0 0 0 0 0"]
            rows += [f"R_0{i}: 1 0 0 0 1 0 0 0 1", f"T_0{i}: {tx} 0 0"]
        path.write_text("\n".join(rows) + "\n")
    elif fmt == "opencv":
        path = tmp_path / "stereo.yml"
        k = _mat([[400, 0, 239.5], [0, 401, 319.5], [0, 0, 1]])
        body = {"image_width": w, "image_height": h, "K1": k, "D1": _mat([[0] * 5]), "K2": k, "D2": _mat([[0] * 5])}
        body |= {"R": _mat(np.eye(3)), "T": _mat([[-0.1], [0], [0]])}
        lines = ["%YAML:1.0"] + [
            f"{key}: {value}"
            if not isinstance(value, dict)
            else f"{key}: !!opencv-matrix\n   rows: {value['rows']}\n   cols: {value['cols']}\n   dt: d\n   data: {value['data']}"
            for key, value in body.items()
        ]
        path.write_text("\n".join(lines) + "\n")
    else:
        path = tmp_path / "profile.json"
        intr = {"fx": 400.0, "fy": 401.0, "ppx": 239.5, "ppy": 319.5, "coeffs": [0] * 5, "model": "brown_conrady"}
        streams = {n: {"width": w, "height": h, "intrinsics": intr} for n in ("infrared1", "infrared2")}
        path.write_text(
            json.dumps(streams | {"extrinsics": {"rotation": np.eye(3).ravel().tolist(), "translation": [-0.05, 0, 0]}})
        )
    rig = g.read_calib(path, fmt)
    assert rig.left.size == (h, w)
    assert (rig.left.width, rig.left.height, rig.left.cx, rig.left.cy) == (w, h, 239.5, 319.5)
    assert g.read_calib(path, fmt, size=(h, w)).left.size == (h, w)
    with pytest.raises(InputError, match=rf"did you mean size=\({h}, {w}\)"):
        g.read_calib(path, fmt, size=(w, h))
    assert g.rectify(rig).size == (h, w)


def test_colmap_camera_models_and_pixel_origin():
    size, path = (64, 48), Path("cameras.txt")
    simple = _colmap_camera("SIMPLE_RADIAL", size, [50.0, 32.0, 24.0, -0.1], path)
    assert type(simple).model_id == "pinhole-opencv"
    assert simple.coefficients[0] == -0.1
    assert not any(simple.coefficients[1:])
    assert (simple.cx, simple.cy) == (31.5, 23.5)
    assert type(_colmap_camera("RADIAL_FISHEYE", size, [50.0, 32.0, 24.0, 0.01, 0.002], path)).model_id == "kb4"
    assert type(_colmap_camera("FOV", size, [50.0, 51.0, 32.0, 24.0, 0.9], path)).model_id == "fov"
    assert type(_colmap_camera("FOV", size, [50.0, 51.0, 32.0, 24.0, 0.0], path)).model_id == "pinhole"
    p = [50.0, 51.0, 32.0, 24.0, 0.02, -0.004, 1e-3, -2e-3, 3e-4, -1e-4, 2e-3, -1e-3]
    cam = _colmap_camera("THIN_PRISM_FISHEYE", size, p, path)
    ray = np.array([[0.3, -0.2, 1.0], [-0.5, 0.4, 0.8]])
    px, ok = cam.project(torch.from_numpy(ray))
    # COLMAP ThinPrismFisheyeCameraModel::WorldToImage (pixel-corner origin), then the OpenCV shift
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 = p
    r = np.hypot(ray[:, 0], ray[:, 1])
    theta = np.arctan2(r, ray[:, 2])
    u, v = theta * ray[:, 0] / r, theta * ray[:, 1] / r
    t2 = u * u + v * v
    radial = k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4
    du = u * radial + 2 * p1 * u * v + p2 * (t2 + 2 * u * u) + sx1 * t2
    dv = v * radial + 2 * p2 * u * v + p1 * (t2 + 2 * v * v) + sy1 * t2
    want = np.stack([fx * (u + du) + cx - 0.5, fy * (v + dv) + cy - 0.5], -1)
    assert bool(ok.all())
    np.testing.assert_allclose(px.numpy(), want, atol=1e-9)


def test_realsense_modified_brown_and_kb4_coefficients(tmp_path):
    def read(model, coeffs):
        s = {"width": W, "height": H, "intrinsics": {"fx": FX, "fy": FY, "ppx": W / 2, "ppy": H / 2, "coeffs": coeffs, "model": model}}  # fmt: skip
        path = tmp_path / "rs.json"
        path.write_text(json.dumps({"left": s, "right": s, "extrinsics": {"rotation": np.eye(3).ravel().tolist(), "translation": [-0.1, 0, 0]}}))  # fmt: skip
        return g.read_calib(path, "realsense")

    with pytest.raises(InputError, match=r"modified_brown_conrady.*tangential"):
        read("modified_brown_conrady", [0.1, -0.05, 0.002, -0.003, 0.01])
    assert read("modified_brown_conrady", [0.1, -0.05, 0, 0, 0.01]).left.model_id == "pinhole-opencv"  # = Brown then
    assert read("kannala_brandt4", [0.01, -0.001, 0, 0, 0]).left.model_id == "kb4"  # 5 slots, fifth unused
    with pytest.raises(InputError, match="fifth"):
        read("kannala_brandt4", [0.01, -0.001, 0, 0, 0.5])
