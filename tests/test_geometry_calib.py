"""`fastforks.stereo.geometry.read_calib` and its YAML subset reader."""

import time
from pathlib import Path
from typing import TypeVar

import numpy as np
import pytest

from fastforks import InputError
from fastforks.stereo import geometry as g
from fastforks.stereo.geometry import _yaml


@pytest.fixture
def cv2():
    """OpenCV, the oracle of the comparison tests: only they skip without it."""
    return pytest.importorskip("cv2")


KITTI_RAW = """calib_time: 09-Jan-2012 13:57:47
corner_dist: 9.950000e-02
S_02: 1.392000e+03 5.120000e+02
K_02: 9.597910e+02 0.000000e+00 6.960217e+02 0.000000e+00 9.569251e+02 2.241806e+02 0.000000e+00 0.000000e+00 1.000000e+00
D_02: -3.691481e-01 1.968681e-01 1.353473e-03 5.677587e-04 -6.770705e-02
R_02: 9.999758e-01 -5.267463e-03 -4.552439e-03 5.251945e-03 9.999804e-01 -3.413835e-03 4.570332e-03 3.389843e-03 9.999838e-01
T_02: 5.956621e-02 2.900141e-04 2.577209e-03
S_rect_02: 1.242000e+03 3.750000e+02
P_rect_02: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03
S_03: 1.392000e+03 5.120000e+02
K_03: 9.037596e+02 0.000000e+00 6.957519e+02 0.000000e+00 9.019653e+02 2.242509e+02 0.000000e+00 0.000000e+00 1.000000e+00
D_03: -3.639558e-01 1.788651e-01 6.029694e-04 -3.922424e-04 -5.382460e-02
R_03: 9.995599e-01 1.699522e-02 -2.431313e-02 -1.704422e-02 9.998531e-01 -1.809756e-03 2.427880e-02 2.223358e-03 9.997028e-01
T_03: -4.731050e-01 5.551470e-03 -5.250882e-03
S_rect_03: 1.242000e+03 3.750000e+02
P_rect_03: 7.215377e+02 0.000000e+00 6.095593e+02 -3.395242e+02 0.000000e+00 7.215377e+02 1.728540e+02 2.199936e+00 0.000000e+00 0.000000e+00 1.000000e+00 2.729905e-03
"""  # noqa: E501 - verbatim KITTI lines
KITTI_ODOMETRY = """P0: 7.188560e+02 0.000000e+00 6.071928e+02 0.000000e+00 0.000000e+00 7.188560e+02 1.852157e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P1: 7.188560e+02 0.000000e+00 6.071928e+02 -3.861448e+02 0.000000e+00 7.188560e+02 1.852157e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P2: 7.188560e+02 0.000000e+00 6.071928e+02 4.538225e+01 0.000000e+00 7.188560e+02 1.852157e+02 -1.130887e-01 0.000000e+00 0.000000e+00 1.000000e+00 3.779761e-03
P3: 7.188560e+02 0.000000e+00 6.071928e+02 -3.372877e+02 0.000000e+00 7.188560e+02 1.852157e+02 2.369057e+00 0.000000e+00 0.000000e+00 1.000000e+00 4.915215e-03
"""  # noqa: E501 - verbatim KITTI lines
MIDDLEBURY = """cam0=[4161.221 0 1445.577; 0 4161.221 984.686; 0 0 1]
cam1=[4161.221 0 1654.636; 0 4161.221 984.686; 0 0 1]
doffs=209.059
baseline=176.252
width=2880
height=1988
ndisp=280
"""
KALIBR = """cam0:
  camera_model: pinhole
  intrinsics: [461.6, 460.3, 363.0, 248.1]  # fu fv pu pv
  distortion_model: radtan
  distortion_coeffs: [-0.2917, 0.08228, 5.333e-05, -0.0001578]
  resolution: [752, 480]
  rostopic: /cam0/image_raw
cam1:
  camera_model: ds
  intrinsics: [-0.2, 0.6, 300.0, 300.0, 376.0, 240.0]
  distortion_model: none
  distortion_coeffs: []
  resolution: [752, 480]
  T_cn_cnm1:
  - [0.999997, 0.002312, 0.000376, -0.110074]
  - [-0.002312, 0.999995, 0.002098, 0.000399]
  - [-0.000371, -0.002099, 0.999997, -0.000853]
  - [0.0, 0.0, 0.0, 1.0]
  rostopic: /cam1/image_raw
cam2:
  camera_model: pinhole
  intrinsics: [460.0, 460.0, 376.0, 240.0]
  distortion_model: equidistant
  distortion_coeffs: [0.01, -0.002, 0.0003, -0.0001]
  resolution: [752, 480]
  T_cn_cnm1:
    - [1.0, 0.0, 0.0, -0.1]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]
"""

ZED = """[LEFT_CAM_2K]
fx=1066.3
fy=1065.9
cx=1123.8
cy=627.6
k1=-0.0571
k2=0.0291
p1=0.0003
p2=-0.0002
k3=-0.0114

[RIGHT_CAM_2K]
fx=1067.1
fy=1066.5
cx=1101.2
cy=636.9
k1=-0.0565
k2=0.0284
p1=0.0001
p2=0.0004
k3=-0.0109

[LEFT_CAM_HD]
fx = 533.15
fy = 532.95
cx = 649.9
cy = 360.8
k1 = -0.0571
k2 = 0.0291
p1 = 0.0003
p2 = -0.0002
k3 = -0.0114

[RIGHT_CAM_HD]
fx = 533.55
fy = 533.25
cx = 638.6
cy = 365.45
k1 = -0.0565
k2 = 0.0284
p1 = 0.0001
p2 = 0.0004
k3 = -0.0109

[STEREO]
Baseline=119.98
TY=0.0412
TZ=-0.2265
CV_2K=0.0052
CV_HD=0.0052
RX_2K=0.0013
RX_HD=0.0013
RZ_2K=-0.0009
RZ_HD=-0.0009

[MISC]
Sensor_ID=0
"""


DATA = Path(__file__).parent / "data"


def _tx(p03, p23, cx, f):
    return (p03 - cx * p23) / f  # exact t_x of a KITTI projection (P[2, 3] = t_z)


RigT = TypeVar("RigT", g.StereoRig, g.RectifiedRig)


def _read(cls: type[RigT], path, *args, **kwargs) -> RigT:
    rig = g.read_calib(path, *args, **kwargs)
    assert isinstance(rig, cls)
    return rig


def _write(tmp_path, text, name="calib.txt"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_kitti_rectified_and_unrectified(tmp_path):
    path = _write(tmp_path, KITTI_RAW)
    rig = _read(g.RectifiedRig, path)
    assert isinstance(rig, g.RectifiedRig)
    assert rig.f == pytest.approx(721.5377)
    assert (rig.cx, rig.cy, rig.doffs, rig.size) == (609.5593, 172.854, 0.0, (375, 1242))
    assert rig.baseline == pytest.approx(
        _tx(44.85728, 2.745884e-3, 609.5593, 721.5377) - _tx(-339.5242, 2.729905e-3, 609.5593, 721.5377)
    )
    raw = _read(g.StereoRig, path, "kitti-unrectified")
    assert isinstance(raw, g.StereoRig)
    assert isinstance(raw.left, g.PinholeOpenCV)
    assert raw.left.size == (512, 1392)
    assert raw.left.k3 == pytest.approx(-6.770705e-02)
    assert raw.baseline == pytest.approx(0.5327, abs=1e-3)
    np.testing.assert_allclose(raw.R @ raw.R.T, np.eye(3), atol=1e-12)
    assert g.rectify(raw).baseline == pytest.approx(raw.baseline)
    with pytest.raises(InputError, match="P_rect_00"):
        g.read_calib(path, cameras=(0, 1))


def test_kitti360_front_pair(tmp_path):
    """KITTI-360 `perspective.txt` has the KITTI raw layout with cameras 0 and 1 only: they are the default."""
    path = _write(tmp_path, KITTI_RAW.replace("_02:", "_00:").replace("_03:", "_01:"), "perspective.txt")
    assert _read(g.RectifiedRig, path).baseline == pytest.approx(
        _read(g.RectifiedRig, _write(tmp_path, KITTI_RAW)).baseline
    )
    assert isinstance(_read(g.StereoRig, path, "kitti-unrectified").left, g.PinholeOpenCV)
    with pytest.raises(InputError, match="P_rect_02"):
        g.read_calib(path, cameras=(2, 3))


def test_zed_conf(cv2, tmp_path):
    path = _write(tmp_path, ZED, "SN12345.conf")
    with pytest.raises(InputError, match=r"pass size=\(height, width\).*2K \(1242, 2208\), HD \(720, 1280\)"):
        g.read_calib(path)
    with pytest.raises(InputError, match="not a resolution in the file"):
        g.read_calib(path, size=(1080, 1920))
    with pytest.raises(InputError, match=r"did you mean \(720, 1280\)"):
        g.read_calib(path, size=(1280, 720))
    rig = _read(g.StereoRig, path, size=(720, 1280))
    assert isinstance(rig.left, g.PinholeOpenCV)
    assert (rig.left.size, rig.left.fx, rig.right.cy, rig.right.k3) == ((720, 1280), 533.15, 365.45, -0.0109)
    np.testing.assert_allclose(rig.R, cv2.Rodrigues(np.array([0.0013, 0.0052, -0.0009]))[0], atol=1e-15)
    np.testing.assert_allclose(rig.T, [-0.11998, 0.0000412, -0.0002265], atol=1e-15)
    assert g.rectify(rig).baseline == pytest.approx(0.11998, rel=1e-3)
    single = ZED.split("[LEFT_CAM_HD]", maxsplit=1)[0] + "[STEREO]" + ZED.split("[STEREO]")[1]
    assert _read(g.StereoRig, _write(tmp_path, single, "SN1.conf"), unit="cm").T[0] == pytest.approx(-1.1998)


def test_kitti_odometry(tmp_path):
    path = _write(tmp_path, KITTI_ODOMETRY)
    rig = _read(g.RectifiedRig, path, size=(376, 1241))
    assert isinstance(rig, g.RectifiedRig)
    assert rig.baseline == pytest.approx(
        _tx(45.38225, 3.779761e-3, 607.1928, 718.856) - _tx(-337.2877, 4.915215e-3, 607.1928, 718.856)
    )
    assert rig.baseline == pytest.approx(0.53329, abs=1e-5)  # the P[0, 3] difference alone gives 0.53233
    assert rig.size == (376, 1241)
    gray = _read(g.RectifiedRig, path, cameras=(0, 1))
    assert gray.baseline == pytest.approx(386.1448 / 718.856)
    assert gray.size is None
    with pytest.raises(InputError, match="right of the first"):
        g.read_calib(path, cameras=(3, 2))


def test_middlebury_and_eth3d(tmp_path):
    rig = _read(g.RectifiedRig, _write(tmp_path, MIDDLEBURY))
    assert (rig.f, rig.cx, rig.cy, rig.doffs, rig.size) == (4161.221, 1445.577, 984.686, 209.059, (1988, 2880))
    assert rig.baseline == pytest.approx(0.176252)
    assert rig.depth(np.full((1988, 2880), 100.0))[0, 0] == pytest.approx(4161.221 * 0.176252 / 309.059)
    eth3d = "cam0=[541.764 0 553.869; 0 541.764 232.396; 0 0 1]\ncam1=[541.764 0 553.869; 0 541.764 232.396; 0 0 1]\n"
    eth3d += "doffs=0\nbaseline=59.9475\nwidth=942\nheight=489\nndisp=64\nisint=0\n"
    assert g.read_calib(_write(tmp_path, eth3d, "eth3d.txt")).baseline == pytest.approx(0.0599475)
    with pytest.raises(InputError, match="not rectified"):
        g.read_calib(_write(tmp_path, MIDDLEBURY.replace("984.686; 0 0 1]\ndoffs", "990.0; 0 0 1]\ndoffs"), "bad.txt"))


def test_middlebury_doffs_and_skew_are_checked(tmp_path):
    with pytest.raises(InputError, match="doffs = 100 contradicts"):
        g.read_calib(_write(tmp_path, MIDDLEBURY.replace("doffs=209.059", "doffs=100"), "doffs.txt"))
    off_by_rounding = MIDDLEBURY.replace("doffs=209.059", "doffs=209.4")
    assert g.read_calib(_write(tmp_path, off_by_rounding, "round.txt")).doffs == pytest.approx(209.4)
    with pytest.raises(InputError, match="skew"):
        g.read_calib(_write(tmp_path, MIDDLEBURY.replace("cam0=[4161.221 0", "cam0=[4161.221 3.0"), "skew.txt"))
    skewed = KITTI_ODOMETRY.replace("P2: 7.188560e+02 0.000000e+00", "P2: 7.188560e+02 5.000000e+00")
    with pytest.raises(InputError, match="skew"):
        g.read_calib(_write(tmp_path, skewed, "calib.txt"))


@pytest.mark.parametrize("ext", [".yml", ".xml"])
def test_opencv_file_storage(cv2, tmp_path, ext):
    path = str(tmp_path / f"stereo{ext}")
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", 640)
    fs.write("image_height", 480)
    fs.write("M1", np.array([[500.0, 0, 320], [0, 501, 240], [0, 0, 1]]))
    fs.write("D1", np.array([[0.1, -0.05, 0.001, 0.002, 0.01, 0.003, -0.002, 0.001]]))
    fs.write("M2", np.array([[502.0, 0, 322], [0, 503, 241], [0, 0, 1]]))
    fs.write("D2", np.zeros((1, 5)))
    fs.write("R", cv2.Rodrigues(np.array([0.01, 0.02, -0.01]))[0])
    fs.write("T", np.array([[-120.0], [1.0], [0.5]]))
    fs.release()
    rig = _read(g.StereoRig, path, unit="mm")
    assert isinstance(rig.left, g.PinholeOpenCV)
    assert rig.left.k6 == pytest.approx(0.001)
    assert isinstance(rig.right, g.Pinhole)
    assert rig.right.fy == 503
    assert rig.left.size == (480, 640)
    np.testing.assert_allclose(rig.T, [-0.12, 0.001, 0.0005])
    np.testing.assert_allclose(rig.R, cv2.Rodrigues(np.array([0.01, 0.02, -0.01]))[0], atol=1e-12)


def test_opencv_named_keys_and_fisheye(tmp_path):
    text = """%YAML:1.0
---
distortion_model: fisheye
left_camera_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 300., 0., 319.5,
           0., 300., 239.5, 0., 0., 1. ]
left_distortion: !!opencv-matrix
   rows: 1
   cols: 4
   dt: d
   data: [ 0.01, -0.002, 0.0003, -0.0001 ]
right_camera_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 300., 0., 319.5, 0., 300., 239.5, 0., 0., 1. ]
right_distortion: !!opencv-matrix
   rows: 1
   cols: 4
   dt: d
   data: [ 0.01, -0.002, 0.0003, -0.0001 ]
rotation_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 1., 0., 0., 0., 1., 0., 0., 0., 1. ]
translation_vector: !!opencv-matrix
   rows: 3
   cols: 1
   dt: d
   data: [ -0.06, 0., 0. ]
"""
    path = _write(tmp_path, text, "fisheye.yml")
    with pytest.raises(InputError, match="pass size="):
        g.read_calib(path)
    rig = _read(g.StereoRig, path, size=(480, 640))
    assert isinstance(rig.left, g.KB4)
    assert rig.left.k4 == pytest.approx(-0.0001)
    assert rig.baseline == pytest.approx(0.06)


def test_kalibr_camchain(tmp_path):
    path = _write(tmp_path, KALIBR, "camchain.yaml")
    rig = _read(g.StereoRig, path)
    assert isinstance(rig.left, g.PinholeOpenCV)
    assert rig.left.p2 == pytest.approx(-0.0001578)
    assert rig.right == g.DoubleSphere(752, 480, 300.0, 300.0, 376.0, 240.0, xi=-0.2, alpha=0.6)
    assert rig.T[0] == pytest.approx(-0.110074)
    chained = _read(g.StereoRig, path, cameras=(0, 2))
    assert isinstance(chained.right, g.KB4)
    np.testing.assert_allclose(chained.T, rig.T + np.array([-0.1, 0, 0]), atol=1e-5)
    swapped = _read(g.StereoRig, path, cameras=(1, 0))  # cam1 as the left camera: the inverse transform
    assert swapped.left == rig.right
    np.testing.assert_allclose(swapped.R, rig.R.T, atol=1e-5)
    np.testing.assert_allclose(swapped.T, -rig.R.T @ rig.T, atol=1e-6)
    with pytest.raises(InputError, match="swap the two cameras"):
        g.rectify(swapped)
    with pytest.raises(InputError, match="two different camera"):
        g.read_calib(path, cameras=(1, 1))
    omni = KALIBR.replace("camera_model: ds", "camera_model: omni").replace("[-0.2, 0.6, 300.0", "[0.8, 300.0")
    omni = omni.replace("distortion_model: none", "distortion_model: radtan")
    with pytest.raises(InputError, match="later release"):
        g.read_calib(_write(tmp_path, omni, "omni.yaml"))


@pytest.mark.parametrize(
    ("text", "kwargs", "match"),
    [
        ("hello world\n", {}, "cannot tell the calibration format"),
        (MIDDLEBURY, {"format": "csv"}, "unknown calibration format"),
        (MIDDLEBURY, {"cameras": (0, 1)}, "cameras= applies to multi-camera formats"),
        (MIDDLEBURY, {"unit": "inch"}, "unit must be one of"),
        (KITTI_RAW, {"unit": "mm"}, "KITTI calibration is in metres"),
        (MIDDLEBURY, {"size": (100, 100)}, r"differs from the image size in the file \(1988, 2880\)"),
        (MIDDLEBURY.replace("baseline=176.252\n", ""), {}, "missing 'baseline'"),
        (KITTI_ODOMETRY.replace("P2: 7.188560e+02 ", "P2: "), {}, "'P2' needs 12 numbers"),
        ("%YAML:1.0\ncamera_matrix: [1, 2]\n", {}, "no stereo calibration keys"),
    ],
)
def test_read_calib_errors(tmp_path, text, kwargs, match):
    with pytest.raises(InputError, match=match):
        g.read_calib(_write(tmp_path, text), **kwargs)
    with pytest.raises(InputError, match="cannot read calibration file"):
        g.read_calib(tmp_path / "missing.txt")


def test_yaml_subset():
    text = """%YAML:1.0
---
# comment
a: 1   # trailing comment
b: [1, 2.5, -3e-2,
    [4, 'x, y'], .inf]
c:
  d: "quoted # not a comment"
  e:
  - 1
  -  [2, 3]
f: !!opencv-matrix
   rows: 1
g: ~
h: true
"""
    assert _yaml.loads(text) == {
        "a": 1,
        "b": [1, 2.5, -0.03, [4, "x, y"], float("inf")],
        "c": {"d": "quoted # not a comment", "e": [1, [2, 3]]},
        "f": {"rows": 1},
        "g": None,
        "h": True,
    }
    for bad, match in (("a: &x 1\n", "anchors"), ("a: [1, 2\n", "unclosed"), ("a: 1\n  b: 2\n", "indentation")):
        with pytest.raises(InputError, match=match):
            _yaml.loads(bad)


@pytest.mark.parametrize(
    "bad",
    [
        "a: [" + ", ".join(["1"] * 40) + ", 'x\n",  # stray quote: was exponential regex backtracking
        "a: [" + "1, " * 2000 + "]]\n",
        "a: [" + "'x, " * 201 + "]\n",  # odd number of quotes
        "a: [1 2 3 " + "4 " * 2000 + "\n",
    ],
)
def test_yaml_malformed_flow_fails_fast(bad):
    t0 = time.perf_counter()
    with pytest.raises(InputError):
        _yaml.loads(bad)
    assert time.perf_counter() - t0 < 1.0


def test_yaml_deep_nesting_is_an_input_error():
    deep = "".join(f"{' ' * i}k{i}:\n" for i in range(200)) + " " * 200 + "v: 1\n"
    with pytest.raises(InputError, match="nested deeper"):
        _yaml.loads(deep)
    ok = "".join(f"{' ' * i}k{i}:\n" for i in range(10)) + " " * 10 + "v: 1\n"
    node = _yaml.loads(ok)
    for i in range(10):
        assert isinstance(node, dict)
        node = node[f"k{i}"]
    assert node == {"v": 1}


def test_opencv_stereo_rectify_output():
    """A file written by OpenCV 5 (`%YAML 1.2`) with stereoRectify's R1 R2 P1 P2 Q next to the calibration."""
    path = DATA / "opencv_stereo_rectify.yml"
    rig = _read(g.StereoRig, path, size=(720, 1280), unit="mm")  # holds P1 / P2 but is not a KITTI file
    assert isinstance(rig.left, g.PinholeOpenCV)
    assert rig.left.k1 == pytest.approx(-0.28)
    np.testing.assert_allclose(rig.T, [-0.1198, 0.0004, -0.0009])
    stored = _yaml.loads(path.read_text())
    assert isinstance(stored, dict)

    def matrix(key):
        node = stored[key]
        assert isinstance(node, dict)
        assert isinstance(node["data"], list)
        rows, cols = node["rows"], node["cols"]
        assert isinstance(rows, int)
        assert isinstance(cols, int)
        return np.array(node["data"], dtype=np.float64).reshape(rows, cols)

    p1, p2 = matrix("P1"), matrix("P2")
    rect = g.rectify(rig, alpha=0)  # the file was written with stereoRectify(alpha=0)
    np.testing.assert_allclose(rect.R1, matrix("R1"), atol=1e-9)
    np.testing.assert_allclose(rect.P1, p1, atol=0.05)  # OpenCV's 5 undistortion iterations, see test_geometry_rectify
    assert rect.P2[0, 3] * 1000 == pytest.approx(p2[0, 3], rel=1e-4)


@pytest.mark.parametrize("bad", ["a: [1 2, 3]\n", "a: [[1, 2] [3]]\n", "a: ['x' 'y']\n", "a: [1, 2 [3]]\n"])
def test_yaml_flow_sequence_missing_comma(bad):
    with pytest.raises(InputError, match="malformed flow sequence"):
        _yaml.loads(bad)
    assert _yaml.loads("a: [1, [2, 3], 'x y',]\n") == {"a": [1, [2, 3], "x y"]}
