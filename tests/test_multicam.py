"""Pose-built rigs (`StereoRig.from_poses`), Aria files without image sizes / jsonl, nerfstudio `transforms.json`."""

import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from fastforks import InputError
from fastforks.stereo import calibrated as c
from fastforks.stereo import geometry as g

CAM = g.camera("pinhole", 64, 48, 50.0, 50.0, 31.5, 23.5)
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenGL <-> OpenCV camera axes


def _pose(angle, t):
    c, s = np.cos(angle), np.sin(angle)
    m = np.eye(4)
    m[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    m[:3, 3] = t
    return m


LEFT, RIGHT = _pose(0.1, [1.0, 2.0, 3.0]), _pose(0.12, [1.0, 2.0, 3.0] + np.array([0.2, 0.01, 0.0]) * 1)


def _check_chain(rig, left, right, scale=1.0):
    """X_right = R X_left + T for a world point X_w: through the poses and through the rig."""
    xw = np.array([0.5, -0.3, 4.0, 1.0])
    xl, xr = np.linalg.inv(left) @ xw, np.linalg.inv(right) @ xw
    np.testing.assert_allclose(rig.R @ xl[:3] * scale + rig.T, xr[:3] * scale, atol=1e-9)


def test_from_poses_chains_the_poses_and_converts_axes_and_units():
    rig = g.StereoRig.from_poses(CAM, CAM, left_to_world=LEFT, right_to_world=RIGHT)
    _check_chain(rig, LEFT, RIGHT)
    gl = g.StereoRig.from_poses(CAM, CAM, left_to_world=LEFT @ FLIP, right_to_world=RIGHT @ FLIP, axes="opengl")
    np.testing.assert_allclose(gl.R, rig.R, atol=1e-12)
    np.testing.assert_allclose(gl.T, rig.T, atol=1e-12)
    big = [np.array(p) * np.array([1, 1, 1, 1.0]) for p in (LEFT, RIGHT)]
    for p in big:
        p[:3, 3] *= 1000
    in_mm = g.StereoRig.from_poses(CAM, CAM, left_to_world=big[0], right_to_world=big[1], unit="mm")
    np.testing.assert_allclose(in_mm.T, rig.T, atol=1e-9)


def test_from_poses_is_keyword_only_and_rejects_non_rigid_poses():
    with pytest.raises(TypeError):
        g.StereoRig.from_poses(CAM, CAM, LEFT, RIGHT)  # ty: ignore[missing-argument, too-many-positional-arguments]
    scaled = LEFT.copy()
    scaled[:3, :3] *= 2
    with pytest.raises(InputError, match=r"left_to_world is not a rigid pose.*scale"):
        g.StereoRig.from_poses(CAM, CAM, left_to_world=scaled, right_to_world=RIGHT)
    with pytest.raises(InputError, match="right_to_world must be a finite 4x4"):
        g.StereoRig.from_poses(CAM, CAM, left_to_world=LEFT, right_to_world=RIGHT[:3])
    with pytest.raises(InputError, match="unit must be"):
        g.StereoRig.from_poses(CAM, CAM, left_to_world=LEFT, right_to_world=RIGHT, unit="km")
    with pytest.raises(InputError, match="axes"):
        g.StereoRig.from_poses(CAM, CAM, left_to_world=LEFT, right_to_world=RIGHT, axes="llff")
    with pytest.raises(InputError, match="baseline"):
        g.StereoRig.from_poses(CAM, CAM, left_to_world=LEFT, right_to_world=LEFT)


# --- Aria -------------------------------------------------------------------------------------------------------
PARAMS = [241.2, 318.9, 241.7, 0.41, -0.47, -0.12, 1.33, -1.65, 0.62, 1.2e-3, -3e-4, -1.1e-3, 2e-4, 4e-4, -1e-4]


def _entry(label, x):
    return {
        "Label": label,
        "Projection": {"Name": "FisheyeRadTanThinPrism", "Params": PARAMS},
        "T_Device_Camera": {"Translation": [x, 0.0, 0.0], "UnitQuaternion": [1.0, [0.0, 0.0, 0.0]]},
    }


def test_aria_sizes_default_pair_and_jsonl(tmp_path):
    gen1 = {"CameraCalibrations": [_entry("camera-slam-left", 0.0), _entry("camera-slam-right", 0.1)]}
    path = tmp_path / "gen1.json"
    path.write_text(json.dumps(gen1))
    rig = g.read_calib(path)
    assert rig.left.size == (480, 640)  # (height, width)
    assert g.read_calib(path, size=(240, 320)).left.size == (240, 320)
    gen2 = {
        "CameraCalibrations": [
            _entry("slam-front-left", 0.0),
            _entry("slam-front-right", 0.1),
            _entry("camera-rgb", 0.05),
        ]
    }
    path2 = tmp_path / "gen2.json"
    path2.write_text(json.dumps(gen2))
    assert g.read_calib(path2).right.size == (512, 512)  # Gen2's default pair, its size
    assert g.read_calib(path2, names=("slam-front-left", "camera-rgb")).right.size == (3024, 4032)
    # MPS online calibration: one calibration per line, ImageSizes listed in camera order; the first line is read
    line = {"ImageSizes": [[640, 480], [640, 480], [1408, 1408]], **gen1}
    line["CameraCalibrations"].append(_entry("camera-rgb", 0.05))
    jsonl = tmp_path / "online_calibration.jsonl"
    jsonl.write_text(json.dumps(line) + "\n" + json.dumps({"CameraCalibrations": []}) + "\n")
    assert g.read_calib(jsonl).left.size == (480, 640)
    assert g.read_calib(jsonl, names=("camera-slam-left", "camera-rgb")).right.size == (1408, 1408)
    odd = {"CameraCalibrations": [_entry("my-left", 0.0), _entry("my-right", 0.1)]}
    path3 = tmp_path / "odd.json"
    path3.write_text(json.dumps(odd))
    with pytest.raises(InputError, match=r"pass size=\(height, width\)"):
        g.read_calib(path3, names=("my-left", "my-right"))


# --- nerfstudio / instant-ngp -----------------------------------------------------------------------------------
def _transforms(tmp_path, top, frames, name="transforms.json"):
    path = tmp_path / name
    path.write_text(json.dumps({**top, "frames": frames}))
    return path


INTR = {"fl_x": 50.0, "fl_y": 51.0, "cx": 32.0, "cy": 24.0, "w": 64, "h": 48}


def test_nerfstudio_pair_from_opengl_poses(tmp_path):
    frames = [
        {"file_path": f"images/{i}.jpg", "transform_matrix": (p @ FLIP).tolist()}
        for i, p in enumerate((LEFT, RIGHT, _pose(0.3, [5, 5, 5])))
    ]
    path = _transforms(tmp_path, {**INTR, "k1": 0.01, "k2": -0.002, "p1": 0.001, "p2": 0.0}, frames)
    rig = g.read_calib(path, names=("images/0.jpg", "1.jpg"))  # content detected; a file name is enough
    assert isinstance(rig, g.StereoRig)
    _check_chain(rig, LEFT, RIGHT)
    assert (rig.left.size, rig.left.cx, rig.left.cy, rig.left.fy) == (
        (48, 64),
        31.5,
        23.5,
        51.0,
    )  # +0.5 centres -> integer
    assert rig.left.k1 == 0.01
    assert g.read_calib(path, names=("images/0.jpg", "images/1.jpg"), unit="cm").baseline == pytest.approx(
        rig.baseline / 100
    )
    with pytest.raises(InputError, match="3 frames; pick two"):
        g.read_calib(path)
    with pytest.raises(InputError, match="not in the file"):
        g.read_calib(path, names=("images/0.jpg", "9.jpg"))


def test_nerfstudio_fisheye_per_frame_intrinsics_and_errors(tmp_path):
    frames = [
        {"file_path": "a.jpg", "transform_matrix": (LEFT @ FLIP).tolist()},
        {"file_path": "b.jpg", "transform_matrix": (RIGHT @ FLIP).tolist(), "fl_x": 60.0, "fl_y": 60.0},
    ]
    top = {**INTR, "camera_model": "OPENCV_FISHEYE", "k1": 0.03, "k2": 0.006, "k3": 0.0004, "k4": 0.0001}
    rig = g.read_calib(_transforms(tmp_path, top, frames))  # exactly two frames: no names
    assert isinstance(rig.left, g.KB4)
    assert (rig.left.fx, rig.right.fx, rig.right.k4) == (50.0, 60.0, 0.0001)
    bad = _transforms(tmp_path, {**top, "camera_model": "EQUIRECTANGULAR"}, frames, "bad.json")
    with pytest.raises(InputError, match="camera_model 'EQUIRECTANGULAR' is not supported"):
        g.read_calib(bad)
    scaled = LEFT.copy()
    scaled[:3, :3] *= 2
    skewed = [dict(frames[0], transform_matrix=scaled.tolist()), frames[1]]
    with pytest.raises(InputError, match="not a rigid pose"):
        g.read_calib(_transforms(tmp_path, top, skewed, "skew.json"))
    no_size = {k: v for k, v in top.items() if k not in ("w", "h")}
    path = _transforms(tmp_path, no_size, frames, "nosize.json")
    with pytest.raises(InputError, match=r"pass size=\(height, width\)"):
        g.read_calib(path)
    assert g.read_calib(path, size=(48, 64)).left.size == (48, 64)
    assert g.read_calib(path, size=(48, 64)).left.width == 64


def test_forward_baseline_error_has_direction():
    rig = g.StereoRig(CAM, CAM, np.eye(3), np.array([-0.01, -0.05, -0.2]))
    with pytest.raises(InputError, match=r"along the optical axis.*\(x, y, z\) = \(-0.05"):
        g.rectify(rig)


def test_poor_rectification_warns_only_for_large_turns(caplog):
    def rig(deg):
        a = np.radians(deg)
        rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        return g.StereoRig(CAM, CAM, rot, np.array([-0.1, 0.0, 0.0]))

    with caplog.at_level(logging.WARNING, logger="fastforks.stereo.geometry.rectify"):
        g.rectify(rig(5.0))
        assert not caplog.records
        g.rectify(rig(30.0))
    poor = [r.getMessage() for r in caplog.records if "poor rectification" in r.getMessage()]
    assert len(poor) == 1
    assert "30 degrees" in poor[0]


def test_poor_rectification_turn_check_skips_fisheye(caplog):
    a = np.radians(30.0)
    rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    fish = g.camera("fisheye-equidistant", 64, 48, 20.0, 20.0, 31.5, 23.5)  # ~180 degrees across
    with caplog.at_level(logging.WARNING, logger="fastforks.stereo.geometry.rectify"):
        g.rectify(g.StereoRig(fish, fish, rot, np.array([-0.1, 0.0, 0.0])), focal=20.0)
    assert not [r for r in caplog.records if "poor rectification" in r.getMessage()]


def test_resize_warning_is_deduped_by_calibration_values(caplog):
    c._WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="fastforks.stereo.calibrated"):
        for _ in range(3):  # equal values, a new rig object and model each time
            rig = g.StereoRig(CAM, CAM, np.eye(3), np.array([-0.1, 0.0, 0.0]))
            c.resolve_geometry(SimpleNamespace(), rig, (96, 128))
    assert len([r for r in caplog.records if "assuming the images" in r.getMessage()]) == 1
