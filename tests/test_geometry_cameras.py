"""Camera models of `fastforks.stereo.geometry`: OpenCV and Rust oracles, closed forms, round trips, errors."""

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from fastforks import InputError
from fastforks.stereo import geometry as g


@pytest.fixture
def cv2():
    """OpenCV, the oracle of the comparison tests: only they skip without it."""
    return pytest.importorskip("cv2")


W, H = 1280, 960
K = (410.0, 405.0, 639.5, 479.5)
OPENCV_14 = (-0.28, 0.07, 0.0012, -0.0008, -0.006, 0.02, -0.01, 0.003, 0.001, -0.0005, 0.0007, 0.0002, 0.01, -0.02)
GOLDENS = json.loads((Path(__file__).parent / "data" / "geometry_goldens.json").read_text())

WIDE = {  # every model with a field of view it supports (degrees, full angle)
    "pinhole": (g.camera("pinhole", W, H, *K), 120),
    "pinhole-opencv": (g.camera("pinhole-opencv", W, H, *K, -0.1, 0.01, 0.001, -0.001, 0.0), 90),
    "kb4": (g.camera("kb4", W, H, *K, -0.013, 0.021, -0.011, 0.0024), 200),
    "fisheye624": (g.camera("fisheye624", W, H, *K, *GOLDENS["fisheye624"]["coefficients"]), 170),
    "double-sphere": (g.DoubleSphere(W, H, *K, xi=-0.2, alpha=0.6), 200),
    "ucm": (g.UCM(W, H, *K, xi=0.9), 190),
    "eucm": (g.EUCM(W, H, *K, alpha=0.6, beta=1.1), 200),
    "fov": (g.FOV(W, H, *K, w=0.9), 160),
}


def _rays(fov_deg: float, n: int = 400, seed: int = 0) -> np.ndarray:
    """Unit rays spread uniformly over directions within `fov_deg / 2` of the optical axis."""
    rng = np.random.default_rng(seed)
    theta = np.radians(fov_deg / 2) * np.sqrt(rng.uniform(0, 1, n))
    phi = rng.uniform(0, 2 * np.pi, n)
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)


def _cv_camera(k=K):
    return np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1.0]])


@pytest.mark.parametrize("count", [4, 5, 8, 12, 14])
def test_pinhole_opencv_matches_cv2(cv2, count):
    coefficients = np.array(OPENCV_14[:count])
    cam = g.camera("pinhole-opencv", W, H, *K, *coefficients)
    rays = _rays(80)
    pixels, valid = cam.project(rays)
    expected = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), _cv_camera(), coefficients)[0][:, 0]
    assert valid.all()
    np.testing.assert_allclose(pixels, expected, atol=1e-9)
    back, ok = cam.unproject(pixels)
    assert ok.all()
    np.testing.assert_allclose(back, rays, atol=1e-9)
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-15)
    undistorted = cv2.undistortPoints(pixels[:, None], _cv_camera(), coefficients, criteria=criteria)[:, 0]
    np.testing.assert_allclose(back[:, :2] / back[:, 2:], undistorted, atol=1e-8)


def test_kb4_matches_cv2_fisheye(cv2):
    coefficients = np.array(GOLDENS["kb4"]["coefficients"])
    cam = g.KB4(W, H, *K, k1=coefficients[0], k2=coefficients[1], k3=coefficients[2], k4=coefficients[3])
    rays = _rays(170)
    pixels, valid = cam.project(rays)
    expected = cv2.fisheye.projectPoints(rays[:, None], np.zeros(3), np.zeros(3), _cv_camera(), coefficients)[0][:, 0]
    assert valid.all()
    np.testing.assert_allclose(pixels, expected, atol=1e-9)
    back, _ = cam.unproject(pixels)
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-15)
    undistorted = cv2.fisheye.undistortPoints(pixels[:, None], _cv_camera(), coefficients, criteria=criteria)[:, 0]
    np.testing.assert_allclose(back[:, :2] / back[:, 2:], undistorted, atol=1e-8)


@pytest.mark.parametrize("model_id", sorted(GOLDENS))
def test_matches_rust_camera_models(model_id):
    """Golden values of an independent implementation (tools/geometry_goldens/ff_goldens.rs)."""
    golden = GOLDENS[model_id]
    cam = g.camera(model_id, W, H, *golden["intrinsics"], *golden["coefficients"])
    pixels, valid = cam.project(np.array(golden["rays"]))
    assert valid.all()
    np.testing.assert_allclose(pixels, golden["pixels"], atol=1e-8)
    rays, valid = cam.unproject(np.array(golden["grid"], dtype=np.float64))
    expected = np.array(golden["unprojected"])
    keep = np.ones(len(expected), dtype=bool)
    if isinstance(cam, g.PinholeOpenCV):  # the reference solves past the distortion fold, we mark those invalid
        keep = (expected[:, :2] ** 2).sum(-1) <= cam._fold2 * expected[:, 2] ** 2
        assert keep.mean() > 0.8
    np.testing.assert_array_equal(valid, keep)
    np.testing.assert_allclose(rays[keep], expected[keep], atol=1e-9)


def test_closed_forms_and_equivalences():
    """Each model against its paper formula or a special case of another model."""
    rays = _rays(170)
    x, y, z = rays.T
    r, theta = np.hypot(x, y), np.arctan2(np.hypot(x, y), z)
    fx, fy, cx, cy = K

    def pix(mx, my):
        return np.stack([fx * mx + cx, fy * my + cy], -1)

    equidistant = pix(theta * x / r, theta * y / r)
    np.testing.assert_allclose(g.camera("fisheye-equidistant", W, H, *K).project(rays)[0], equidistant, atol=1e-9)
    fov_w = 0.9
    rd = np.arctan(2 * np.tan(fov_w / 2) * r / z) / fov_w  # Devernay & Faugeras, in front of the camera
    front = z > 0
    np.testing.assert_allclose(
        g.FOV(W, H, *K, w=fov_w).project(rays[front])[0], pix(rd * x / r, rd * y / r)[front], atol=1e-9
    )
    xi, alpha = -0.2, 0.6  # Usenko et al., double sphere
    d1 = np.linalg.norm(rays, axis=-1)
    d2 = np.sqrt(x**2 + y**2 + (xi * d1 + z) ** 2)
    den = alpha * d2 + (1 - alpha) * (xi * d1 + z)
    np.testing.assert_allclose(g.DoubleSphere(W, H, *K, xi=xi, alpha=alpha).project(rays)[0], pix(x / den, y / den))
    a, b = 0.6, 1.1  # Khomutenko et al., EUCM
    den = a * np.sqrt(b * (x**2 + y**2) + z**2) + (1 - a) * z
    np.testing.assert_allclose(g.EUCM(W, H, *K, alpha=a, beta=b).project(rays)[0], pix(x / den, y / den))
    ucm = g.UCM(W, H, *K, xi=0.9).project(rays)[0]
    den = z + 0.9 * d1
    np.testing.assert_allclose(ucm, pix(x / den, y / den))
    kb = (-0.013, 0.021, -0.011, 0.0024)
    np.testing.assert_allclose(
        g.camera("fisheye624", W, H, *K, *kb, *[0.0] * 8).project(rays)[0],
        g.camera("kb4", W, H, *K, *kb).project(rays)[0],
    )


@pytest.mark.parametrize("model_id", sorted(WIDE))
def test_round_trip_over_the_whole_field_of_view(model_id):
    cam, fov = WIDE[model_id]
    rays = _rays(fov)
    pixels, valid = cam.project(rays)
    assert valid.all(), f"{model_id}: {(~valid).sum()} rays not projected"
    back, ok = cam.unproject(pixels)
    assert ok.all()
    np.testing.assert_allclose(back, rays, atol=1e-7)
    # and pixels -> rays -> pixels over the image (outside an image circle both directions must agree on validity)
    grid = np.stack(np.meshgrid(np.linspace(0, W - 1, 33), np.linspace(0, H - 1, 25)), -1)
    rays2, ok2 = cam.unproject(grid)
    again, ok3 = cam.project(np.where(ok2[..., None], rays2, [0, 0, 1]))
    np.testing.assert_allclose(again[ok2], grid[ok2], atol=1e-6)
    assert ok3[ok2].all()
    assert np.isnan(rays2[~ok2]).all()


def test_torch_in_torch_out_and_device_kind():
    cam = WIDE["kb4"][0]
    rays = torch.from_numpy(_rays(150)).float()
    pixels, valid = cam.project(rays)
    assert isinstance(pixels, torch.Tensor)
    assert pixels.dtype == torch.float64
    assert valid.dtype == torch.bool
    np.testing.assert_allclose(pixels.numpy(), cam.project(rays.double().numpy())[0])


def test_points_behind_a_pinhole_are_invalid():
    pixels, valid = WIDE["pinhole"][0].project(np.array([[0.1, 0.2, -1.0], [0.0, 0.0, 0.0], [0.1, 0.0, 1.0]]))
    assert valid.tolist() == [False, False, True]
    assert np.isnan(pixels[:2]).all()


def test_pinhole_opencv_rays_past_the_distortion_fold_are_invalid():
    """k1 = -0.35 folds back at r = 0.976 (d(r (1 + k1 r^2))/dr = 0): rays beyond it would alias inner pixels."""
    cam = g.PinholeOpenCV(W, H, *K, k1=-0.35)
    r = np.array([0.5, 0.97, 0.98, 1.2, 1.5])
    rays = np.stack([r, np.zeros_like(r), np.ones_like(r)], -1)
    pixels, valid = cam.project(rays)
    assert valid.tolist() == [True, True, False, False, False]
    assert np.isfinite(pixels[:2]).all()
    _, back_valid = cam.unproject(pixels[:2])
    assert back_valid.all()
    wide = g.PinholeOpenCV(W, H, 1100.0, 1100.0, 639.5, 479.5, k1=-0.35)  # corners (r_d 0.73 > 0.65) have no ray
    rect = g.rectify(g.StereoRig(wide, wide, np.eye(3), np.array([-0.1, 0.0, 0.0])), alpha=1.0)
    assert 0 < rect.f < 1100
    _, valid_maps = rect.maps()
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = ((xx - rect.P1[0, 2]) ** 2 + (yy - rect.P1[1, 2]) ** 2) / rect.f**2
    assert valid_maps[0].any()
    assert r2[valid_maps[0].numpy()].max() <= 0.9759**2  # no rectified pixel samples the folded-back image ring
    off = g.PinholeOpenCV(W, H, 410.0, 405.0, -5000.0, 479.5, k1=-0.35)  # every pixel is past the fold's image
    with pytest.raises(InputError, match="no valid pixels"):
        g.rectify(g.StereoRig(off, off, np.eye(3), np.array([-0.1, 0.0, 0.0])))


def test_numpy_scalars_are_accepted():
    cam = g.Pinhole(np.int64(W), np.int32(H), np.float32(500), np.float64(500), 320, np.float32(240))
    assert cam == g.Pinhole(W, H, 500.0, 500.0, 320.0, 240.0)
    assert type(cam.width) is int
    assert type(cam.fx) is float
    rig = g.RectifiedRig(np.float32(500), np.float32(0.1), np.int64(320), 240.0)
    assert type(rig.f) is float
    with pytest.raises(InputError, match="finite"):
        g.Pinhole(W, H, True, 500.0, 320.0, 240.0)  # noqa: FBT003


def test_camera_factory_aliases_and_types():
    assert isinstance(g.camera("fisheye-opencv", W, H, *K, 0.1, 0, 0, 0), g.KB4)
    assert isinstance(g.camera("omni", W, H, *K, 0.5), g.UCM)
    assert isinstance(g.camera("ds", W, H, *K, 0.1, 0.5), g.DoubleSphere)
    assert isinstance(g.camera("pinhole-ideal", W, H, *K), g.Pinhole)
    assert g.camera("kb3", W, H, *K, 0.1, 0.02, 0.003) == g.KB4(W, H, *K, k1=0.1, k2=0.02, k3=0.003, k4=0.0)
    assert g.camera("fisheye-kb3", W, H, *K, 0.1, 0, 0, 0.004) == g.KB4(W, H, *K, k1=0.1, k4=0.004)
    assert isinstance(g.camera("e57-spherical", W, H, *K), g.E57Spherical)
    for cam, _ in WIDE.values():
        assert isinstance(cam, g.Camera)
        assert cam.size == (H, W)
        assert (cam.width, cam.height) == (W, H)
    cam = WIDE["kb4"][0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cam.fx = 1.0


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: g.camera("sphere", W, H, *K), "unknown camera model 'sphere'"),
        (lambda: g.camera("kb4", W, H, *K, 0.1), "kb4"),
        (lambda: g.camera("pinhole-opencv", W, H, *K, 0.1, 0.2, 0.3), "pinhole-opencv"),
        (lambda: g.camera("kb3", W, H, *K, 0.1, 0.2), "kb3"),
        (lambda: g.camera("e57-pinhole", W, H, *K, 0.1), "e57-pinhole"),
        (lambda: g.Pinhole(W, 1, *K), "height must be an int >= 2"),
        (lambda: g.Pinhole(640.0, H, *K), "width must be an int >= 2"),  # wrong on purpose
        (lambda: g.Pinhole(W, H, -1.0, 1.0, 0.0, 0.0), "focal lengths must be positive"),
        (lambda: g.KB4(W, H, *K, k1=math.nan), "k1"),
        (lambda: WIDE["kb4"][0].project(np.zeros((4, 2))), r"points must have shape \[..., 3\]"),
        (
            lambda: WIDE["kb4"][0].unproject([[1.0, 2.0]]),
            "pixels must be a numpy array or torch tensor",
        ),  # wrong on purpose
    ],
)
def test_input_errors(build, match):
    with pytest.raises(InputError, match=match):
        build()


@pytest.mark.parametrize(
    ("model_id", "axis_sign", "lo", "hi"),
    [("e57-pinhole", -1, 0.0, 1.2), ("e57-spherical", 1, 0.05, 3.09), ("e57-cylindrical", 1, 0.3, 2.84)],
)
def test_e57_round_trip(model_id, axis_sign, lo, hi):
    """E57 images: project -> unproject returns the ray; points the image cannot see are invalid."""
    cam = g.camera(model_id, W, H, *K)
    rng = np.random.default_rng(1)
    theta = rng.uniform(lo, hi, 300)
    phi = rng.uniform(-np.pi + 1e-3, np.pi - 1e-3, 300)
    rays = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), axis_sign * np.cos(theta)], -1)
    pixels, valid = cam.project(rays * 3.0)
    assert valid.all()
    back, valid = cam.unproject(pixels)
    assert valid.all()
    np.testing.assert_allclose(back, rays, atol=1e-12)
    _, valid = cam.project(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]))
    assert valid.tolist() == ([False, True] if model_id == "e57-pinhole" else [False, False])


def test_rectify_rejects_e57():
    cam = g.camera("e57-pinhole", W, H, *K)
    rig = g.StereoRig(cam, cam, np.eye(3), np.array([-0.1, 0.0, 0.0]))
    with pytest.raises(InputError, match="e57-pinhole"):
        g.rectify(rig)
