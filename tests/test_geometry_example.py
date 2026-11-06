"""The end-to-end example of docs/stereo_geometry.md, run on rendered images with a stand-in stereo model."""

import re
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from fastforks import stereo
from fastforks.stereo import geometry as g

DOC = Path(__file__).resolve().parents[1] / "docs" / "stereo_geometry.md"
W, H, Z0 = 320, 240, 4.0  # image size; the scene is the plane z = Z0 (metres) in the left camera frame
DIST = ((-0.28, 0.09, 0.001, -0.0005, -0.01), (-0.26, 0.08, -0.0007, 0.0004, -0.012))
ANGLE = np.radians(1.2)
R = np.array([[np.cos(ANGLE), 0, np.sin(ANGLE)], [0, 1, 0], [-np.sin(ANGLE), 0, np.cos(ANGLE)]])
T = np.array([-0.3, 0.004, 0.002])


def _example() -> str:
    section = DOC.read_text().split("## End-to-end example", 1)[1]
    return re.search(r"```python\n(.*?)```", section, re.DOTALL).group(1)


def _texture(xy: np.ndarray) -> np.ndarray:
    """Smooth RGB pattern on the plane, uint8."""
    x, y = xy[..., 0], xy[..., 1]
    channels = [np.sin(7 * x + 3 * y), np.cos(5 * y - 2 * x), np.sin(4 * x) * np.cos(6 * y)]
    return np.clip(np.stack(channels, -1) * 100 + 128, 0, 255).round().astype(np.uint8)


def _render(cam: g.Camera, origin: np.ndarray, to_left: np.ndarray) -> np.ndarray:
    """Image of the textured plane seen by `cam`, whose rays are `to_left @ ray` from `origin` in the left frame."""
    u, v = np.meshgrid(np.arange(cam.width, dtype=np.float64), np.arange(cam.height, dtype=np.float64))
    rays, valid = cam.unproject(np.stack([u, v], -1))
    assert valid.all()
    rays = rays @ to_left.T
    hit = origin + rays * ((Z0 - origin[2]) / rays[..., 2:])
    return _texture(hit[..., :2])


def _write_calib(path: Path, cams: list[g.Camera]) -> None:
    """KITTI raw `calib_cam_to_cam.txt` with cameras 2 (left, the reference) and 3 (right)."""
    fmt = lambda a: " ".join(f"{float(v):.12e}" for v in np.ravel(a))  # noqa: E731
    lines = []
    for i, cam, rot, t in ((2, cams[0], np.eye(3), np.zeros(3)), (3, cams[1], R, T)):
        k = [[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]]
        lines += [f"S_0{i}: {fmt((cam.width, cam.height))}", f"K_0{i}: {fmt(k)}", f"D_0{i}: {fmt(DIST[i - 2])}"]
        lines += [f"R_0{i}: {fmt(rot)}", f"T_0{i}: {fmt(t)}"]
    path.write_text("\n".join(lines) + "\n")


def test_docs_end_to_end_example(tmp_path, monkeypatch):
    cams = [
        g.camera("pinhole-opencv", W, H, 250.0, 250.0, 160.0 + dx, 120.0, *d)
        for d, dx in zip(DIST, (2, -3), strict=True)
    ]
    _write_calib(tmp_path / "calib_cam_to_cam.txt", cams)
    Image.fromarray(_render(cams[0], np.zeros(3), np.eye(3))).save(tmp_path / "left.png")
    Image.fromarray(_render(cams[1], -R.T @ T, R.T)).save(tmp_path / "right.png")  # X_left = R^T (X_right - T)
    calls = []

    class Model:
        """Stand-in for `stereo.load(...)`: the exact disparity of the plane in the rectified frame."""

        device = torch.device("cpu")

        def __init__(self, model_id: str, **_: object) -> None:
            assert model_id == "igev-stereo-sceneflow"
            self.rect = g.rectify(g.read_calib(tmp_path / "calib_cam_to_cam.txt", "kitti-unrectified"))

        def __call__(self, left: torch.Tensor, right: torch.Tensor) -> types.SimpleNamespace:
            calls.append((left, right))
            rect = self.rect
            u, v = np.meshgrid(np.arange(rect.size[1]), np.arange(rect.size[0]))
            q = np.stack([(u - rect.P1[0, 2]) / rect.f, (v - rect.P1[1, 2]) / rect.f, np.ones(u.shape)], -1)
            z = Z0 / (q @ rect.R1[:, 2])  # X_rect = z q and X_left = R1^T X_rect lies on z_left = Z0
            return types.SimpleNamespace(disparity=torch.from_numpy(rect.f * rect.baseline / z).float())

    monkeypatch.setattr(stereo, "load", Model)
    monkeypatch.chdir(tmp_path)
    scope: dict = {}
    exec(compile(_example(), "stereo_geometry.md example", "exec"), scope)  # noqa: S102 - the documented example

    rect, left_r, points, colors, depth_left = (scope[k] for k in ("rect", "left_r", "points", "colors", "depth_left"))
    assert len(calls) == 1
    assert calls[0][0].shape == (3, H, W)
    assert calls[0][0].dtype == torch.uint8
    # The rectified left image equals the plane rendered directly by the rectified camera.
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    q = np.stack([(u - rect.P1[0, 2]) / rect.f, (v - rect.P1[1, 2]) / rect.f, np.ones(u.shape)], -1) @ rect.R1
    expected = _texture(q[..., :2] * (Z0 / q[..., 2:]))
    x, y, w, h = rect.roi_left
    inner = (slice(y + 2, y + h - 2), slice(x + 2, x + w - 2))
    diff = np.abs(left_r.permute(1, 2, 0).numpy()[inner].astype(int) - expected[inner].astype(int))
    assert diff.mean() < 1.5
    # The cloud lies on the plane in the original left frame, coloured by the rectified image.
    assert len(points) > 0.9 * W * H
    np.testing.assert_allclose(points[:, 2], Z0, rtol=1e-4)
    assert colors.shape == (len(points), 3)
    assert colors.dtype == np.uint8
    # Depth at the raw left pixels is the plane's depth along the left optical axis.
    known = depth_left.numpy() > 0
    assert known.mean() > 0.8  # the rectified image misses some raw image corners
    np.testing.assert_allclose(depth_left.numpy()[known], Z0, rtol=2e-3)


@pytest.mark.parametrize("name", ["calib_cam_to_cam.txt", "left.png", "right.png"])
def test_example_names_its_inputs(name):
    assert f'"{name}"' in _example()
