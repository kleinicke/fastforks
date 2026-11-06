"""`MultiViewResult.save_scene`: COLMAP text model and nerfstudio `transforms.json`, read back with `read_calib`.

The independent consumers (pycolmap, nerfstudio's dataparser, gsplat's COLMAP parser) are checked outside the test
suite (no extra dependencies); see the 017 results file `2026-11-06_scene_writers.md`.
"""

import dataclasses
import json
from itertools import permutations

import numpy as np
import pytest
import torch
from PIL import Image

from fastforks import InputError, OptionsError
from fastforks.multiview import MultiViewResult, unproject
from fastforks.multiview._scene import _quaternion
from fastforks.stereo import geometry as g
from fastforks.stereo.geometry.calib_formats import _colmap_cameras, _colmap_images, _quat_to_rot
from fastforks.stereo.geometry.rectify import _rotation_from_vector

H, W = 6, 9  # non-square
OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])


def _result(n=4, views=None, *, world_points=True, image=True):
    """`n` views with different intrinsics and non-trivial rotations; `views` keeps a subset (calibrated MVS)."""
    rng = np.random.default_rng(1)
    k = np.zeros((n, 3, 3))
    k[:, 0, 0], k[:, 1, 1] = 7 + rng.random(n), 7.5 + rng.random(n)
    k[:, 0, 2], k[:, 1, 2], k[:, 2, 2] = (W - 1) / 2 + rng.random(n), (H - 1) / 2 - rng.random(n), 1
    k[0] = k[1]  # two views share a camera
    c2w = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        c2w[i, :3, :3] = _rotation_from_vector(rng.normal(size=3))
        c2w[i, :3, 3] = rng.normal(size=3) * 2
    views = tuple(range(n)) if views is None else views
    k, c2w = torch.tensor(k[list(views)]), torch.tensor(c2w[list(views)])
    depth = torch.tensor(2 + 3 * rng.random((len(views), H, W)), dtype=torch.float32)
    mask = torch.tensor(rng.random((len(views), H, W)) > 0.2)
    pts = unproject(depth, k, c2w).float() if world_points else None
    r = MultiViewResult(
        depth=depth, world_points=pts, confidence=None, mask=mask, intrinsics=k.float(), cam_to_world=c2w, extras={},
        views=views, kind="metric", model="m", backend="portable",
    )  # fmt: skip
    if image:
        object.__setattr__(r, "_image", torch.tensor(rng.random((len(views), 3, H, W)), dtype=torch.float32))
    return r


def _expected_rig(r, a, b):
    c2w = r.cam_to_world.numpy()
    x = np.linalg.inv(c2w[b]) @ c2w[a]  # right-from-left
    return x[:3, :3], x[:3, 3]


def _check_rig(rig, r, a, b):
    k = r.intrinsics.numpy()
    for cam, i in ((rig.left, a), (rig.right, b)):
        assert (cam.width, cam.height) == (W, H)
        np.testing.assert_allclose([cam.fx, cam.fy, cam.cx, cam.cy], k[i][[0, 1, 0, 1], [0, 1, 2, 2]], atol=1e-5)
    rot, t = _expected_rig(r, a, b)
    np.testing.assert_allclose(rig.R, rot, atol=1e-12)
    np.testing.assert_allclose(rig.T, t, atol=1e-12)


def _ply(path):
    head, body = path.read_bytes().split(b"end_header\n", 1)
    return head, np.frombuffer(body, np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)]))


@pytest.mark.parametrize("fmt", ["colmap", "nerfstudio"])
@pytest.mark.parametrize("views", [None, (1, 3, 4)])
def test_round_trip_through_read_calib(tmp_path, fmt, views):
    r = _result(5, views)
    r.save_scene(tmp_path, fmt)
    names = [f"frame_{i + 1:05d}.png" for i in r.views]
    path = tmp_path / ("sparse/0" if fmt == "colmap" else "transforms.json")
    for a, b in permutations(range(len(r.views)), 2):
        name = (names[a], names[b]) if fmt == "colmap" else (f"images/{names[a]}", f"images/{names[b]}")
        _check_rig(g.read_calib(path, fmt, names=name), r, a, b)
    assert sorted(p.name for p in (tmp_path / "images").iterdir()) == names


def test_colmap_poses_points_and_pixels(tmp_path):
    """Every written point projects (COLMAP K with its +0.5 centres, w2c pose) onto the pixel it came from."""
    r = _result(3)
    r.save_scene(tmp_path, "colmap", max_points=None)
    sparse = tmp_path / "sparse/0"
    cams, images = _colmap_cameras(sparse, ".txt"), _colmap_images(sparse, ".txt")
    assert len(cams) == 2  # views 0 and 1 share intrinsics
    assert "PINHOLE" in (sparse / "cameras.txt").read_text()
    lines = [ln.split() for ln in (sparse / "points3D.txt").read_text().splitlines() if not ln.startswith("#")]
    xyz = np.array([ln[1:4] for ln in lines], float)
    rgb = np.array([ln[4:7] for ln in lines], np.uint8)
    assert all(len(ln) == 8 for ln in lines)  # empty track
    valid = r.mask.numpy()
    assert len(xyz) == valid.sum()
    offset = 0
    for i in range(3):
        w2c, cid = images[f"frame_{i + 1:05d}.png"]
        np.testing.assert_allclose(w2c, np.linalg.inv(r.cam_to_world[i].numpy()), atol=1e-12)
        cam = cams[cid]
        n = valid[i].sum()
        p = xyz[offset : offset + n] @ w2c[:3, :3].T + w2c[:3, 3]
        u, v = cam.fx * p[:, 0] / p[:, 2] + cam.cx, cam.fy * p[:, 1] / p[:, 2] + cam.cy
        ys, xs = np.nonzero(valid[i])
        np.testing.assert_allclose(u, xs, atol=1e-4)
        np.testing.assert_allclose(v, ys, atol=1e-4)
        colours = np.rint(r.__dict__["_image"][i].movedim(0, -1).numpy()[ys, xs] * 255)
        np.testing.assert_array_equal(rgb[offset : offset + n], colours)
        offset += n


def test_nerfstudio_file_ply_and_images(tmp_path):
    r = _result(3)
    r.save_scene(tmp_path, "nerfstudio", max_points=20)
    scene = json.loads((tmp_path / "transforms.json").read_text())
    assert scene["camera_model"] == "PINHOLE"
    assert scene["ply_file_path"] == "sparse_pc.ply"
    k = r.intrinsics.numpy()
    for i, f in enumerate(scene["frames"]):
        assert f["file_path"] == f"images/frame_{i + 1:05d}.png"
        np.testing.assert_allclose(np.array(f["transform_matrix"]) @ OPENGL, r.cam_to_world[i].numpy(), atol=1e-15)
        np.testing.assert_allclose([f["cx"], f["cy"]], k[i, :2, 2] + 0.5, atol=1e-6)
        assert (f["w"], f["h"]) == (W, H)
        png = np.asarray(Image.open(tmp_path / f["file_path"]))  # lossless: the input's 8-bit values
        np.testing.assert_array_equal(png, np.rint(r.__dict__["_image"][i].movedim(0, -1).numpy() * 255))
    head, ply = _ply(tmp_path / "sparse_pc.ply")
    assert len(ply) == 20
    assert b"frame: world" in head
    all_pts = r.world_points.numpy()[r.mask.numpy()]
    assert all(np.any(np.all(all_pts == p, axis=1)) for p in ply["xyz"])  # a subset of the valid points


def test_both_formats_share_the_points_and_a_fixed_seed(tmp_path):
    r = _result(3)
    for d in ("a", "b"):
        r.save_scene(tmp_path / d, "colmap", max_points=7)
        r.save_scene(tmp_path / d, "nerfstudio", max_points=7)
    text = (tmp_path / "a/sparse/0/points3D.txt").read_text()
    assert text == (tmp_path / "b/sparse/0/points3D.txt").read_text()
    ply = (tmp_path / "a/sparse_pc.ply").read_bytes()
    assert ply == (tmp_path / "b/sparse_pc.ply").read_bytes()
    xyz = np.array([ln.split()[1:4] for ln in text.splitlines() if not ln.startswith("#")], np.float32)
    np.testing.assert_array_equal(xyz, _ply(tmp_path / "a/sparse_pc.ply")[1]["xyz"])


def test_image_names_are_referenced_not_written(tmp_path):
    r = _result(4, (0, 2))  # calibrated MVS: two reference views of four inputs
    r.save_scene(tmp_path, "colmap", image_names=["images/IMG_0001.JPG", "images/sub/IMG_0003.JPG"])
    r.save_scene(tmp_path, "nerfstudio", image_names=["images/IMG_0001.JPG", "images/sub/IMG_0003.JPG"])
    assert not (tmp_path / "images").exists()
    _check_rig(g.read_calib(tmp_path / "sparse/0", "colmap", names=("IMG_0001.JPG", "sub/IMG_0003.JPG")), r, 0, 1)
    rig = g.read_calib(tmp_path / "transforms.json", "nerfstudio", names=("images/IMG_0001.JPG", "IMG_0003.JPG"))
    _check_rig(rig, r, 0, 1)
    # without the input images (and with image_names): grey points, unprojected depth for a result without points
    dataclasses.replace(r, world_points=None).save_scene(tmp_path / "c", "colmap", image_names=["a.png", "b.png"])
    line = (tmp_path / "c/sparse/0/points3D.txt").read_text().splitlines()[3].split()
    assert line[4:7] == ["128", "128", "128"]
    assert "../a.png" in (tmp_path / "c/sparse/0/images.txt").read_text()  # COLMAP names: relative to images/


def test_float64_poses_keep_precision(tmp_path):
    r = _result(2, world_points=False)
    c2w = r.cam_to_world.clone()
    c2w[:, :3, 3] += torch.tensor([500000.0, 5400000.0, 100.0], dtype=torch.float64)  # UTM-sized
    moved = dataclasses.replace(r, cam_to_world=c2w)
    object.__setattr__(moved, "_image", r.__dict__["_image"])
    moved.save_scene(tmp_path, "colmap")
    w2c, _ = _colmap_images(tmp_path / "sparse/0", ".txt")["frame_00002.png"]
    np.testing.assert_allclose(np.linalg.inv(w2c)[:3, 3], c2w[1, :3, 3].numpy(), rtol=0, atol=1e-8)


def test_quaternion_matches_the_reader():
    rng = np.random.default_rng(0)
    rots = [_rotation_from_vector(rng.normal(size=3) * s) for s in (0.01, 1, 3)]
    rots += [np.diag([1.0, -1, -1]), np.diag([-1.0, 1, -1]), np.diag([-1.0, -1, 1]), np.eye(3)]  # 180 degrees
    for rot in rots:
        q = _quaternion(rot)
        assert q[0] >= 0
        np.testing.assert_allclose(_quat_to_rot(tuple(q)), rot, atol=1e-14)


def test_errors(tmp_path):
    r = _result(3)
    with pytest.raises(InputError, match="intrinsics / cam_to_world"):
        dataclasses.replace(r, cam_to_world=None).save_scene(tmp_path, "colmap")
    with pytest.raises(InputError, match="intrinsics / cam_to_world"):
        dataclasses.replace(r, intrinsics=None).save_scene(tmp_path, "nerfstudio")
    with pytest.raises(InputError, match="3 paths relative"):
        r.save_scene(tmp_path, "colmap", image_names=["a.png", "b.png"])
    with pytest.raises(InputError, match="3 paths relative"):
        r.save_scene(tmp_path, "colmap", image_names=["a.png", "b.png", "/abs/c.png"])
    with pytest.raises(InputError, match="no input images"):
        _result(3, image=False).save_scene(tmp_path, "colmap")
    for bad in (0, -1, True, 2.5):
        with pytest.raises(InputError, match="max_points"):
            r.save_scene(tmp_path, "colmap", max_points=bad)
    with pytest.raises(OptionsError, match="format"):
        r.save_scene(tmp_path, "ply")
    assert not any(tmp_path.iterdir())  # nothing written before the checks pass
