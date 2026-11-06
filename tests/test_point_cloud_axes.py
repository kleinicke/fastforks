"""Point-cloud axes presets, header comments, colours of depth / multi-view clouds, metric depth as a cloud."""

import dataclasses

import numpy as np
import pytest
import torch

import fastforks
from fastforks import InputError, OptionsError
from fastforks.depth import DepthResult
from fastforks.multiview import MultiViewResult
from fastforks.stereo import geometry as g

PTS = np.array([[1, 2, 3], [-4, 5, 6]], np.float32)


def _read(path):
    head, body = path.read_bytes().split(b"end_header\n", 1)
    return head.decode(), np.frombuffer(body, "<f4").reshape(-1, 3)


def test_default_axes_are_the_input_and_the_header_names_everything(tmp_path):
    g.write_ply(tmp_path / "a.ply", PTS)
    head, xyz = _read(tmp_path / "a.ply")
    np.testing.assert_array_equal(xyz, PTS)
    assert f"comment fastforks {fastforks.__version__}" in head
    assert "comment units: metres" in head
    assert "comment axes: RDF" in head
    assert "pixel centres" in head
    g.write_ply(tmp_path / "b.ply", PTS, frame="world")
    assert "comment frame: world" in _read(tmp_path / "b.ply")[0]


@pytest.mark.parametrize(
    ("axes", "expected"),
    [
        ("opengl", [[1, -2, -3], [-4, -5, -6]]),
        ("blender", [[1, -2, -3], [-4, -5, -6]]),
        ("ros", [[3, -1, -2], [6, 4, -5]]),
        ("pytorch3d", [[-1, -2, 3], [4, -5, 6]]),
        ("RFU", [[1, 3, -2], [-4, 6, -5]]),
        ("rdf", [[1, 2, 3], [-4, 5, 6]]),
    ],
)
def test_axes_presets_and_codes(tmp_path, axes, expected):
    g.write_ply(tmp_path / "a.ply", PTS, axes=axes)
    np.testing.assert_array_equal(_read(tmp_path / "a.ply")[1], np.array(expected, np.float32))
    fastforks.save(tmp_path / "b.ply", PTS, axes=axes)
    assert _read(tmp_path / "b.ply")[1].tolist() == np.array(expected, np.float32).tolist()


@pytest.mark.parametrize("axes", ["rdd", "RD", "xyz", "RDFU", "", None])
def test_wrong_axes_raise_and_list_the_presets(tmp_path, axes):
    with pytest.raises(OptionsError, match=r"opengl = RUB.*ros = FLU.*pytorch3d = LUF"):
        g.write_ply(tmp_path / "a.ply", PTS, axes=axes)
    with pytest.raises(OptionsError, match="axes"):
        fastforks.save(tmp_path / "a.laz", PTS, axes=axes)
    assert not (tmp_path / "a.ply").exists()


@pytest.mark.parametrize("suffix", [".las", ".laz"])
def test_las_axes_and_vlr(tmp_path, suffix):
    laspy = pytest.importorskip("laspy")
    if suffix == ".laz":
        pytest.importorskip("lazrs")
    g.write_las(tmp_path / f"a{suffix}", PTS, axes="opengl", frame="camera")
    las = laspy.read(tmp_path / f"a{suffix}")
    np.testing.assert_allclose(np.stack([las.x, las.y, las.z], 1), [[1, -2, -3], [-4, -5, -6]], atol=1e-4)
    assert las.header.system_identifier == "fastforks"
    text = next(v for v in las.vlrs if v.user_id == "fastforks").record_data.decode()
    assert "units: metres" in text
    assert "axes: RUB" in text
    assert "frame: camera" in text


def _depth(**kw):
    fields = dict.fromkeys(("depth", "inverse_depth", "points", "intrinsics", "confidence", "mask", "normals"))
    return DepthResult(**(fields | {"extras": {}, "kind": "metric", "model": "m", "backend": "portable"} | kw))


def test_metric_depth_with_intrinsics_saves_as_a_cloud(tmp_path):
    depth = torch.full((4, 6), 2.0)
    k = torch.tensor([[10.0, 0, 2.0], [0, 10.0, 1.0], [0, 0, 1]])
    _depth(depth=depth, intrinsics=k).save(tmp_path / "d.ply")
    xyz = _read(tmp_path / "d.ply")[1].reshape(4, 6, 3)
    np.testing.assert_allclose(xyz[1, 2], [0, 0, 2.0], atol=1e-6)  # principal point
    np.testing.assert_allclose(xyz[1, 3], [0.2, 0, 2.0], atol=1e-6)
    with pytest.raises(InputError, match="intrinsics=K"):
        _depth(depth=depth).save(tmp_path / "e.ply")
    with pytest.raises(InputError, match="intrinsics=K"):
        _depth(depth=depth, intrinsics=k, kind="scale_invariant").save(tmp_path / "e.ply")


def test_depth_and_multiview_clouds_are_coloured_by_the_input_image(tmp_path):
    points = torch.rand(4, 5, 3)
    image = torch.rand(3, 4, 5)
    r = _depth(points=points)
    object.__setattr__(r, "_image", image)
    r.save(tmp_path / "c.ply")
    head = _read(tmp_path / "c.ply")[0]
    assert "property uchar red" in head
    assert "property uchar red" in _read_head_after_to(r, tmp_path)  # `.to` keeps the colours
    assert "property uchar red" not in _read(_save(_depth(points=points), tmp_path / "n.ply"))[0]


def _save(r, path):
    r.save(path)
    return path


def _read_head_after_to(r, tmp_path):
    r.to("cpu").save(tmp_path / "t.ply")
    return _read(tmp_path / "t.ply")[0]


def test_multiview_merged_cloud_is_coloured_and_in_one_frame(tmp_path):
    v = 2
    pts = torch.rand(v, 3, 4, 3)
    mask = torch.ones(v, 3, 4, dtype=torch.bool)
    mask[1, 0, 0] = False
    fields = dict.fromkeys(("confidence", "intrinsics", "cam_to_world"))
    r = MultiViewResult(
        depth=pts[..., 2], world_points=pts, mask=mask, **fields, extras={}, views=(0, 1), kind="metric",
        model="m", backend="portable",
    )  # fmt: skip
    object.__setattr__(r, "_image", torch.rand(v, 3, 3, 4))
    r.save(tmp_path / "all.ply")
    head, xyz = _read_xyz_colour(tmp_path / "all.ply")
    assert len(xyz) == 2 * 12 - 1
    assert "frame: world" in head
    r.save(tmp_path / "one.ply", view=1)
    assert len(_read_xyz_colour(tmp_path / "one.ply")[1]) == 11
    r.save(tmp_path / "gl.ply", axes="opengl")


def _read_xyz_colour(path):
    head, body = path.read_bytes().split(b"end_header\n", 1)
    n = int(next(ln for ln in head.decode().splitlines() if ln.startswith("element vertex")).split()[-1])
    assert len(body) == n * 15
    return head.decode(), np.frombuffer(body, np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)]))


def test_multiview_cloud_without_world_points_unprojects_depth(tmp_path):  # LEON-R1-01: calibrated MVS
    depth = torch.full((2, 3, 4), 2.0)
    k = torch.tensor([[2.0, 0, 1.5], [0, 2.0, 1.0], [0, 0, 1]]).expand(2, 3, 3)
    c2w = torch.eye(4).repeat(2, 1, 1)
    c2w[1, 0, 3] = 10.0
    fields = dict.fromkeys(("confidence", "world_points", "mask"))
    r = MultiViewResult(
        depth=depth, intrinsics=k, cam_to_world=c2w, **fields, extras={}, views=(0, 1), kind="metric",
        model="m", backend="portable",
    )  # fmt: skip
    r.save(tmp_path / "mvs.ply")
    xyz = np.frombuffer(tmp_path.joinpath("mvs.ply").read_bytes().split(b"end_header\n", 1)[1], "<f4").reshape(-1, 3)
    assert len(xyz) == 24
    assert np.allclose(xyz[:, 2], 2.0)
    assert np.isclose(xyz[12:, 0].min() - xyz[:12, 0].min(), 10.0)
    assert "units: metres" not in tmp_path.joinpath("mvs.ply").read_bytes().decode("latin1")  # the user's units
    with pytest.raises(InputError, match="no world_points and no depth"):
        dataclasses.replace(r, cam_to_world=None).save(tmp_path / "x.ply")
    with pytest.raises(OptionsError, match="max_depth"):  # VERIFY-MVS-NAN: as `fastforks.save`
        r.save(tmp_path / "nan.ply", max_depth=float("nan"))
    assert not (tmp_path / "nan.ply").exists()


def test_las_keeps_float64_precision(tmp_path):
    laspy = pytest.importorskip("laspy")
    pts = np.array([[500000.001, 5400000.001, 100.001], [500000.009, 5400000.009, 100.009]], np.float64)
    g.write_las(tmp_path / "utm.las", pts)
    las = laspy.read(tmp_path / "utm.las")
    np.testing.assert_allclose(np.stack([las.x, las.y, las.z], 1), pts, atol=6e-5)
    assert abs((las.x[1] - las.x[0]) - 0.008) < 2e-4


def test_nan_max_depth_raises(tmp_path):
    with pytest.raises(OptionsError, match="max_depth"):
        fastforks.save(tmp_path / "b.ply", PTS, max_depth=float("nan"))
    assert not (tmp_path / "b.ply").exists()
    fastforks.save(tmp_path / "c.ply", PTS, max_depth=None)
