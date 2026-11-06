"""LAS / LAZ point cloud files (laspy + lazrs): roundtrip, precision, colours, holes, and the install hint."""

import builtins

import numpy as np
import pytest

import fastforks
from fastforks import BackendUnavailableError, InputError
from fastforks.stereo import geometry as g


def _cloud(n=500, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.uniform([-3, -2, 0.5], [3, 2, 60], (n, 3)).astype(np.float32)
    pts[::7] = np.nan  # holes
    rgb = rng.integers(0, 256, (n, 3), dtype=np.uint8)
    return pts, rgb


@pytest.mark.parametrize("suffix", [".las", ".laz"])
def test_las_roundtrip_keeps_sub_millimetre_precision_and_colours(tmp_path, suffix):
    laspy = pytest.importorskip("laspy")
    if suffix == ".laz":
        pytest.importorskip("lazrs")
    pts, rgb = _cloud()
    n = g.write_las(tmp_path / f"c{suffix}", pts, rgb)
    ok = np.isfinite(pts).all(1)
    assert n == ok.sum()
    las = laspy.read(tmp_path / f"c{suffix}")
    got = np.stack([las.x, las.y, las.z], 1)
    assert np.abs(got - pts[ok]).max() < 1e-4  # scale 0.1 mm: rounding error at most 0.05 mm
    np.testing.assert_array_equal(np.stack([las.red, las.green, las.blue], 1) // 257, rgb[ok])


def test_las_without_colours_and_via_fastforks_save(tmp_path):
    laspy = pytest.importorskip("laspy")
    pytest.importorskip("lazrs")
    pts, _ = _cloud()
    fastforks.save(tmp_path / "plain.laz", pts)
    las = laspy.read(tmp_path / "plain.laz")
    assert len(las.x) == np.isfinite(pts).all(1).sum()
    assert "red" not in las.point_format.dimension_names


def test_las_large_extent_falls_back_to_a_coarser_scale(tmp_path):
    laspy = pytest.importorskip("laspy")
    pts = np.array([[0, 0, 0], [5e5, 1.0, 2.0]], np.float32)
    g.write_las(tmp_path / "far.las", pts)
    las = laspy.read(tmp_path / "far.las")
    assert las.header.scales[0] <= 5e5 / 2e9 * 1.0001 or las.header.scales[0] == 1e-4
    assert abs(las.x[1] - 5e5) < 1e-3


def test_laz_without_the_libraries_tells_how_to_install(tmp_path, monkeypatch):
    real = builtins.__import__

    def block(name, *args, **kwargs):
        if name in ("laspy", "lazrs"):
            msg = f"No module named {name!r}"
            raise ImportError(msg, name=name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block)
    with pytest.raises(BackendUnavailableError, match=r"pip install (laspy )?lazrs"):
        g.write_las(tmp_path / "x.laz", np.zeros((2, 3), np.float32))
    assert not (tmp_path / "x.laz").exists()


def test_las_shape_errors(tmp_path):
    pytest.importorskip("laspy")
    with pytest.raises(InputError, match=r"\[\.\.\., 3\]"):
        g.write_las(tmp_path / "x.las", np.zeros((4, 2), np.float32))
