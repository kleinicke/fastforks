"""`result.save(...)` and the float32 TIFF writer (read back with Pillow's libtiff, an independent reader)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from fastforks import InputError, OptionsError
from fastforks._core import io as ffio
from fastforks.depth import DepthResult
from fastforks.multiview import MultiViewResult
from fastforks.stereo import StereoResult

RNG = np.random.default_rng(0)


def read(path):
    return np.asarray(Image.open(path))


def bits_of(a):
    return np.ascontiguousarray(a).view(np.uint32)


@pytest.fixture(params=["isal", "zlib"])
def compressor(request, monkeypatch):
    if request.param == "zlib":
        import zlib  # noqa: PLC0415

        monkeypatch.setattr(ffio, "_deflate", zlib.compress)
    return request.param


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (70, 1), (64, 5), (65, 33), (200, 300), (5, 1000)])
def test_tiff_round_trip_exact(tmp_path, compressor, shape):
    a = (RNG.standard_normal(shape) * 50).astype(np.float32)
    ffio.write_tiff(tmp_path / "a.tif", a)
    b = read(tmp_path / "a.tif")
    assert b.dtype == np.float32
    np.testing.assert_array_equal(bits_of(b), bits_of(a))


def test_tiff_compresses_smooth_maps(tmp_path):
    a = np.linspace(0, 100, 256 * 256, dtype=np.float32).reshape(256, 256)
    ffio.write_tiff(tmp_path / "a.tif", a)
    assert (tmp_path / "a.tif").stat().st_size < a.nbytes / 1.5


def test_tiff_non_finite_preserved(tmp_path):
    a = np.array([[np.nan, np.inf, -np.inf, 0.0, -0.0, 1.5]], np.float32)
    for bits in (None, 8):
        ffio.write_tiff(tmp_path / "a.tif", a, bits)
        b = read(tmp_path / "a.tif")
        assert np.isnan(b[0, 0])
        assert b[0, 1] == np.inf
        assert b[0, 2] == -np.inf
        assert b[0, 3] == 0
        assert bits_of(b)[0, 4] == bits_of(a)[0, 4]  # -0.0 keeps its sign


@pytest.mark.parametrize("bits", [1, 5, 10, 16, 22, 23])
def test_tiff_bits_error_bound(tmp_path, bits):
    a = (RNG.standard_normal((100, 90)) * 10 ** RNG.uniform(-3, 4, (100, 90))).astype(np.float32)
    ffio.write_tiff(tmp_path / "a.tif", a, bits)
    b = read(tmp_path / "a.tif")
    rel = np.abs(b.astype(np.float64) - a) / np.abs(a)
    assert rel.max() <= 2.0 ** -(bits + 1) * (1 + 1e-6)
    if bits == 23:
        np.testing.assert_array_equal(b, a)


def test_tiff_bits_makes_files_smaller(tmp_path):
    a = (RNG.standard_normal((128, 128)).cumsum(1) + 100).astype(np.float32)
    ffio.write_tiff(tmp_path / "full.tif", a)
    ffio.write_tiff(tmp_path / "b10.tif", a, 10)
    assert (tmp_path / "b10.tif").stat().st_size < (tmp_path / "full.tif").stat().st_size / 2


@pytest.mark.parametrize("bits", [0, 24, -1])
def test_bits_invalid(tmp_path, bits):
    with pytest.raises(InputError, match="bits"):
        ffio.write_tiff(tmp_path / "a.tif", np.ones((4, 4), np.float32), bits)


def test_write_map_formats_and_errors(tmp_path):
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    ffio.write_map(tmp_path / "a.npy", torch.from_numpy(a))
    np.testing.assert_array_equal(np.load(tmp_path / "a.npy"), a)
    ffio.write_map(tmp_path / "a.TIFF", a)
    np.testing.assert_array_equal(read(tmp_path / "a.TIFF"), a)
    with pytest.raises(InputError, match="unknown map format"):
        ffio.write_map(tmp_path / "a.jpg", a)
    with pytest.raises(InputError, match=r"must be \[H, W\]"):
        ffio.write_map(tmp_path / "a.tif", a[None])


def stereo(*, batch=False):
    d = torch.rand(2, 5, 6) * 100 if batch else torch.rand(5, 6) * 100
    return StereoResult(
        disparity=d, confidence=torch.rand_like(d), extras={"occlusion": d > 50}, model="m", backend="portable"
    )


def test_stereo_save(tmp_path):
    r = stereo()
    d = r.disparity.numpy()
    r.save(tmp_path / "d.tif")
    np.testing.assert_array_equal(read(tmp_path / "d.tif"), d)
    r.save(tmp_path / "d.npy")
    np.testing.assert_array_equal(np.load(tmp_path / "d.npy"), d)
    r.save(tmp_path / "d.pfm")
    assert (tmp_path / "d.pfm").stat().st_size > d.nbytes
    r.save(tmp_path / "d.png")
    np.testing.assert_allclose(read(tmp_path / "d.png") / 256, d, atol=1 / 512 + 1e-6)
    r.save(tmp_path / "c.tif", key="confidence", bits=12)
    np.testing.assert_allclose(read(tmp_path / "c.tif"), r.confidence.numpy(), rtol=2.0**-13)


def test_stereo_save_errors(tmp_path):
    with pytest.raises(InputError, match=r"must be \[H, W\]"):
        stereo(batch=True).save(tmp_path / "d.tif")
    with pytest.raises(InputError, match="no output 'nope'"):
        stereo().save(tmp_path / "d.tif", key="nope")
    with pytest.raises(InputError, match=r"StereoResult\.save cannot write"):
        stereo().save(tmp_path / "d.exr")
    with pytest.raises(OptionsError, match="only holds key='disparity'"):
        stereo().save(tmp_path / "c.png", key="confidence")


def depth_result(**kw):
    fields = (
        dict.fromkeys(("depth", "inverse_depth", "points", "intrinsics", "confidence", "mask", "normals"), None)
        | {"extras": {}, "kind": "metric", "model": "m", "backend": "portable"}
        | kw
    )
    return DepthResult(**fields)


def test_depth_save_picks_depth_then_inverse_depth(tmp_path):
    x = torch.rand(1, 4, 4)  # a leading 1 is dropped
    depth_result(inverse_depth=x).save(tmp_path / "a.tif")
    np.testing.assert_array_equal(read(tmp_path / "a.tif"), x[0].numpy())
    y = torch.rand(4, 4)
    depth_result(depth=y, inverse_depth=x[0]).save(tmp_path / "b.tif")
    np.testing.assert_array_equal(read(tmp_path / "b.tif"), y.numpy())
    with pytest.raises(InputError, match="no output"):
        depth_result().save(tmp_path / "c.tif")


def test_multiview_save(tmp_path):
    fields = dict.fromkeys(("world_points", "confidence", "mask", "intrinsics", "cam_to_world"), None)
    kw = {"extras": {}, "views": (0,), "kind": "metric", "model": "m", "backend": "portable"}
    d = torch.rand(1, 4, 5)
    MultiViewResult(depth=d, **fields, **kw).save(tmp_path / "d.tif")
    np.testing.assert_array_equal(read(tmp_path / "d.tif"), d[0].numpy())
    with pytest.raises(InputError, match=r"must be \[H, W\]"):
        MultiViewResult(depth=torch.rand(2, 4, 5), **fields, **kw).save(tmp_path / "e.tif")


def test_save_transfers_only_the_requested_map(tmp_path, monkeypatch):
    from fastforks._core import outputs  # noqa: PLC0415

    r = stereo()
    want = r.disparity.numpy().copy()
    seen: list[tuple[int, ...]] = []
    real = outputs._array
    monkeypatch.setattr(outputs, "_array", lambda t: (seen.append(tuple(t.shape)), real(t))[1])
    r.save(tmp_path / "d.npy")
    assert len(seen) == 1
    seen.clear()
    r.save(tmp_path / "o.npy", key="occlusion")
    assert len(seen) == 1
    np.testing.assert_array_equal(np.load(tmp_path / "d.npy"), want)
    with pytest.raises(InputError, match="available"):
        r.save(tmp_path / "x.npy", key="nope")
