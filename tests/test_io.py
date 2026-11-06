from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from fastforks import InputError
from fastforks.stereo import write_disparity

D = np.array([[0.0, 1.5, 300.0], [2.25, 4.0, 8.0]], np.float32)


def read_pfm(path):
    with path.open("rb") as f:
        assert f.readline() == b"Pf\n"
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        data = np.frombuffer(f.read(), "<f4" if scale < 0 else ">f4")
    return data.reshape(h, w)[::-1]


def test_formats(tmp_path):
    write_disparity(tmp_path / "d.npy", torch.from_numpy(D))
    np.testing.assert_array_equal(np.load(tmp_path / "d.npy"), D)
    write_disparity(tmp_path / "d.pfm", D)
    np.testing.assert_array_equal(read_pfm(tmp_path / "d.pfm"), D)
    write_disparity(tmp_path / "d.png", D)
    png = np.asarray(Image.open(tmp_path / "d.png")).astype(np.float32) / 256
    np.testing.assert_array_equal(png, np.minimum(D, 65535 / 256))


def test_rejects(tmp_path):
    with pytest.raises(InputError, match="unknown map format"):
        write_disparity(tmp_path / "d.jpg", D)
    with pytest.raises(InputError, match=r"must be \[H, W\]"):
        write_disparity(tmp_path / "d.npy", D[None])


def test_png_invalid_is_zero(tmp_path):
    d = np.array([[np.nan, np.inf, -np.inf, 2.0]], np.float32)
    write_disparity(tmp_path / "d.png", d)  # no cast warning (filterwarnings = error)
    assert np.asarray(Image.open(tmp_path / "d.png")).tolist() == [[0, 0, 0, 512]]


@pytest.mark.gpu
def test_write_map_stages_cuda_maps_through_one_pinned_buffer(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from fastforks._core import io  # noqa: PLC0415

    big = torch.rand(1100, 1000, device="cuda") * 100  # > 4 MB: staged
    small = torch.rand(30, 40, device="cuda")
    for name, t in (("big", big), ("small", small), ("big2", big[:900, :700])):
        for suffix in (".tif", ".npy"):
            write_disparity(tmp_path / f"{name}{suffix}", t)
    assert np.array_equal(np.load(tmp_path / "big.npy"), big.cpu().numpy())
    assert np.array_equal(np.load(tmp_path / "big2.npy"), big[:900, :700].cpu().numpy())  # strided view
    assert np.array_equal(np.load(tmp_path / "small.npy"), small.cpu().numpy())
    buf = io._STAGING.buf
    assert buf is not None
    assert buf.nbytes >= big.numel() * 4

    with io._STAGING.lock:  # busy (another writer): falls back to a pageable download
        write_disparity(tmp_path / "busy.npy", big)
    assert np.array_equal(np.load(tmp_path / "busy.npy"), big.cpu().numpy())
    assert io._STAGING.buf is buf
