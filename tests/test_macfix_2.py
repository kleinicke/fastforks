"""Mac test-user fixes (batch 2): TIFF description, mantissa rounding limits, EXIF warning, CMYK hint, CLI messages."""

from __future__ import annotations

import io
import logging

import numpy as np
import pytest
import torch
from PIL import Image

from fastforks import InputError, OptionsError, classification
from fastforks._core import cli
from fastforks._core import image as image_mod
from fastforks._core import io as ffio
from fastforks._core.image import decode_image
from fastforks.depth import DepthResult


def test_tiff_description_tag(tmp_path):
    for shape in [(3, 5), (200, 4)]:  # one strip, several strips
        a = np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape)
        ffio.write_tiff(tmp_path / "a.tif", a, 10, "fastforks s2m2-s@1: disparity (pixels)")
        with Image.open(tmp_path / "a.tif") as im:
            assert im.tag_v2[270] == "fastforks s2m2-s@1: disparity (pixels)"
            np.testing.assert_allclose(np.asarray(im), a, rtol=2.0**-11)
    ffio.write_tiff(tmp_path / "b.tif", a, None, "é")  # non-ASCII replaced, tag still valid
    with Image.open(tmp_path / "b.tif") as im:
        assert 270 in im.tag_v2
        np.testing.assert_array_equal(np.asarray(im), a)


def test_result_save_writes_model_and_kind(tmp_path):
    r = DepthResult(
        depth=torch.ones(4, 6),
        inverse_depth=None,
        points=None,
        intrinsics=None,
        mask=None,
        normals=None,
        confidence=None,
        extras={},
        model="moge-3-vitl@1",
        kind="metric",
        backend=None,
    )
    r.save(tmp_path / "d.tif")
    with Image.open(tmp_path / "d.tif") as im:
        assert im.tag_v2[270] == "fastforks moge-3-vitl@1: depth, kind metric"


def test_mantissa_rounding_keeps_overflow_and_subnormals(tmp_path):
    big = np.finfo(np.float32).max
    sub = np.float32(1e-45)  # the smallest subnormal: would round to 0
    a = np.array([[big, -big, sub, 0.0, 1.0001, np.inf, np.nan]], np.float32)
    ffio.write_tiff(tmp_path / "a.tif", a, 10)
    b = np.asarray(Image.open(tmp_path / "a.tif"))
    assert b[0, 0] == big
    assert b[0, 1] == -big
    assert b[0, 2] == sub
    assert b[0, 5] == np.inf
    assert np.isnan(b[0, 6])
    np.testing.assert_allclose(b[0, 4], 1.0001, rtol=2.0**-11)


def _jpeg(orientation):
    exif = Image.Exif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    Image.new("RGB", (8, 4), (10, 20, 30)).save(buf, "JPEG", exif=exif)
    return buf.getvalue()


def test_exif_orientation_warns_once_without_changing_pixels(monkeypatch, caplog):
    monkeypatch.setattr(image_mod, "_exif_warned", False)
    with caplog.at_level(logging.WARNING, logger="fastforks._core.image"):
        assert decode_image(_jpeg(1)).shape == (4, 8, 3)
        assert not caplog.records
        assert decode_image(_jpeg(6)).shape == (4, 8, 3)  # stored pixels, not rotated
        assert decode_image(_jpeg(3)).shape == (4, 8, 3)
    assert len(caplog.records) == 1
    assert "exif_transpose" in caplog.records[0].getMessage()


def test_cmyk_hint():
    buf = io.BytesIO()
    Image.new("CMYK", (4, 4)).save(buf, "JPEG")
    with pytest.raises(InputError, match=r'convert\("RGB"\)'):
        decode_image(buf.getvalue())


def test_empty_label_name_rejected():
    with pytest.raises(OptionsError, match="non-empty"):
        classification.load("resnet18.tv_in1k", labels=[""] * 1000)


def test_export_output_directory(tmp_path, capsys):
    assert cli.main(["export", "resnet18.tv_in1k", "--format", "onnx", "-o", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "fastforks: error:" in err
    assert "is a directory" in err


def test_info_apple_row_names_auto(capsys):
    assert cli.main(["info", "moge-3-vitl"]) == 0
    out = capsys.readouterr().out
    assert "auto: kernels (Metal" in out
    assert cli.main(["info", "resnet18.tv_in1k"]) == 0
    assert "apple gpu" in capsys.readouterr().out


def test_download_dry_run_counts_the_archive_once(capsys):
    assert cli.main(["download", "raft-stereo-middlebury", "--dry-run"]) == 0
    line = capsys.readouterr().out.splitlines()[0]
    assert "192" in line  # the 192 MiB zip, not the 43 MB member
