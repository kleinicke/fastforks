"""Mac test round 2 fixes: output overwrite checks, `.NPY` names, truncated images, COLMAP images.txt, numeric unit."""

from __future__ import annotations

import io
import types

import numpy as np
import pytest
import torch
from PIL import Image

import fastforks
from fastforks import InputError
from fastforks._core.image import to_rgb_batch
from fastforks.stereo import geometry as g
from fastforks.stereo.pipeline import run_many


def _images(d, *names):
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(d / n)


class _Fake:
    device = torch.device("cpu")

    def __call__(self, left, right):  # noqa: ARG002
        z = torch.zeros(left.shape[0], *left.shape[-2:])
        return types.SimpleNamespace(disparity=z, confidence=None, extras={})


def test_output_may_not_overwrite_an_input(tmp_path):  # T16-B1
    _images(tmp_path / "L", "a.png")
    _images(tmp_path / "R", "a.png")
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "L")
    for out in (tmp_path / "L", link):
        with pytest.raises(InputError, match="overwrite the input"):
            run_many(None, str(tmp_path / "L"), str(tmp_path / "R"), save_to=out, save_format=".png")  # type: ignore[arg-type]
    with Image.open(tmp_path / "L" / "a.png") as im:  # untouched
        assert im.mode == "RGB"
    # the input directory with other names stays allowed (a .tif next to the .png passes the check)
    list(run_many(_Fake(), str(tmp_path / "L"), str(tmp_path / "R"), save_to=tmp_path / "L", save_format=".tif"))  # type: ignore[arg-type]


def test_output_that_is_a_directory_is_an_input_error(tmp_path):  # T16-M1
    _images(tmp_path / "L", "frame.png")
    _images(tmp_path / "R", "frame.png")
    (tmp_path / "OUT" / "frame.npy").mkdir(parents=True)
    with pytest.raises(InputError, match=r"frame\.npy is a directory"):
        list(
            run_many(
                _Fake(),
                str(tmp_path / "L"),
                str(tmp_path / "R"),
                save_to=tmp_path / "OUT",  # type: ignore[arg-type]
                save_format=".npy",
                skip_existing=True,
            )
        )


def test_save_keeps_an_uppercase_npy_name(tmp_path):  # HANA-1
    fastforks.save(tmp_path / "map.NPY", np.ones((2, 3), np.float32))
    assert [p.name for p in tmp_path.iterdir()] == ["map.NPY"]
    assert np.load(tmp_path / "map.NPY").shape == (2, 3)


def test_truncated_and_oversized_images_are_input_errors(monkeypatch):  # IM-1
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(buf, "JPEG")
    cut = buf.getvalue()[:-300]
    with pytest.raises(InputError):
        to_rgb_batch(Image.open(io.BytesIO(cut)), torch.device("cpu"))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    png = io.BytesIO()
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(png, "PNG")
    with pytest.raises(InputError):
        to_rgb_batch(png.getvalue(), torch.device("cpu"))


def test_colmap_images_txt_alternates_image_and_observation_lines(tmp_path):  # VIK-01
    (tmp_path / "cameras.txt").write_text("# c\n1 PINHOLE 64 48 50 50 32 24\n")
    (tmp_path / "images.txt").write_text(
        "# images\n"
        "1 1 0 0 0 -1000 -2000 -3000 1 l.png\n"
        "0 1 0 0 0 -1 0 0 1 10 10 -1\n"  # an observation row that looks like an image line
        "2 1 0 0 0 -1100 -2000 -3000 1 r.png\n"
        "\n"  # an empty observation line
    )
    rig = g.read_calib(tmp_path, "colmap", unit="mm")
    assert rig.baseline == pytest.approx(0.1)


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), "km", True])
def test_unit_accepts_positive_numbers_only(tmp_path, bad):  # VIK-02
    (tmp_path / "cameras.txt").write_text("1 PINHOLE 64 48 50 50 32 24\n")
    (tmp_path / "images.txt").write_text("1 1 0 0 0 0 0 0 1 l.png\n\n2 1 0 0 0 -100 0 0 1 r.png\n\n")
    assert g.read_calib(tmp_path, "colmap", unit=0.001).baseline == pytest.approx(0.1)
    with pytest.raises(InputError, match="unit must be"):
        g.read_calib(tmp_path, "colmap", unit=bad)
