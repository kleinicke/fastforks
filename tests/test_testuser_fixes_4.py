"""Test-user findings TK-1, TK-3, TK-4, TD-1, TD-3, TD-4, TD-5 (CPU, no weights)."""

from __future__ import annotations

import io
import logging
import os
import types

import numpy as np
import pytest
import torch
from PIL import Image

import fastforks
import fastforks.stereo.geometry as g
from fastforks import FastforksError, InputError
from fastforks._core import files
from fastforks._core.errors import ArgumentError
from fastforks._core.image import to_rgb_batch
from fastforks.stereo.calibrated import resolve_geometry
from fastforks.stereo.pipeline import run_many

CPU = torch.device("cpu")


def _png(arr: np.ndarray, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt)
    return buf.getvalue()


def test_encoded_bytes_decode_like_a_path(tmp_path):
    arr = np.random.default_rng(0).integers(0, 256, (6, 8, 3), dtype=np.uint8)
    data = _png(arr)
    (tmp_path / "a.png").write_bytes(data)
    ref = to_rgb_batch(str(tmp_path / "a.png"), CPU)
    for src in (data, bytearray(data), io.BytesIO(data)):
        assert torch.equal(to_rgb_batch(src, CPU), ref)
    assert ref.shape == (1, 3, 6, 8)
    jpg = _png(arr, "JPEG")
    assert to_rgb_batch(jpg, CPU).shape == (1, 3, 6, 8)


@pytest.mark.parametrize("bad", [b"not an image", b"", bytearray(b"\x89PNG\r\n\x1a\ntruncated")])
def test_corrupt_bytes_raise_input_error(bad):
    with pytest.raises(InputError, match="cannot read image"):
        to_rgb_batch(bad, CPU)


def test_16bit_colour_png_bytes_are_refused_like_a_file():
    arr = np.zeros((4, 4, 3), np.uint16)
    buf = io.BytesIO()
    import zlib  # noqa: PLC0415

    def chunk(kind: bytes, body: bytes) -> bytes:
        return len(body).to_bytes(4, "big") + kind + body + zlib.crc32(kind + body).to_bytes(4, "big")

    raw = b"".join(b"\x00" + arr[r].astype(">u2").tobytes() for r in range(4))
    ihdr = (4).to_bytes(4, "big") * 2 + bytes([16, 2, 0, 0, 0])
    buf.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    with pytest.raises(InputError, match="16-bit colour PNG"):
        to_rgb_batch(buf.getvalue(), CPU)


def test_run_many_takes_bytes_lists_and_refuses_a_single_bytes_side(tmp_path):
    data = _png(np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(InputError, match="one image"):
        run_many(None, data, [data])  # type: ignore[arg-type]
    with pytest.raises(InputError, match="one image"):
        run_many(None, [data], io.BytesIO(data))  # type: ignore[arg-type]


def test_load_keyword_typo_is_an_argument_error():
    with pytest.raises(ArgumentError, match="did you mean 'mode'") as e:
        fastforks.load("s2m2-s", mdoe="kernels")  # ty: ignore[unknown-argument]
    assert isinstance(e.value, FastforksError)
    assert isinstance(e.value, TypeError)
    for task in (fastforks.stereo, fastforks.depth, fastforks.classification, fastforks.promptable_segmentation):
        with pytest.raises(ArgumentError, match="did you mean 'dtype'"):
            task.load("x", dtpye="float16")  # ty: ignore[unknown-argument]
    assert fastforks.ArgumentError is ArgumentError


def test_vendored_loggers_live_under_fastforks():
    from fastforks.shared.depth_anything_v2.net import dinov2  # noqa: PLC0415

    assert dinov2.logger.name.startswith("fastforks.")
    assert logging.getLogger("fastforks.dinov2") is dinov2.logger


def test_same_aspect_size_change_warns_once(caplog, monkeypatch):
    monkeypatch.setattr("fastforks.stereo.calibrated._WARNED", set())  # an earlier test may have warned already
    rig = g.RectifiedRig(700.0, 0.54, size=(375, 1242))
    model = types.SimpleNamespace()
    with caplog.at_level(logging.WARNING, logger="fastforks.stereo.calibrated"):
        resolve_geometry(model, rig, (188, 621), size=None, focal=None)
        resolve_geometry(model, rig, (188, 621), size=None, focal=None)  # cached: silent
        resolve_geometry(model, rig, (375, 1242), size=None, focal=None)  # the calibrated size: silent
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 1
    assert "resize" in msgs[0]
    assert "188x621" in msgs[0]
    assert "375x1242" in msgs[0]
    assert "crop" in msgs[0]


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_read_only_cache_is_a_weights_download_error(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        spec = files.FileSpec("w.bin", "0" * 64, 4, "https://example.invalid/w.bin")
        with pytest.raises(files.WeightsDownloadError, match=r"FASTFORKS_CACHE.*FASTFORKS_WEIGHTS_PATH"):
            files.download(spec.url or "", ro / "sub" / "blob", spec)
    finally:
        ro.chmod(0o700)


def test_colliding_output_names_write_nothing(tmp_path):
    out = tmp_path / "out"
    a, b = tmp_path / "a" / "im0.png", tmp_path / "b" / "im0.png"
    with pytest.raises(InputError, match="both be saved as im0"):
        run_many(None, [a, b], [a, b], save_to=out)  # type: ignore[arg-type]
    assert not out.exists() or not any(out.iterdir())
    with pytest.raises(InputError, match="both be saved as im0"):  # case-insensitive file systems
        run_many(None, [tmp_path / "IM0.png", a], [a, b], save_to=out)  # type: ignore[arg-type]
