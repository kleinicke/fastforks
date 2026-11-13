from __future__ import annotations

import functools
import http.server
import struct
import threading
import zlib

import numpy as np
import pytest
import torch
from PIL import Image

import fastforks
from fastforks import InputError
from fastforks._core import image as image_mod
from fastforks._core.image import to_rgb_batch
from fastforks.stereo import pipeline

CPU = torch.device("cpu")
RGB = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)  # H=2, W=3, C=3


def test_numpy_hwc_uint8():
    t = to_rgb_batch(RGB, CPU)
    assert t.shape == (1, 3, 2, 3)
    assert t.dtype == torch.float32
    torch.testing.assert_close(t[0], torch.from_numpy(RGB).permute(2, 0, 1).float() / 255)


def test_layouts_agree(tmp_path):
    ref = to_rgb_batch(RGB, CPU)
    path = tmp_path / "x.png"
    Image.fromarray(RGB).save(path)
    for img in (torch.from_numpy(RGB).permute(2, 0, 1), RGB[None], Image.fromarray(RGB), path, str(path)):
        torch.testing.assert_close(to_rgb_batch(img, CPU), ref)


def test_gray_alpha_uint16_float():
    gray = to_rgb_batch(RGB[..., 0], CPU)
    assert gray.shape == (1, 3, 2, 3)
    assert torch.equal(gray[:, 0], gray[:, 2])
    rgba = np.concatenate([RGB, np.full((2, 3, 1), 7, np.uint8)], -1)
    torch.testing.assert_close(to_rgb_batch(rgba, CPU), to_rgb_batch(RGB, CPU))
    u16 = to_rgb_batch(np.full((2, 2), 65535, np.uint16), CPU)
    assert torch.all(u16 == 1)
    f = np.full((2, 2, 3), 0.5, np.float32)
    assert torch.all(to_rgb_batch(f, CPU) == 0.5)


@pytest.mark.parametrize(
    ("img", "match"),
    [
        (np.zeros((2, 2, 2), np.uint8), "1, 3 or 4 channels"),
        (np.zeros((2, 2, 3), np.int32), "unsupported image dtype"),
        (np.full((2, 2, 3), 255.0, np.float32), r"finite and in \[0, 1\]"),
        (np.zeros((1, 1, 2, 2, 3), np.uint8), "HW, HWC or BHWC"),
        ([[0, 1]], "unsupported image type list: lists are not batches; stack"),
        ({"a": 1}, r"unsupported image type dict \(path"),
        (torch.zeros(3, 2, 2, 2, 2), "2-4 dimensions"),
        (np.full((2, 2, 3), np.nan, np.float32), "finite and in"),
        (torch.full((3, 2, 2), float("nan")), "finite and in"),
        (np.zeros((0, 2, 2, 3), np.uint8), "empty image or batch"),
        (torch.zeros(0, 3, 2, 2), "empty image or batch"),
        (np.zeros((3, 4, 5), np.uint8), "CHW"),
    ],
)
def test_rejects(img, match):
    with pytest.raises(InputError, match=match):
        to_rgb_batch(img, CPU)


def test_byte_order():
    big = (np.arange(12, dtype=np.uint16).reshape(2, 2, 3) * 1000).astype(">u2")
    torch.testing.assert_close(to_rgb_batch(big, CPU), to_rgb_batch(big.astype("<u2"), CPU))


def _png16(path, rgb):
    """Write `rgb` (uint16 HWC) as a 16-bit colour PNG (Pillow cannot write one)."""

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    h, w, _ = rgb.shape
    rows = b"".join(b"\x00" + rgb[y].astype(">u2").tobytes() for y in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
    )


def _tiff16(path, rgb):
    """Write `rgb` (uint16 HWC) as an uncompressed 16-bit RGB TIFF (Pillow cannot write one)."""
    h, w, c = rgb.shape
    bits_at, data_at = 8 + 2 + 10 * 12 + 4, 8 + 2 + 10 * 12 + 4 + 2 * c
    entries = [  # tag, type (3 SHORT, 4 LONG), count, value
        (256, 4, 1, w),
        (257, 4, 1, h),
        (258, 3, c, bits_at),
        (259, 3, 1, 1),
        (262, 3, 1, 2),
        (273, 4, 1, data_at),
        (277, 3, 1, c),
        (278, 4, 1, h),
        (279, 4, 1, rgb.nbytes),
        (284, 3, 1, 1),
    ]
    ifd = struct.pack("<H", len(entries)) + b"".join(struct.pack("<HHII", *e) for e in entries) + struct.pack("<I", 0)
    path.write_bytes(
        b"II*\x00" + struct.pack("<I", 8) + ifd + struct.pack(f"<{c}H", *[16] * c) + rgb.astype("<u2").tobytes()
    )


RGB16 = (np.arange(2 * 3 * 3, dtype=np.uint16).reshape(2, 3, 3) * 3000 + 7).astype(np.uint16)


def test_rejects_16bit_colour_png(tmp_path):
    _png16(tmp_path / "x.png", RGB16)
    with Image.open(tmp_path / "x.png") as im:
        assert im.mode == "RGB"  # what Pillow would silently give: 8 bits
    with pytest.raises(InputError, match=r"16-bit colour PNG.*cv2.IMREAD_UNCHANGED"):
        to_rgb_batch(tmp_path / "x.png", CPU)


def test_rejects_16bit_rgb_tiff(tmp_path):
    _tiff16(tmp_path / "x.tif", RGB16)
    with Image.open(tmp_path / "x.tif") as im:
        assert im.mode == "RGB"
    with pytest.raises(InputError, match=r"3-channel 16-bit TIFF.*tifffile"):
        to_rgb_batch(tmp_path / "x.tif", CPU)


def test_rejects_float_tiff(tmp_path):
    Image.fromarray(np.full((2, 3), 0.25, np.float32)).save(tmp_path / "x.tif")
    with pytest.raises(InputError, match="mode F would be converted"):
        to_rgb_batch(tmp_path / "x.tif", CPU)


def test_rejects_mode_i(tmp_path):
    im = Image.fromarray(np.full((2, 3), 70000, np.int32))
    assert im.mode == "I"
    im.save(tmp_path / "x.tif")
    with pytest.raises(InputError, match="mode I would be converted"):
        to_rgb_batch(tmp_path / "x.tif", CPU)
    with pytest.raises(InputError, match="PIL image: Pillow mode I"):
        to_rgb_batch(im, CPU)


def test_lossless_files_load(tmp_path):
    Image.fromarray(RGB).save(tmp_path / "rgb.png")
    torch.testing.assert_close(to_rgb_batch(tmp_path / "rgb.png", CPU), to_rgb_batch(RGB, CPU))
    gray16 = RGB16[..., 0]
    for ext in ("png", "tif"):
        Image.fromarray(gray16).save(tmp_path / f"g16.{ext}")
        torch.testing.assert_close(to_rgb_batch(tmp_path / f"g16.{ext}", CPU), to_rgb_batch(gray16, CPU))
    Image.fromarray(RGB).convert("P").save(tmp_path / "pal.png")
    with Image.open(tmp_path / "pal.png") as im:
        pal = np.asarray(im.convert("RGB"))
    torch.testing.assert_close(to_rgb_batch(tmp_path / "pal.png", CPU), to_rgb_batch(pal, CPU))


def test_16bit_gray_files_opened_as_mode_i(tmp_path, monkeypatch):
    """Pillow < 13 opens 16-bit grayscale PNG / TIFF as mode `I` (int32): the header makes that lossless uint16."""
    gray16 = RGB16[..., 0]
    real_open = Image.open

    def old_pillow_open(path):
        im = real_open(path)
        out = im.convert("I")  # values unchanged, as int32
        out.format = im.format
        if hasattr(im, "tag_v2"):
            out.tag_v2 = im.tag_v2  # TIFF only
        return out

    monkeypatch.setattr(image_mod.Image, "open", old_pillow_open)
    for ext in ("png", "tif"):
        Image.fromarray(gray16).save(tmp_path / f"g16.{ext}")
        with Image.open(tmp_path / f"g16.{ext}") as im:
            assert im.mode == "I"
            assert np.asarray(im).max() > 255  # 16-bit values
        torch.testing.assert_close(to_rgb_batch(tmp_path / f"g16.{ext}", CPU), to_rgb_batch(gray16, CPU))


def test_mode_i_needs_a_16bit_gray_header():
    im = Image.fromarray(RGB16[..., 0].astype(np.int32))
    assert im.mode == "I"
    np.testing.assert_array_equal(image_mod._pil_array(im, "x: ", gray16=True), RGB16[..., 0])
    with pytest.raises(InputError, match="mode I would be converted"):
        image_mod._pil_array(im, "x: ")
    with pytest.raises(InputError, match="outside"):
        image_mod._pil_array(Image.fromarray(np.full((2, 2), 70000, np.int32)), "x: ", gray16=True)


@pytest.fixture
def served(tmp_path):
    """A local http server (thread) serving `x.png` from tmp_path; yields its base URL."""
    Image.fromarray(RGB).save(tmp_path / "x.png")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    handler.log_message = lambda *_: None  # type: ignore[attr-defined]
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", tmp_path / "x.png"
    srv.shutdown()
    srv.server_close()


def test_url_decodes_like_bytes(served):
    url, path = served
    np.testing.assert_array_equal(image_mod.decode_image(f"{url}/x.png"), image_mod.decode_image(path.read_bytes()))
    torch.testing.assert_close(to_rgb_batch(f"{url}/x.png", CPU), to_rgb_batch(path, CPU))


def test_url_errors_are_input_errors(served):
    url, _ = served
    with pytest.raises(InputError, match=r"missing\.png"):
        image_mod.decode_image(f"{url}/missing.png")
    with pytest.raises(InputError, match="cannot download"):
        image_mod.decode_image("http://127.0.0.1:9/x.png")


def test_url_size_cap(served, monkeypatch):
    url, _ = served
    monkeypatch.setattr(image_mod, "_URL_MAX_BYTES", 10)
    with pytest.raises(InputError, match="larger than"):
        image_mod.decode_image(f"{url}/x.png")


def test_url_in_stereo_pipeline(served):

    url, path = served
    names = [f"{url}/x.png?v=1"]
    left, right, n, _ = pipeline.check_inputs(names, f"{url}/x.png")
    assert n == 1
    assert [nm for _, nm in left] == ["x"]
    assert [nm for _, nm in right] == ["x"]
    assert pipeline._path(names[0]) is None
    assert pipeline._peek(names[0]) is None
    assert pipeline._peek(str(path)) is not None


def test_files_of_one_call_decode_in_parallel(tmp_path, monkeypatch):
    """A pair / view list of files: the same tensors as one by one, decoded on several threads, first error first."""
    rng = np.random.default_rng(0)
    arrays = [rng.integers(0, 255, (9, 12, 3), dtype=np.uint8) for _ in range(3)]
    paths = [tmp_path / f"{i}.png" for i in range(3)]
    for a, p in zip(arrays, paths, strict=True):
        Image.fromarray(a).save(p)
    threads = set()
    decode = image_mod.decode_image
    monkeypatch.setattr(image_mod, "decode_image", lambda src: (threads.add(threading.get_ident()), decode(src))[1])
    mixed = [paths[0], arrays[1], paths[2].read_bytes()]
    for got, a in zip(image_mod.to_rgb_batches(mixed, CPU), (arrays[0], arrays[1], arrays[2]), strict=True):
        assert torch.equal(got, image_mod.to_rgb_batch(a, CPU))
    assert len(threads) == 2  # the caller's thread and one worker
    left, right = image_mod.to_rgb_pair(paths[0], str(paths[0]), CPU)
    assert torch.equal(left, right)
    with pytest.raises(InputError, match="missing_a"):
        image_mod.to_rgb_batches([tmp_path / "missing_a.png", tmp_path / "missing_b.png"], CPU)


def test_read_images_decodes_ahead_in_order(tmp_path) -> None:
    """`read_images`: the arrays of `decode_image` in order, from a generator, bounded ahead; errors at their item."""
    paths = []
    for i in range(12):
        paths.append(tmp_path / f"{i}.png")
        Image.fromarray(np.full((4, 6, 3), i, np.uint8)).save(paths[-1])
    taken = []
    reader = image_mod.read_images((taken.append(p) or p for p in paths), workers=2)
    assert not taken  # nothing is read before the loop starts
    first = next(reader)
    assert len(taken) == 4  # 2 * workers ahead, not the whole stream
    out = [first, *reader]
    assert [int(a[0, 0, 0]) for a in out] == list(range(12))
    assert not [t for t in threading.enumerate() if t.name.startswith("fastforks-read")]
    mixed = [paths[0], paths[1].read_bytes(), str(tmp_path / "missing.png"), paths[2]]
    reader = image_mod.read_images(mixed)
    assert [int(next(reader)[0, 0, 0]) for _ in range(2)] == [0, 1]
    with pytest.raises(InputError, match="missing"):
        next(reader)
    assert fastforks.read_images is image_mod.read_images
