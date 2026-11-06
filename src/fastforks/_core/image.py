"""Input images -> float32 RGB tensors `[B, 3, H, W]` in `[0, 1]`.

Accepted: file paths, encoded image bytes (`bytes`, `bytearray`, `io.BytesIO`: a PNG / JPEG / ... file's content, e.g.
an upload; decoded like a path), PIL images, numpy arrays (channel-last: `HW`, `HWC`, `BHWC`) and torch tensors
(channel-first: `HW`, `CHW`, `BCHW`). Integer images are scaled by their dtype's maximum (uint8: 255, uint16: 65535);
floating-point images must already be in `[0, 1]`. One channel is replicated to RGB, a fourth (alpha) channel is
dropped. numpy arrays and PIL images are RGB (convert OpenCV's BGR first). Files are read as stored: the EXIF
orientation tag of a JPEG is not applied (apply `PIL.ImageOps.exif_transpose` and pass the image if the photo should
be upright; one WARNING per process says so).

Paths are a convenience for standard images read with Pillow: PNG, JPEG, BMP, WebP, and 8/16-bit grayscale TIFF / PNG.
Files Pillow would read lossily (16-bit colour PNG / TIFF, float or 32-bit integer images) raise `InputError`; load
anything else (scientific TIFF, EXR, raw, .npy, OpenCV) yourself and pass a numpy array (HWC, RGB) or a torch tensor
(CHW).
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import torch
from PIL import Image

from fastforks._core.errors import InputError

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

log = logging.getLogger(__name__)
_exif_warned = False

ImageLike: TypeAlias = str | os.PathLike[str] | bytes | bytearray | io.BytesIO | Image.Image | np.ndarray | torch.Tensor
"""Anything `to_rgb_batch` accepts."""

_INT_MAX = {torch.uint8: 255.0, torch.uint16: 65535.0}
_PIL_MODES = ("L", "I;16", "I;16L", "I;16B", "RGB", "RGBA")  # read as they are
_PIL_TO_RGB = ("P", "1", "LA")  # 8-bit or less per channel: convert("RGB") is lossless
_PIL_8BIT_OTHER = ("CMYK", "YCbCr", "LAB", "HSV")  # 8 bits per channel, not RGB
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_GRAY = 0  # IHDR colour type
_HEAD = 26  # bytes of a PNG up to its colour type
_BYTE = 8
_UINT16_BITS, _UINT16_MAX = 16, 65535
_EXIF_ORIENTATION = 0x0112
_TIFF_BITS, _TIFF_SAMPLES = 258, 277  # BitsPerSample, SamplesPerPixel
_CHECKED: ContextVar[bool] = ContextVar("fastforks_images_checked", default=False)  # see `checked_ranges`
_LOSSLESS_FIX = (
    "load it with tifffile or cv2.imread(path, cv2.IMREAD_UNCHANGED) (then BGR->RGB) and pass the numpy array"
    " (HWC, uint16 or float in [0, 1])"
)
_RGB, _RGBA = 3, 4
_BATCH_DIMS = 4


def _as_tensor(img: ImageLike) -> torch.Tensor:
    """Channel-first tensor `[B, C, H, W]` in the input dtype, except float64 -> float32.

    The one conversion of every image input (model calls, calibrated / two-view calls, `run_many`): models compute in
    float32 and MPS has no float64, so float64 images give the same numbers as their float32 copy on every path.
    """
    if isinstance(img, (str, os.PathLike, bytes, bytearray, io.BytesIO)):
        return _as_tensor(decode_image(img))
    shape = tuple(getattr(img, "shape", ()))  # as the caller passed it, for the error messages
    if isinstance(img, Image.Image):
        img = _pil_image(img)
    if isinstance(img, np.ndarray):
        t = _from_numpy(img)
    elif isinstance(img, torch.Tensor):
        t = img.float() if img.dtype == torch.float64 else img
    else:
        hint = (
            ": lists are not batches; stack same-size images into a BHWC array or BCHW tensor (np.stack / torch.stack)"
            if isinstance(img, (list, tuple))
            else " (path, encoded bytes, PIL image, numpy array or torch tensor)"
        )
        msg = f"unsupported image type {type(img).__name__}{hint}"
        raise InputError(msg)
    if t.ndim == 2:  # noqa: PLR2004 - HW
        t = t[None]
    if t.ndim == 3:  # noqa: PLR2004 - CHW
        t = t[None]
    if t.ndim != 4:  # noqa: PLR2004 - BCHW
        msg = f"image must have 2-4 dimensions, got shape {tuple(t.shape)}"
        raise InputError(msg)
    if 0 in t.shape:
        msg = f"empty image or batch, shape {shape or tuple(t.shape)}"
        raise InputError(msg)
    _check_channels(t, img, shape or tuple(t.shape))
    return t


def _check_channels(t: torch.Tensor, img: object, shape: tuple[int, ...]) -> None:
    """Raise `InputError` unless `t` (`[B, C, H, W]`) has 1, 3 or 4 channels; `shape` is the caller's input shape."""
    if t.shape[1] in (1, _RGB, _RGBA):
        return
    hint = "numpy is HWC, torch is CHW"
    if isinstance(img, np.ndarray) and img.ndim == _RGB and img.shape[-1] > _RGBA:
        hint += "; a gray batch BHW? use arr[..., None] (BHW1)"
    msg = f"image must have 1, 3 or 4 channels, got {t.shape[1]} (shape {shape}; {hint})"
    raise InputError(msg)


def decode_image(source: str | os.PathLike[str] | bytes | bytearray | io.BytesIO) -> np.ndarray:
    """Read an image file (or its encoded bytes) as the channel-last numpy array the model calls use (`HW` / `HWC`).

    The decode of every path / bytes input (Pillow, same checks); thread-safe, so pipelines call it from reader threads.
    The array is uint8 / uint16.

    Raises:
        InputError: The file is missing or unreadable, the bytes are not an image, or Pillow would read it lossily
            (see the module docstring).
    """
    if isinstance(source, (bytes, bytearray, io.BytesIO)):
        data = source.getvalue() if isinstance(source, io.BytesIO) else bytes(source)
        name, head, fp = f"<{type(source).__name__}>", data[:_HEAD], io.BytesIO(data)
    else:
        name = os.fsdecode(source)
        head, fp = None, name
    try:
        with Image.open(fp) as im:
            gray16 = _check_lossless_file(im, name, head)
            if not _exif_warned:
                _warn_orientation(im, name)
            return _pil_array(im, f"{name}: ", gray16=gray16)
    except InputError:
        raise
    except (OSError, ValueError, Image.DecompressionBombError) as e:  # missing, not an image, truncated, too large
        msg = f"cannot read image {name!r}: {e}"
        raise InputError(msg) from e


def _warn_orientation(im: Image.Image, name: str) -> None:
    """One WARNING per process the first time a decoded file has an EXIF orientation other than 1 (pixels unchanged)."""
    global _exif_warned  # noqa: PLW0603 - once per process
    try:
        orientation = im.getexif().get(_EXIF_ORIENTATION, 1)
    except Exception:  # noqa: BLE001 - a damaged EXIF block must not fail the decode
        return
    if orientation != 1:
        _exif_warned = True
        log.warning(
            "%s has EXIF orientation %s: the stored pixels are used, not the rotated view (this warning is shown"
            " once; for an upright photo apply PIL.ImageOps.exif_transpose(Image.open(path)) and pass the image)",
            name,
            orientation,
        )


def _pil_image(im: Image.Image) -> np.ndarray:
    """`_pil_array` of a passed PIL image; a lazily opened truncated or oversized file is an `InputError`."""
    try:
        return _pil_array(im, "PIL image: ")
    except InputError:
        raise
    except (OSError, ValueError, Image.DecompressionBombError) as e:
        msg = f"cannot read the PIL image: {e}"
        raise InputError(msg) from e


def _pil_array(im: Image.Image, where: str, *, gray16: bool = False) -> np.ndarray:
    """The pixels of `im` as numpy; modes that `convert("RGB")` would clip or round raise.

    `gray16`: the file header says 16-bit grayscale, which Pillow < 13 opens as mode `I` (int32) instead of `I;16`;
    such an image is returned as uint16.
    """
    if im.mode in _PIL_MODES:
        return np.asarray(im)
    if im.mode == "I" and gray16:
        a = np.asarray(im)
        if a.size and (a.min() < 0 or a.max() > _UINT16_MAX):
            msg = f"{where}16-bit grayscale file with values outside [0, {_UINT16_MAX}] ({a.min()}..{a.max()})"
            raise InputError(msg)
        return a.astype(np.uint16)
    if im.mode in _PIL_TO_RGB:
        return np.asarray(im.convert("RGB"))
    if im.mode in _PIL_8BIT_OTHER:  # 8-bit per channel: the Pillow conversion is the right one
        msg = f'{where}Pillow mode {im.mode} is not RGB; convert it yourself: Image.open(path).convert("RGB")'
        raise InputError(msg)
    msg = f"{where}Pillow mode {im.mode} would be converted to 8-bit RGB lossily; {_LOSSLESS_FIX}"
    raise InputError(msg)


def _check_lossless_file(im: Image.Image, path: str, head: bytes | None = None) -> bool:
    """Raise for files Pillow opens in a lossy 8-bit mode (16-bit colour PNG, >8-bit multi-channel TIFF).

    Returns:
        Whether the header says 9-16-bit grayscale (one channel), so a mode `I` image of it is uint16 data.
    """
    if im.format == "PNG":
        if head is None:
            with Path(path).open("rb") as f:
                head = f.read(_HEAD)  # signature, IHDR length + type, width, height, bit depth, colour type
        if head[:8] == _PNG_SIGNATURE and head[24] > _BYTE and head[25] != _PNG_GRAY:
            msg = f"{path}: {head[24]}-bit colour PNG, which Pillow reads as 8-bit; {_LOSSLESS_FIX}"
            raise InputError(msg)
        return head[:8] == _PNG_SIGNATURE and head[24] > _BYTE and head[25] == _PNG_GRAY
    if im.format == "TIFF":
        tags = getattr(im, "tag_v2", {})
        bits = tags.get(_TIFF_BITS, 1)
        bits = max(bits) if isinstance(bits, tuple) else bits
        samples = tags.get(_TIFF_SAMPLES, 1)
        if bits > _BYTE and samples > 1:
            msg = f"{path}: {samples}-channel {bits}-bit TIFF, which Pillow reads as 8-bit; {_LOSSLESS_FIX}"
            raise InputError(msg)
        return samples == 1 and _BYTE < bits <= _UINT16_BITS
    return False


def _from_numpy(img: np.ndarray) -> torch.Tensor:
    """Channel-last numpy (`HW`, `HWC`, `BHWC`, any byte order) -> channel-first tensor."""
    if img.ndim not in (2, 3, 4):
        msg = f"numpy image must be HW, HWC or BHWC, got shape {img.shape}"
        raise InputError(msg)
    if img.ndim == _RGB and img.shape[0] in (1, _RGB, _RGBA) and img.shape[-1] not in (1, _RGB, _RGBA):
        msg = (
            f"numpy images are channel-last (HWC), got shape {img.shape}: CHW? use img.transpose(1, 2, 0);"
            " a gray batch BHW? use img[..., None] (BHW1)"
        )
        raise InputError(msg)
    if img.dtype.kind == "f" and img.dtype.itemsize == 8:  # noqa: PLR2004 - float64, any byte order
        img = img.astype(np.float32)  # see `_as_tensor`; one copy, also for the byte order
    elif not img.dtype.isnative:
        img = img.astype(img.dtype.newbyteorder("="))
    try:
        t = torch.from_numpy(np.require(img, requirements="CW"))  # PIL arrays are read-only: copy those
    except TypeError as e:  # object, string, datetime arrays
        msg = f"unsupported image dtype {img.dtype} (use uint8, uint16 or float in [0, 1])"
        raise InputError(msg) from e
    return t[None] if t.ndim == 2 else t.movedim(-1, -3)  # noqa: PLR2004 - HW has no channel axis


def to_rgb_batch(img: ImageLike, device: torch.device) -> torch.Tensor:
    """Convert one image (or a batch) to float32 RGB `[B, 3, H, W]` in `[0, 1]` on `device`.

    Raises:
        InputError: Unsupported type, shape, channel count, dtype or value range.
    """
    t, _, bounds = _to_rgb(img, device)
    _check_ranges([bounds])
    return t


def to_rgb_batches(imgs: Sequence[ImageLike], device: torch.device) -> list[torch.Tensor]:
    """`to_rgb_batch` of each image, with one host sync for all value-range checks.

    Raises:
        InputError: As `to_rgb_batch` (the first image out of range is named).
    """
    conv = [_to_rgb(img, device) for img in imgs]
    _check_ranges([b for _, _, b in conv])
    return [t for t, _, _ in conv]


def to_rgb_pair(left: ImageLike, right: ImageLike, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """`to_rgb_batch` of both images of a pair, which must have the same shape and both be gray or both colour.

    Raises:
        InputError: As `to_rgb_batch`, or the two images differ in shape or in gray / colour.
    """
    (lt, lc, lb), (rt, rc, rb) = _to_rgb(left, device), _to_rgb(right, device)
    _check_ranges([lb, rb])  # one host sync for both
    if lt.shape != rt.shape:
        msg = f"left and right differ in shape: {tuple(lt.shape)} vs {tuple(rt.shape)}"
        raise InputError(msg)
    if (lc == 1) != (rc == 1):
        msg = f"left and right differ in colour: {lc} vs {rc} channel(s); convert both to RGB (or both to gray)"
        raise InputError(msg)
    return lt, rt


@contextlib.contextmanager
def checked_ranges() -> Generator[None]:
    """Internal: inside, float images are not range-checked (the caller checked these pixel values already).

    For a second model run on the same images within one logical call (stereo `views="both"`: the mirrored pair).
    """
    token = _CHECKED.set(True)
    try:
        yield
    finally:
        _CHECKED.reset(token)


def _check_ranges(bounds: Sequence[tuple[torch.Tensor, torch.Tensor] | None]) -> None:
    """Raise `InputError` for the first float image whose `[min, max]` (`bounds`) is not finite and in `[0, 1]`."""
    given = [b for b in bounds if b is not None]
    if not given:
        return
    ok = [(lo >= 0) & (hi <= 1) for lo, hi in given]  # also rejects NaN (every comparison with NaN is False)
    if bool(ok[0] if len(ok) == 1 else torch.stack(ok).all()):  # one host sync
        return
    lo, hi = next(b for b, good in zip(given, ok, strict=True) if not bool(good))
    msg = (
        f"float images must be finite and in [0, 1], got [{lo.item():.9g}, {hi.item():.9g}]"
        " (divide 0..255 images by 255; clamp(0, 1) values that overshoot, e.g. after bicubic resizing)"
    )
    raise InputError(msg)


def _to_rgb(img: ImageLike, device: torch.device) -> tuple[torch.Tensor, int, tuple[torch.Tensor, torch.Tensor] | None]:
    """`to_rgb_batch` without the value-range check: the image, its input channels and its float `[min, max]`.

    The bounds are `None` for integer images and inside `checked_ranges`.
    """
    t = _as_tensor(img)
    if t.device.type == "meta":
        msg = "image is a meta tensor (it has no data)"
        raise InputError(msg)
    c = t.shape[1]  # 1, 3 or 4 (checked by `_as_tensor`)
    if c == _RGBA:
        t = t[:, :_RGB]
    bounds = None
    if t.is_floating_point():
        t = t.to(device, torch.float32, non_blocking=True)
        if not _CHECKED.get():
            bounds = t.aminmax()
    else:
        scale = _INT_MAX.get(t.dtype)
        if scale is None:
            msg = f"unsupported image dtype {t.dtype} (use uint8, uint16 or float in [0, 1])"
            raise InputError(msg)
        t = t.to(device, non_blocking=True).to(torch.float32).div_(scale)
    if t.shape[1] == 1:
        t = t.expand(-1, _RGB, -1, -1)
    return t.contiguous(), c, bounds


def is_batch(img: ImageLike) -> bool:
    """Whether `img` is a batch (`BHWC` numpy / `BCHW` torch), i.e. whether results keep the leading `B`."""
    return getattr(img, "ndim", 0) == _BATCH_DIMS


def unbatch(maps: dict[str, torch.Tensor], img: ImageLike) -> dict[str, torch.Tensor]:
    """`maps` of a `[1, ...]` batch without the leading dim, unless `img` was a batch."""
    return maps if is_batch(img) else {k: v[0] for k, v in maps.items()}
