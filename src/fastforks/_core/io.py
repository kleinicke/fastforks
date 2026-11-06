"""Writing dense float maps: float32 TIFF (deflate + floating-point predictor), `.npy`, `.pfm`."""

from __future__ import annotations

import contextlib
import os
import struct
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fastforks._core.errors import InputError

try:  # Intel ISA-L: same deflate format as zlib, several times faster
    from isal.isal_zlib import compress as _deflate
except ImportError:  # platform without isal wheels
    from zlib import compress as _deflate

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    import torch

_ROWS = 64  # rows per TIFF strip: the unit of parallel compression
_MANTISSA = 23  # float32 mantissa bits
_TAGS = 11  # without the optional ImageDescription
_ASCII, _DESCRIPTION = 2, 270  # TIFF field type, tag
_SHORT, _LONG = 3, 4  # TIFF field types
_PINNED_MIN = 1 << 22  # bytes: smaller maps are downloaded through pageable memory
_PAGE = 4096


class _Staging:
    """One page-locked host buffer for downloading maps to write (grows to the largest map, kept for the process).

    `cudaHostRegister` on an aligned numpy buffer: no power-of-two rounding as with `pin_memory`, and the copy out of
    the GPU runs at full PCIe speed (12 MP float32: 3 ms instead of 28 ms pageable). A writer that finds it busy
    (another thread is writing) downloads through pageable memory instead of waiting.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.buf: np.ndarray | None = None
        self.unregister: Callable[[], object] | None = None
        self.broken = False  # registration refused once: do not retry

    def _grow(self, nbytes: int) -> np.ndarray | None:
        import torch  # noqa: PLC0415 - only for GPU maps

        if self.buf is not None and self.buf.nbytes >= nbytes:
            return self.buf
        if self.unregister is not None:
            self.unregister()
            self.buf = self.unregister = None
        raw = np.empty(nbytes + _PAGE, np.uint8)
        buf = raw[-raw.ctypes.data % _PAGE :][:nbytes]
        cudart = torch.cuda.cudart()
        if int(cudart.cudaHostRegister(buf.ctypes.data, buf.nbytes, 0)) != 0:
            self.broken = True
            return None
        ptr = buf.ctypes.data
        self.buf, self.unregister = buf, lambda: cudart.cudaHostUnregister(ptr)
        return buf

    @contextlib.contextmanager
    def host(self, t: torch.Tensor) -> Generator[np.ndarray]:
        """`t` (float32) as a numpy array, valid inside the block."""
        nbytes = t.numel() * 4
        if t.device.type != "cuda" or nbytes < _PINNED_MIN or self.broken or not self.lock.acquire(blocking=False):
            yield t.float().cpu().numpy()
            return
        try:
            buf = self._grow(nbytes)
            if buf is None:
                yield t.float().cpu().numpy()
                return
            import torch  # noqa: PLC0415

            out = torch.from_numpy(buf[:nbytes].view(np.float32)).view(t.shape)
            out.copy_(t)  # synchronous, as .cpu()
            yield out.numpy()
        finally:
            self.lock.release()


_STAGING = _Staging()


@cache
def _pool() -> ThreadPoolExecutor:  # one pool for the process: starting threads per call costs ~0.6 ms
    return ThreadPoolExecutor(min(8, os.cpu_count() or 1), thread_name_prefix="fastforks-tiff")


def _compress(data: bytes) -> bytes:
    return _deflate(data, 1)  # level 1 is as small as level 6 on float maps and much faster


def _round_mantissa(a: np.ndarray, bits: int) -> np.ndarray:
    """Round finite float32 values to `bits` mantissa bits (nearest; relative error <= 2**-(bits+1) for normal floats).

    A value whose rounding would overflow to inf or flush to 0 (float32 max, subnormals) keeps its exact value.
    """
    if not 1 <= bits <= _MANTISSA:
        msg = f"bits must be in 1..{_MANTISSA}, got {bits}"
        raise InputError(msg)
    drop = _MANTISSA - bits
    if drop == 0:
        return a
    u = a.view(np.uint32)
    rounded = ((u + np.uint32(1 << (drop - 1))) & np.uint32(~((1 << drop) - 1) & 0xFFFFFFFF)).view(np.float32)
    keep = ~np.isfinite(a) | ~np.isfinite(rounded) | ((rounded == 0) & (a != 0))  # NaN / inf; overflow; subnormals
    return np.where(keep, a, rounded)


def _strip(rows: np.ndarray) -> bytes:
    """Predictor 3 (byte planes, then byte-wise horizontal differences) and deflate of one strip."""
    h, w = rows.shape
    planes = rows.astype(">f4").view(np.uint8).reshape(h, w, 4).transpose(0, 2, 1).reshape(h, 4 * w)
    diff = planes.copy()
    diff[:, 1:] -= planes[:, :-1]
    return _compress(diff.tobytes())


def write_tiff(path: str | Path, a: np.ndarray, bits: int | None = None, description: str | None = None) -> None:
    """Write a `[H, W]` float32 map as a classic little-endian TIFF: deflate + floating-point predictor, 64-row strips.

    Strips are compressed in parallel. Any TIFF reader that supports predictor 3 opens the file (tifffile, OpenCV,
    Pillow, GDAL). With `bits` the float32 mantissa is first rounded to that many bits (round to nearest, relative
    error at most `2**-(bits+1)`; `bits=10` is below 0.05 %), which compresses 7-10x instead of ~1.9x.
    `description` becomes the ImageDescription tag (ASCII; other characters are replaced by `?`).
    """
    a = np.ascontiguousarray(a, dtype=np.float32)
    if bits is not None:
        a = _round_mantissa(a, bits)
    h, w = a.shape
    strips = list(_pool().map(lambda i: _strip(a[i : i + _ROWS]), range(0, h, _ROWS)))
    n = len(strips)
    text = description.encode("ascii", "replace").ljust(8) + b"\x00" if description else b""  # > 4 bytes: not inline
    pad = b"\x00" * (len(text) % 2)  # the value area is word-aligned
    ntags = _TAGS + bool(text)
    table = 8 + 2 + ntags * 12 + 4 + len(text + pad)  # description, offset table, byte-count table, then the strips
    offsets = np.cumsum([table + 8 * n] + [len(s) for s in strips[:-1]]).tolist()
    sizes = [len(s) for s in strips]
    if n == 1:  # a single strip stores its offset / size inline
        offsets_at, sizes_at = offsets[0], sizes[0]
    else:
        offsets_at, sizes_at = table, table + 4 * n
    tags = [  # (tag, type, count, value)
        (256, _LONG, 1, w),
        (257, _LONG, 1, h),
        (258, _SHORT, 1, 32),
        (259, _SHORT, 1, 8),
        (262, _SHORT, 1, 1),
        *([(_DESCRIPTION, _ASCII, len(text), 8 + 2 + ntags * 12 + 4)] if text else []),
        (273, _LONG, n, offsets_at),
        (277, _SHORT, 1, 1),
        (278, _LONG, 1, _ROWS),
        (279, _LONG, n, sizes_at),
        (317, _SHORT, 1, 3),
        (339, _SHORT, 1, 3),
    ]
    out = [b"II*\x00", struct.pack("<IH", 8, ntags)]
    for tag, kind, count, value in tags:
        out.append(
            struct.pack("<HHII", tag, kind, count, value)
            if kind in {_LONG, _ASCII}
            else struct.pack("<HHIHH", tag, kind, count, value, 0)
        )
    out += [struct.pack("<I", 0), text, pad, struct.pack(f"<{n}I", *offsets), struct.pack(f"<{n}I", *sizes), *strips]
    with Path(path).open("wb") as f:
        f.writelines(out)


def atomic_write(path: str | Path, write: Callable[[Path], None]) -> None:
    """`write(tmp)` to a hidden file next to `path`, then rename it to `path`: readers never see a partial file.

    The temporary name keeps the suffix (the writers pick the format by it) and is the same on every run, so a run
    killed mid-write leaves at most one stale hidden file per output, which the next write of that output replaces.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.stem}.fastforks-tmp{path.suffix}")
    try:
        write(tmp)
        tmp.replace(path)  # atomic on POSIX and Windows (same directory)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_map(
    path: str | Path, a: torch.Tensor | np.ndarray, bits: int | None = None, description: str | None = None
) -> None:
    """Write a `[H, W]` float map; the format follows the suffix (`MAP_SUFFIXES`).

    - `.tif` / `.tiff`: float32 TIFF, deflate + floating-point predictor (see `write_tiff`); `bits` rounds the mantissa,
      `description` is stored as its ImageDescription tag (other formats ignore it).
    - `.npy`: float32; integer and bool maps keep their dtype (label maps stay exact).
    - `.pfm`: float32 (Middlebury / SceneFlow convention, bottom row first).
    - `.png`: integer and bool maps only, exact: a bool mask as a 1-bit black / white PNG, an integer map (labels) in
      0..255 as 8-bit, in 0..65535 as 16-bit. Float maps are refused (a stereo disparity `.png` is KITTI's format:
      `fastforks.stereo.io.write_disparity`).

    Raises:
        InputError: Unknown suffix, not a 2-D map, a `.png` of a float map or of values outside 0..65535, or `bits`
            outside 1..23.
    """
    if not isinstance(a, np.ndarray):
        if not a.is_floating_point():  # labels / masks keep their dtype (no float staging)
            write_map(path, a.detach().cpu().numpy(), bits, description)
            return
        with _STAGING.host(a.detach()) as host:
            write_map(path, host, bits, description)
        return
    path = Path(path)
    suffix = path.suffix.lower()
    exact = suffix in {".npy", ".png"} and np.asarray(a).dtype.kind in "biu"  # labels / masks: no float32 round trip
    a = np.asarray(a) if exact else np.asarray(a, dtype=np.float32)
    if a.ndim != 2:  # noqa: PLR2004 - [H, W]
        msg = f"a map must be [H, W], got shape {a.shape}"
        raise InputError(msg)
    if suffix in {".tif", ".tiff"}:
        write_tiff(path, a, bits, description)
    elif suffix == ".npy":
        with path.open("wb") as f:  # a file object keeps the name (np.save(path) appends ".npy" to "map.NPY")
            np.save(f, a)
    elif suffix == ".pfm":
        scale = -1.0 if sys.byteorder == "little" else 1.0
        with path.open("wb") as f:
            f.write(f"Pf\n{a.shape[1]} {a.shape[0]}\n{scale}\n".encode())
            f.write(np.ascontiguousarray(a[::-1]).tobytes())
    elif suffix == ".png":
        _write_png(path, a)
    else:
        msg = f"unknown map format {suffix!r} (use {', '.join(MAP_SUFFIXES)})"
        raise InputError(msg)


MAP_SUFFIXES = (".tif", ".tiff", ".npy", ".pfm", ".png")
"""The suffixes `write_map` writes."""

_U8, _U16 = 255, 65535


def _write_png(path: Path, a: np.ndarray) -> None:
    """A bool mask as a 1-bit PNG, an integer map as an 8-bit (0..255) or 16-bit (0..65535) PNG; exact."""
    from PIL import Image  # noqa: PLC0415 - only for this format

    if a.dtype.kind == "b":
        Image.fromarray(a).save(path)  # mode "1": reads back as bool
        return
    if a.dtype.kind not in "iu":
        msg = (
            "a .png holds integer or bool maps (labels, masks); write a float map as "
            f"{', '.join(s for s in MAP_SUFFIXES if s != '.png')}"
        )
        raise InputError(msg)
    lo, hi = (int(a.min()), int(a.max())) if a.size else (0, 0)
    if lo < 0 or hi > _U16:
        msg = f"a .png holds values in 0..{_U16}, the map has {lo}..{hi}; use .npy"
        raise InputError(msg)
    Image.fromarray(a.astype(np.uint8 if hi <= _U8 else np.uint16)).save(path)


_PNG_SCALE = 256.0


def write_png_x256(path: str | Path, a: np.ndarray) -> None:
    """A float `[H, W]` map as a uint16 PNG, value x 256, 0 = invalid (KITTI disparity convention).

    Values above 255.996 are clipped, negative and non-finite values are written as 0.

    Raises:
        InputError: Not a 2-D map.
    """
    from PIL import Image  # noqa: PLC0415 - only for this format

    a = np.asarray(a, dtype=np.float32)
    if a.ndim != 2:  # noqa: PLR2004 - [H, W]
        msg = f"a map must be [H, W], got shape {a.shape}"
        raise InputError(msg)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    Image.fromarray(np.clip(a * _PNG_SCALE, 0, _U16).round().astype(np.uint16)).save(path)


def check_suffix(path: str | Path, accepted: tuple[str, ...], what: str) -> str:
    """The lower-case suffix of `path`; `InputError` naming the `accepted` ones if it is not one of them."""
    suffix = Path(path).suffix.lower()
    if suffix not in accepted:
        msg = f"{what}.save cannot write {suffix or 'a path without a suffix'!r}: use {', '.join(accepted)}"
        raise InputError(msg)
    return suffix
