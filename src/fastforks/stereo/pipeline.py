"""`StereoModel.run_many`: many stereo pairs through one model, with decoding, GPU work and saving overlapped."""

from __future__ import annotations

import dataclasses
import glob
import io
import os
import re
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image

from fastforks._core.errors import InputError
from fastforks._core.image import _as_tensor, is_batch, is_url
from fastforks._core.io import atomic_write, write_map
from fastforks._core.pipeline import Done, OnError, Pipeline, default_prefetch, default_workers
from fastforks.stereo.calibrated import (
    PairViews,
    Rig,
    Views,
    depth_of,
    load_rig,
    make_view,
    resolve_geometry,
    run_views,
    view_names,
)
from fastforks.stereo.geometry.rectify import Rectification
from fastforks.stereo.io import write_disparity

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from fastforks._core.image import ImageLike
    from fastforks._core.pipeline import Copies
    from fastforks.stereo._api import StereoModel
    from fastforks.stereo.calibrated import Geometry
    from fastforks.stereo.geometry.rig import RectifiedRig

Images: TypeAlias = "str | os.PathLike[str] | Iterable[ImageLike] | np.ndarray | torch.Tensor"
"""One side of `run_many`: a directory, a glob pattern, or an iterable of images (paths, numpy arrays, tensors)."""

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".ppm", ".pgm")
"""Files a directory given to `run_many` contributes (case-insensitive)."""
_MAGIC = re.compile(r"[*?[]")
_DIGITS = re.compile(r"(\d+)")
_FORMATS = (".tif", ".tiff", ".npy", ".pfm", ".png")
_BATCH = 4


@dataclasses.dataclass(frozen=True, eq=False)
class PairResult(PairViews):
    """One pair of `StereoModel.run_many`, yielded in input order.

    The maps are read as for a single call (`StereoPairResult`): `r.left` / `r.right` are `StereoView`s,
    `r.disparity`, `r.depth()`, `r.points()`, ... those of the main view `r.view`. They are numpy arrays (tensors on
    the model's device for tensor inputs); point clouds have no colours (no images are kept). With `save_to`, a
    skipped output or an error there are no maps: `r.left`, `r.right` and `r.disparity` are None.

    Attributes:
        index: Position of the pair in the input.
        left_path: Path of the left image, or None for an array.
        right_path: Path of the right image, or None for an array.
        path: The file written with `save_to` (also for an output that `skip_existing` found).
        error: The error of this pair with `on_error="skip"`.
        existing: The output already existed and `skip_existing` left it alone (nothing was computed).
        views: The views that were run (`"left"`, `"right"` or `"both"`).
        calibrated: Whether `rig` was given (`depth()` then works).
    """

    index: int
    left_path: Path | None
    right_path: Path | None
    path: Path | None = None
    error: Exception | None = None
    views: str = "left"
    calibrated: bool = False
    existing: bool = False


@dataclasses.dataclass(frozen=True)
class _Pair:
    index: int
    left: ImageLike
    right: ImageLike
    out: Path | None


def natural_key(text: str) -> tuple[list[int | str], str]:
    """Sort key that orders the digit runs numerically: `frame_2` before `frame_10`."""
    return [int(t) if t.isdigit() else t.casefold() for t in _DIGITS.split(text)], text


def _stem(img: str | os.PathLike[str]) -> str:
    """Output name of an image: the file stem (of the URL path for a URL)."""
    return Path(urlparse(img).path).stem or "image" if is_url(img) else Path(img).stem


def _glob_root(pattern: str) -> Path:
    """The directory part of `pattern` before its first wildcard."""
    parts = Path(pattern).parts
    fixed = parts[: next(i for i, p in enumerate(parts) if _MAGIC.search(p))]
    return Path(*fixed) if fixed else Path()


def _is_entry(p: Path) -> bool:
    """A file, or a symlink that does not resolve (a broken one is a failed pair, not silently dropped)."""
    return p.is_file() or (p.is_symlink() and not p.is_dir())


def check_output_dir(path: str | os.PathLike[str]) -> Path:
    """Create `path` if needed and check that files can be written there.

    Raises:
        InputError: `path` is a file, cannot be created, or is not writable.
    """
    out = Path(path)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        msg = f"save_to={os.fspath(path)!r} cannot be used as an output directory: {e.strerror or e}"
        raise InputError(msg) from e
    if not os.access(out, os.W_OK | os.X_OK):
        msg = f"save_to={os.fspath(path)!r} is not writable"
        raise InputError(msg)
    return out


def _files(spec: str | os.PathLike[str], what: str) -> list[tuple[Path, str]]:
    """`(file, output name)` of a directory or a glob pattern, natural order."""
    path = Path(spec)
    if path.is_dir():
        files = [
            p
            for p in path.iterdir()
            if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".") and _is_entry(p)
        ]
        if not files:
            msg = f"{what}: no images in directory {str(spec)!r} (looked for {', '.join(IMAGE_SUFFIXES)})"
            raise InputError(msg)
        root = path
    else:
        pattern = os.fspath(spec)
        files = [Path(p) for p in glob.glob(pattern, recursive=True) if _is_entry(Path(p))]  # noqa: PTH207 - ** support
        if not files:
            msg = f"{what}: {pattern!r} is not a directory and matches no file"
            raise InputError(msg)
        root = _glob_root(pattern) if _MAGIC.search(pattern) else path.parent
    files.sort(key=lambda p: natural_key(str(p)))
    return [(p, p.relative_to(root).with_suffix("").as_posix()) for p in files]


def _side(spec: Images, what: str) -> tuple[Iterable[tuple[ImageLike, str | None]], int | None, ImageLike | None]:
    """The images of one side with their output names, their count if known up front, and the first image if cheap."""
    if is_url(spec):  # one image given by URL (is_url narrows to str)
        return [(spec, _stem(spec))], 1, None
    if isinstance(spec, (str, os.PathLike)):
        files = _files(spec, what)
        return files, len(files), files[0][0]
    if isinstance(spec, (np.ndarray, torch.Tensor)):
        if spec.ndim != _BATCH:
            msg = f"{what} is one array of shape {tuple(spec.shape)}: pass a list of images (or a 4-D batch)"
            raise InputError(msg)
        return ((img, None) for img in spec), len(spec), spec[0] if len(spec) else None
    if isinstance(spec, (Image.Image, bytes, bytearray, io.BytesIO)):
        msg = f"{what} is one image: pass a list of images"
        raise InputError(msg)
    try:
        it = iter(spec)
    except TypeError:
        msg = f"{what} must be a directory, a glob pattern or a list of images, got {type(spec).__name__}"
        raise InputError(msg) from None
    count = len(spec) if hasattr(spec, "__len__") else None  # ty: ignore[invalid-argument-type]
    first = spec[0] if count and isinstance(spec, (list, tuple)) else None
    named = ((img, _stem(img) if isinstance(img, (str, os.PathLike)) else None) for img in it)
    return (list(named) if isinstance(spec, (list, tuple)) else named), count, first


def check_inputs(
    lefts: Images, rights: Images
) -> tuple[
    Iterable[tuple[ImageLike, str | None]], Iterable[tuple[ImageLike, str | None]], int | None, ImageLike | None
]:
    """Both sides (see `_side`) and the left count; raises `InputError` for missing paths or unequal / zero counts."""
    left, n_left, first = _side(lefts, "lefts")
    right, n_right, _ = _side(rights, "rights")
    if n_left is not None and n_right is not None and n_left != n_right:
        msg = f"{n_left} left images but {n_right} right images"
        raise InputError(msg)
    if n_left == 0 or n_right == 0:
        msg = "no image pairs (lefts and rights are empty)"
        raise InputError(msg)
    return left, right, n_left, first


def _outputs(save_to: Path, name: str, suffix: str, views: Sequence[str]) -> list[Path]:
    """The files one pair writes: `<name><suffix>` for the left view, `<name>_right<suffix>` for the right."""
    return [save_to / f"{name if v == 'left' else f'{name}_right'}{suffix}" for v in views]


def _reject_overwrite(outs: Iterable[Path], images: Iterable[ImageLike]) -> None:
    """Raise `InputError` if an output file is one of the input files (compared resolved and ignoring case)."""
    ins = {str(p.resolve()).casefold(): p for img in images if (p := _path(img)) is not None}
    for out in outs:
        if (hit := ins.get(str(out.resolve()).casefold())) is not None:
            msg = f"the output {out} would overwrite the input image {hit}: write to another directory or format"
            raise InputError(msg)


def _reject_overwrite_all(
    left: list[tuple[ImageLike, str | None]], right: object, save_to: Path, suffix: str, views: Sequence[str]
) -> None:
    """`_reject_overwrite` over every pair, before any write (when both sides are known up front)."""
    if isinstance(right, list):
        outs = (o for _, nm in left if nm is not None for o in _outputs(save_to, nm, suffix, views))
        _reject_overwrite(outs, [img for img, _ in left + right])


def _pairs(
    lefts: Images, rights: Images, save_to: Path | None, suffix: str, *, views: Sequence[str] = ("left",)
) -> tuple[Iterator[_Pair], ImageLike | None, int | None]:
    """The pairs (lazily), the first left image (to size the defaults) and the pair count if known."""
    left, right, n_left, first = check_inputs(lefts, rights)

    def claim(i: int, name: str | None, seen: set[str]) -> str:
        """The output name of pair `i`; raises if it (or its right view) is taken (`seen`)."""
        name = name if name is not None else f"{i:06d}"
        # casefold: names differing only in case are one file on macOS (APFS) and Windows (TM-5)
        files = [(name if v == "left" else f"{name}_right").casefold() for v in views]
        if any(f in seen for f in files):
            msg = (
                f"two left images would both be saved as {name}{suffix} (output names are compared ignoring "
                "case); use a directory or a glob pattern, or rename one"
            )
            raise InputError(msg)
        seen.update(files)
        return name

    if save_to is not None and isinstance(left, list):  # known up front: no file is written before a collision (TD-4)
        known: set[str] = set()
        for i, (_, name) in enumerate(left):
            claim(i, name, known)
        _reject_overwrite_all(left, right, save_to, suffix, views)

    def pairs() -> Iterator[_Pair]:
        seen: set[str] = set()
        n = 0
        for i, (lt, rt) in enumerate(zip_longest(left, right)):
            if lt is None or rt is None:
                more, side = ("left", "right") if rt is None else ("right", "left")
                msg = f"the {side} images ended after {i} but there are more {more} images"
                raise InputError(msg)
            (limg, name), (rimg, _) = lt, rt
            out = None
            if save_to is not None:
                stem = claim(i, name, seen)
                out = save_to / f"{stem}{suffix}"
                _reject_overwrite(_outputs(save_to, stem, suffix, views), [limg, rimg])
            n += 1
            yield _Pair(i, limg, rimg, out)
        if n == 0:
            msg = "no image pairs (lefts and rights are empty)"
            raise InputError(msg)

    return pairs(), first, n_left


def _peek(first: ImageLike | None) -> tuple[int, int, int] | None:
    """`(height, width, bytes of a pair in flight)` of the first image, from the file header if it is a path."""
    try:
        if isinstance(first, (str, os.PathLike)) and not is_url(first):  # a URL is not downloaded twice
            with Image.open(first) as im:
                (w, h), c = im.size, len(im.getbands())
            return h, w, h * w * (2 * c + 4)  # two decoded images and a float32 map
        if isinstance(first, (bytes, bytearray, io.BytesIO)):
            with Image.open(io.BytesIO(first) if not isinstance(first, io.BytesIO) else first) as im:
                (w, h), c = im.size, len(im.getbands())
            return h, w, h * w * (2 * c + 4)
        if isinstance(first, (np.ndarray, torch.Tensor)):
            t = _as_tensor(first)
            h, w = t.shape[-2:]
            return h, w, 2 * t.numel() * t.element_size() + h * w * 4
    except (OSError, InputError, ValueError):
        return None
    return None


def _name(img: ImageLike) -> str:
    return os.fsdecode(img) if isinstance(img, (str, os.PathLike)) else f"<{type(img).__name__}>"


def _path(img: ImageLike) -> Path | None:
    return Path(img) if isinstance(img, (str, os.PathLike)) and not is_url(img) else None


def _host(img: ImageLike, what: str, index: int) -> torch.Tensor:
    """`[1, C, H, W]` tensor of one image as the single call reads it (decoded in a reader thread)."""
    if is_batch(img):
        msg = f"{what} image {index} is a batch of shape {tuple(img.shape)}; run_many takes one image per pair"  # ty: ignore[unresolved-attribute]
        raise InputError(msg)
    return _as_tensor(img)


def _pack(maps: dict[tuple[str, str], torch.Tensor]) -> tuple[torch.Tensor, list[tuple[Any, ...]]]:
    """All maps of a pair as one flat byte tensor (8-byte aligned pieces) and where each one sits in it."""
    spec, offset = [], 0
    for key, t in maps.items():
        n = t.numel() * t.element_size()
        spec.append((key, offset, n, str(t.dtype).removeprefix("torch."), tuple(t.shape)))
        offset += -(-n // 8) * 8
    buf = torch.empty(offset, dtype=torch.uint8, device=next(iter(maps.values())).device)
    for (_, start, n, _, _), t in zip(spec, maps.values(), strict=True):
        buf[start : start + n].copy_(t.contiguous().reshape(-1).view(torch.uint8))
    return buf, spec


def _unpack(host: np.ndarray, spec: list[tuple[Any, ...]]) -> dict[tuple[str, str], np.ndarray]:
    """The maps of `_pack` from its downloaded bytes (views into `host`)."""
    return {key: host[start : start + n].view(dtype).reshape(shape) for key, start, n, dtype, shape in spec}


def _within(depth: np.ndarray, max_depth: float | None) -> np.ndarray:
    """`depth` with the values beyond `max_depth` set to NaN (the written map; `None`: unchanged)."""
    return depth if max_depth is None else np.where(depth > max_depth, np.float32("nan"), depth)


_MAX_BITS = 23  # float32 mantissa bits (`write_tiff`)


def run_many(  # noqa: C901, PLR0913, PLR0915 - the options of the method and its stage closures
    model: StereoModel,
    lefts: Images,
    rights: Images,
    *,
    rig: Rig | None = None,
    views: Views = "left",
    size: tuple[int, int] | None = None,
    focal: float | None = None,
    save_to: str | os.PathLike[str] | None = None,
    save_format: str = ".tif",
    bits: int | None = None,
    max_depth: float | None = None,
    skip_existing: bool = False,
    on_error: OnError = "raise",
    prefetch: int | None = None,
    readers: int | None = None,
    writers: int | None = None,
    warmup: bool = False,
    _overlap: bool = True,
) -> Pipeline[PairResult]:
    """See `StereoModel.run_many`."""
    if on_error not in ("raise", "skip"):
        msg = f"on_error must be 'raise' or 'skip', got {on_error!r}"
        raise InputError(msg)
    suffix = save_format if save_format.startswith(".") else f".{save_format}"
    suffix = suffix.lower()
    if suffix not in _FORMATS:
        msg = f"save_format must be one of {', '.join(_FORMATS)}, got {save_format!r}"
        raise InputError(msg)
    for name, count in (("readers", readers), ("writers", writers)):
        if count is not None and count < 0:
            msg = f"{name} must be >= 0 (0: no threads, the work runs inline), got {count}"
            raise InputError(msg)
    if bits is not None and not 1 <= bits <= _MAX_BITS:
        msg = f"bits must be in 1..{_MAX_BITS}, got {bits}"
        raise InputError(msg)
    if save_to is None and (bits is not None or skip_existing or suffix != ".tif"):
        msg = "bits=, skip_existing= and save_format= only apply when files are written: pass save_to="
        raise InputError(msg)
    view_list = view_names(views)
    if max_depth is not None and (save_to is None or rig is None):
        msg = "max_depth= sets the written depth beyond it to NaN: it needs rig= (depth maps) and save_to="
        raise InputError(msg)
    loaded = None if rig is None else load_rig(rig)  # a calibration file is read at once (errors early)
    if loaded is None and (size is not None or focal is not None):
        msg = "size= and focal= need a calibration: pass rig="
        raise InputError(msg)
    if suffix == ".png" and loaded is not None:
        msg = "save_format .png writes KITTI disparity; with rig= the output is depth: use .tif, .npy or .pfm"
        raise InputError(msg)
    out_dir = check_output_dir(save_to) if save_to is not None else None
    pairs, first, _ = _pairs(lefts, rights, out_dir, suffix, views=view_list)
    peek = _peek(first)
    if prefetch is None:
        prefetch = default_prefetch(peek[2]) if peek else 4
    if prefetch < 1:
        msg = f"prefetch must be at least 1, got {prefetch}"
        raise InputError(msg)
    device = model.device

    def targets(pair: _Pair) -> list[Path]:
        """The files of one pair, one per view (the right view as `<name>_right.<suffix>`)."""
        assert pair.out is not None  # noqa: S101 - only called with save_to
        return [
            pair.out if v == "left" else pair.out.with_name(f"{pair.out.stem}_right{pair.out.suffix}")
            for v in view_list
        ]

    def geometry_for(image_size: tuple[int, int]) -> Rectification | RectifiedRig | None:
        return None if loaded is None else resolve_geometry(model, loaded, image_size, size=size, focal=focal)

    def read(pair: _Pair) -> object:
        if skip_existing and pair.out is not None and all(p.is_file() for p in targets(pair)):  # not a dir (TM-7)
            return Done(
                PairResult(
                    pair.index,
                    _path(pair.left),
                    _path(pair.right),
                    path=targets(pair)[0],
                    views=views,
                    calibrated=loaded is not None,
                    existing=True,
                )
            )
        if pair.index == 0:  # the first result sooner: its two images decode in parallel (a loop decodes one by one)
            with ThreadPoolExecutor(1, thread_name_prefix="fastforks-read0") as pool:
                right = pool.submit(_host, pair.right, "right", pair.index)
                return pair, _host(pair.left, "left", pair.index), right.result()
        return pair, _host(pair.left, "left", pair.index), _host(pair.right, "right", pair.index)

    def run(data: tuple[_Pair, torch.Tensor, torch.Tensor], copies: Copies) -> tuple[_Pair, Any, Geometry | None]:
        pair, lt, rt = data
        lt, rt = copies.upload(lt), copies.upload(rt)
        geometry = geometry_for((lt.shape[-2], lt.shape[-1]))
        _, _, maps = run_views(model, lt, rt, geometry, views)  # the views run one after the other, sharing lt, rt
        if out_dir is not None:  # one map per view goes to a file: depth with a rig, else disparity
            outs = []
            for v in view_list:
                d = maps[v]["disparity"][0]
                outs.append(copies.download(d if geometry is None else cast("torch.Tensor", depth_of(geometry, d, v))))
            return pair, outs, geometry
        flat = {(v, k): t[0] for v in view_list for k, t in maps[v].items() if t is not None}
        if geometry is not None:
            for v in view_list:
                flat[v, "depth"] = cast("torch.Tensor", depth_of(geometry, flat[v, "disparity"], v))
        if isinstance(pair.left, torch.Tensor):
            return pair, flat, geometry
        packed, spec = _pack(flat)
        return pair, (copies.download(packed), spec), geometry  # one download per pair: the host slots are few

    def finish(payload: tuple[_Pair, Any, Geometry | None]) -> PairResult:
        pair, outs, geometry = payload
        left, right = _path(pair.left), _path(pair.right)
        if pair.out is None:
            flat = outs if isinstance(outs, dict) else _unpack(outs[0].take(), outs[1])
            by_view = {}
            for v in view_list:
                m = {k: t for (view, k), t in flat.items() if view == v}
                by_view[v] = make_view(v, m, geometry, device=device, depth=m.pop("depth", None))
            result = PairResult(pair.index, left, right, views=views, calibrated=loaded is not None)
            object.__setattr__(result, "_by_view", by_view)
            return result
        paths = targets(pair)
        if (bad := next((p for p in paths if p.is_dir()), None)) is not None:
            for out in outs:
                out.release()
            msg = f"the output {bad} is a directory: remove it or write to another directory"
            raise InputError(msg)
        write = write_disparity if suffix == ".png" else write_map
        try:
            for path, out in zip(paths, outs, strict=True):
                path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write(path, lambda tmp, out=out: write(tmp, _within(out.view(), max_depth), bits))
        finally:
            for out in outs:
                out.release()
        return PairResult(pair.index, left, right, path=paths[0], views=views, calibrated=loaded is not None)

    def on_skip(index: int, pair: _Pair, e: Exception) -> PairResult:
        return PairResult(
            index, _path(pair.left), _path(pair.right), error=e, views=views, calibrated=loaded is not None
        )

    def describe(index: int, pair: _Pair) -> str:
        return f"pair {index} ({_name(pair.left)}, {_name(pair.right)})"

    def setup() -> None:
        geometry = geometry_for(peek[:2]) if peek else None
        if isinstance(geometry, Rectification):
            geometry.prepare(device)  # overlaps the decoding of the first pairs
        if warmup:
            hw = geometry.size if isinstance(geometry, Rectification) else peek[:2] if peek else None
            if hw is not None:
                zeros = torch.zeros(1, 3, *hw, device=device)
                model(zeros, zeros)

    workers = default_workers()
    return Pipeline(
        pairs,
        read=read,
        run=run,
        finish=finish,
        on_skip=on_skip,
        describe=describe,
        setup=setup,
        device=device,
        readers=workers if readers is None else readers,
        writers=workers if writers is None else writers,
        prefetch=prefetch,
        on_error=on_error,
        overlap=_overlap,
    )


__all__ = ["IMAGE_SUFFIXES", "Images", "PairResult", "Rig", "natural_key", "run_many"]
