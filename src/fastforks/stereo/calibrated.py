"""Calibrated stereo in one call: rectify, run the model (left and/or right view), depth and point clouds.

`StereoModel.__call__(left, right, rig=..., views=...)` and `StereoModel.run_many(..., rig=..., views=...)` share the
functions here, so a pair gives the same numbers either way.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, TypeAlias, cast

import numpy as np
import torch

from fastforks._core.errors import InputError, OptionsError
from fastforks._core.image import checked_ranges
from fastforks._core.io import MAP_SUFFIXES, check_suffix
from fastforks._core.outputs import MapOutputs
from fastforks._core.save import CLOUD_SUFFIXES
from fastforks.stereo.geometry.calib import read_calib
from fastforks.stereo.geometry.ply import write_cloud
from fastforks.stereo.geometry.rectify import Rectification, rectify
from fastforks.stereo.geometry.rig import Frame, PointCloud, RectifiedRig, StereoRig, side_of
from fastforks.stereo.io import write_disparity

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Any

    from fastforks._core.model import Backend
    from fastforks.stereo._api import StereoModel

Views: TypeAlias = Literal["left", "right", "both"]
Rig: TypeAlias = "str | os.PathLike[str] | StereoRig | Rectification | RectifiedRig"
"""A calibration file (`read_calib`), a `StereoRig`, its `Rectification`, or a `RectifiedRig` (rectified images)."""
Geometry: TypeAlias = "Rectification | RectifiedRig"
_CACHE = 8  # rigs kept per model
_log = logging.getLogger(__name__)
_WARNED: set[tuple] = set()  # resize warnings already logged, by calibration values and image size
_CACHE_LOCK = threading.Lock()  # run_many workers resolve geometry concurrently


def view_names(views: str) -> tuple[str, ...]:
    """`("left",)`, `("right",)` or `("left", "right")` of a `views` argument."""
    if views == "both":
        return ("left", "right")
    if views in ("left", "right"):
        return (views,)
    msg = f"views must be 'left', 'right' or 'both', got {views!r}"
    raise InputError(msg)


def load_rig(rig: Rig) -> StereoRig | Rectification | RectifiedRig:
    """A calibration file read; anything else checked."""
    if isinstance(rig, (str, os.PathLike)):
        try:
            return read_calib(cast("str | Path", rig))
        except FileNotFoundError as err:
            msg = f"calibration file not found: {os.fspath(rig)}"
            raise InputError(msg) from err
    if isinstance(rig, (StereoRig, Rectification, RectifiedRig)):
        return rig
    msg = f"rig must be a calibration file, StereoRig, Rectification or RectifiedRig, got {type(rig).__name__}"
    raise InputError(msg)


def _stamp(path: str | os.PathLike[str]) -> tuple[int, int]:
    """Newest modification time and total size of a calibration file, or of all files inside a calibration directory."""
    try:
        if Path(path).is_dir():
            stats = [p.stat() for p in Path(path).rglob("*") if p.is_file()]
            return max((st.st_mtime_ns for st in stats), default=0), sum(st.st_size for st in stats)
        st = os.stat(path)  # noqa: PTH116
    except FileNotFoundError as err:
        msg = f"calibration file not found: {os.fspath(path)}"
        raise InputError(msg) from err
    return st.st_mtime_ns, st.st_size


def resolve_geometry(
    model: StereoModel,
    rig: Rig,
    image_size: tuple[int, int],
    *,
    size: tuple[int, int] | None = None,
    focal: float | None = None,
) -> Geometry:
    """The rectification (or rectified rig) of `rig` for images of `image_size = (height, width)`, cached per model.

    A `StereoRig` is rescaled to the image size (`StereoRig.resized`; same aspect ratio, see `_rescale_size`) and
    rectified with `rectify`'s defaults (the image size and the left camera's focal length; `size` and `focal`
    override them); a `RectifiedRig` with a size must match the images. The maps are built once per (calibration,
    size, device) and reused by every call.
    """
    cache: dict[Any, tuple[Geometry, object]] = model.__dict__.setdefault("_rig_cache", {})
    is_path = isinstance(rig, (str, os.PathLike))
    stamp = _stamp(rig) if is_path else None  # a changed file is read again
    key = (os.fspath(rig) if is_path else id(rig), stamp, image_size, size, focal)
    with _CACHE_LOCK:
        found = cache.get(key)
    if found is not None:
        return found[0]
    loaded = load_rig(rig)
    calibrated = loaded.left.size if isinstance(loaded, StereoRig) else getattr(loaded, "size", None)
    if not isinstance(loaded, (StereoRig, RectifiedRig)):
        calibrated = None
    if isinstance(loaded, StereoRig):
        calib_key = (loaded.left, loaded.right, np.asarray(loaded.R).tobytes(), np.asarray(loaded.T).tobytes())
    else:
        calib_key = (calibrated,)
    if isinstance(loaded, StereoRig):
        stereo = loaded if loaded.left.size == image_size else loaded.resized(_rescale_size(loaded, image_size))
        loaded = rectify(stereo, size=size, focal=focal)
    elif size is not None or focal is not None:
        msg = "size= and focal= apply to a StereoRig; a Rectification or RectifiedRig has its own"
        raise InputError(msg)
    elif isinstance(loaded, RectifiedRig) and loaded.size not in (None, image_size):
        (h, w), (rh, rw) = image_size, loaded.size
        if (
            abs(rh * w / rw - h) <= 1 and (h - w) * (rh - rw) >= 0
        ):  # an aspect-preserving resize (one pixel of rounding)
            loaded = loaded.resized(image_size)
        else:
            msg = (
                f"the images are {h}x{w} (height x width) but the RectifiedRig is {rh}x{rw}: a RectifiedRig can only be"
                " resized keeping the aspect ratio; for other sizes use a StereoRig (its fx / fy follow"
                " StereoRig.resized) or a calibration for the cropped size (cx, cy shifted by the crop offset)"
            )
            raise InputError(msg)
    warn_key = (calib_key, image_size)
    if calibrated not in (None, image_size) and warn_key not in _WARNED:  # once per calibration values and size
        _WARNED.add(warn_key)
        _log.warning(
            "images are %dx%d (height x width) but the calibration is %dx%d: assuming the images are a resize of the"
            " whole calibrated view and scaling the calibration (pass rig.resized(size) to state a resize; depth is"
            " wrong for crops: build the calibration for the cropped size, with cx, cy shifted by the crop offset)",
            *image_size,
            *calibrated,
        )
    with _CACHE_LOCK:
        while len(cache) >= _CACHE:
            cache.pop(next(iter(cache)))
        cache[key] = (loaded, rig)  # keeps `rig` alive, so its id in the key stays unique
    return loaded


def _rescale_size(rig: StereoRig, image_size: tuple[int, int]) -> tuple[int, int]:
    """`image_size` if the calibration of `rig` may be rescaled to it: same orientation and aspect ratio.

    A calibrated call scales the cameras to the image size, which is only right for resized images (the whole field
    of view). A changed aspect ratio (beyond one pixel of rounding) means cropped or padded images, which need other
    principal points; an anisotropic resize on purpose goes through an explicit `rig.resized(size)`.

    Raises:
        InputError: The images are rotated against the calibration, or their aspect ratio differs.
    """
    (h, w), (ch, cw) = image_size, rig.left.size
    if (h - w) * (ch - cw) < 0:
        msg = (
            f"the images are {h}x{w} (height x width) but the calibration is {ch}x{cw}: portrait and landscape are"
            " swapped (rotated images?)"
        )
        raise InputError(msg)
    if abs(ch * w / cw - h) > 1 and abs(cw * h / ch - w) > 1:
        msg = (
            f"the images are {h}x{w} (height x width) but the calibration is {ch}x{cw}: the aspect ratio differs, so"
            " the calibration cannot be rescaled to them. Resize, don't crop (or pad); for crops shift cx / cy of the"
            f" cameras and pass a StereoRig of the cropped size; for an anisotropic resize pass rig.resized({(h, w)})"
        )
        raise InputError(msg)
    return image_size


def run_views(
    model: StereoModel, left: torch.Tensor, right: torch.Tensor, geometry: Geometry | None, views: Views
) -> tuple[torch.Tensor, torch.Tensor, dict[str, dict[str, Any]]]:
    """Rectify (if needed) and run the model once per view, one after the other.

    `left` and `right` are `[B, C, H, W]` tensors on the model's device. The left view is `model(left, right)`; the
    right view is `model(flip(right), flip(left))` with every map flipped back, which is the left-view problem of
    the mirrored pair. Returns the (rectified) images and `{view: {"disparity", "confidence", **extras}}` of
    `[B, H, W]` maps.
    """
    if isinstance(geometry, Rectification):
        left, right = geometry.remap(left, right)
    maps: dict[str, dict[str, Any]] = {}
    for i, view in enumerate(view_names(views)):
        # the mirrored pair of "both" has the pixel values just checked by the left run
        with checked_ranges() if i else contextlib.nullcontext():
            res = model(left, right) if view == "left" else model(right.flip(-1), left.flip(-1))
        out = {"disparity": res.disparity, "confidence": res.confidence, **res.extras}
        if view == "right":
            out = {k: None if v is None else v.flip(-1) for k, v in out.items()}
        maps[view] = out
    return left, right, maps


def depth_of(
    geometry: Geometry | None, disparity: torch.Tensor, view: str = "left", *, euclidean: bool = False
) -> torch.Tensor | None:
    """Depth in the original image of `view` (NaN holes) from its rectified disparity; None without a rig.

    A `RectifiedRig` has no original images: its depth is the depth in the (already rectified) input.
    """
    if isinstance(geometry, Rectification):
        z = geometry.depth(disparity, euclidean=euclidean, view=view)  # ty: ignore[invalid-argument-type]
        return geometry.unrectify(z, euclidean=euclidean, view=view)  # ty: ignore[invalid-argument-type]
    if isinstance(geometry, RectifiedRig):
        return geometry.depth(disparity, euclidean=euclidean, view=view)  # ty: ignore[invalid-argument-type]
    return None


Map: TypeAlias = "torch.Tensor | np.ndarray"


Grid: TypeAlias = Literal["original", "rectified"]


def _array(value: Map) -> np.ndarray:
    """A read-only numpy array of a map: a view of a CPU tensor or numpy map (no copy), so it cannot go stale."""
    a = value if isinstance(value, np.ndarray) else value.detach().cpu().numpy()
    return _readonly(a)


def _readonly(a: np.ndarray) -> np.ndarray:
    if a.flags.writeable:
        a = a.view()
        a.flags.writeable = False
    return a


def _version(m: Map | None) -> int | None:
    """In-place edit counter of a tensor map.

    None for read-only numpy maps and inference tensors (model outputs), whose in-place edits outside inference mode
    raise.
    """
    if m is None or isinstance(m, np.ndarray) or m.is_inference():
        return None
    return m._version  # noqa: SLF001 - bumped by every in-place edit


@dataclasses.dataclass(frozen=True)
class StereoView(MapOutputs):
    """The maps of one camera of a calibrated or two-view stereo call: `result.left` and `result.right`.

    Two pixel grids: `disparity`, `confidence`, `extras` and `points()` are on this camera's *rectified* grid
    (`[H, W]` = `rectification.size`); `depth()` is on its *original* image grid (`[h, w]`, the input size).
    `valid()` is the mask of `depth()`, `valid(grid="rectified")` the one of `disparity` / `points()`. Without `rig`
    or with a `RectifiedRig` the two grids are the same (the input). Maps are tensors on the model's device from a
    call and from `run_many` with tensor inputs, read-only numpy arrays from `run_many` otherwise; a leading `B` for a
    batch call. `depth()` is cached (recomputed if `disparity` or the returned tensor was edited in place).

    Attributes:
        name: `"left"` or `"right"`.
        disparity: Disparity in pixels, `x_left - x_right >= 0`, at this camera's rectified pixels.
        confidence: Confidence in `[0, 1]` of the same shape, or None if the model has none.
        extras: Further model outputs by name, same shape.
        rectification: The `Rectification` or `RectifiedRig` used, or None without `rig`.
    """

    name: str
    disparity: Map
    confidence: Map | None
    extras: Mapping[str, Map]
    rectification: Geometry | None

    _save_fields = ("depth", "disparity")
    _save_suffixes = MAP_SUFFIXES + CLOUD_SUFFIXES

    def depth(self, *, euclidean: bool = False) -> Map:
        """Depth `[h, w]`, NaN where unknown, in the unit of the calibration (metres from `read_calib`).

        With a `Rectification` this is the depth in the *original* image of this camera (along its optical axis, or
        with `euclidean=True` the distance along each pixel's ray); with a `RectifiedRig` it is the depth in the
        (already rectified) input. Same kind as `disparity` (numpy: read-only); computed once and cached.

        Args:
            euclidean: The distance along each pixel's ray instead of the depth along the optical axis.

        Raises:
            InputError: No `rig` was given.
        """
        if self.rectification is None:
            msg = "depth needs a calibration: pass rig="
            raise InputError(msg)
        cache = self.__dict__.setdefault("_depths", {})
        hit = cache.get(euclidean)
        if hit is not None and hit[1] == _version(self.disparity) and hit[2] == _version(hit[0]):
            return hit[0]
        z = depth_of(self.rectification, self._disparity_tensor(), self.name, euclidean=euclidean)
        if isinstance(self.disparity, np.ndarray):  # run_many result: computed where the model runs, numpy out
            z = _array(cast("torch.Tensor", z))
        cache[euclidean] = (z, _version(self.disparity), _version(z))
        return cast("Map", z)

    def to(self, device: str | torch.device) -> Self:
        """The view on another device (same type): tensor maps, the colour image and the cached depth move.

        Moves tensors only: the maps of a numpy view (`run_many` with array inputs) stay NumPy arrays, but `depth()`
        and `points()` then run on `device`. Tensors already there are not copied.

        Args:
            device: Target, e.g. `"cpu"` or `"cuda:0"`.
        """
        return self._carry(torch.device(device))

    def cpu(self) -> Self:
        """The view on the host; shorthand for `.to("cpu")`."""
        return self._carry(torch.device("cpu"))

    def _carry(self, device: torch.device) -> Self:
        """`dataclasses.replace` with the maps moved and the non-field state (image, device, source, depth cache)."""
        new = self._moved(lambda t: t.to(device))
        image = self.__dict__.get("_image")
        object.__setattr__(new, "_image", None if image is None else image.to(device))
        object.__setattr__(new, "_device", device)
        if "_source" in self.__dict__:  # numpy maps: the tensor of the same memory stays on the host
            object.__setattr__(new, "_source", self.__dict__["_source"])
        depths = {}
        for euclidean, (old, v_disp, v_old) in self.__dict__.get("_depths", {}).items():
            if v_disp != _version(self.disparity) or v_old != _version(old):
                continue  # stale (edited in place): the moved view recomputes it
            z = old if isinstance(old, np.ndarray) else old.to(device)
            depths[euclidean] = (z, _version(new.disparity), _version(z))
        if depths:
            object.__setattr__(new, "_depths", depths)
        return new

    def _disparity_tensor(self) -> torch.Tensor:
        """The disparity as a tensor where the model runs (numpy maps: a tensor of the same memory, no copy)."""
        d = self.disparity
        return d if isinstance(d, torch.Tensor) else self.__dict__["_source"].to(self.__dict__["_device"])

    def valid(self, *, grid: Grid = "original") -> Map:
        """Bool mask of the pixels with a value, on one of the two grids (see the class docstring).

        Args:
            grid: `"original"` (default): finite `depth()`, `[h, w]` (finite disparity without `rig`).
                `"rectified"`: the mask of `disparity` and `points()`, `[H, W]`: a finite positive disparity at a
                rectified pixel with a source pixel, as `points().valid`. The same without `rig` or with a
                `RectifiedRig`.

        Raises:
            InputError: An unknown `grid`.
        """
        if grid not in ("original", "rectified"):
            msg = f"grid must be 'original' (the grid of depth()) or 'rectified' (disparity, points()), got {grid!r}"
            raise InputError(msg)
        if grid == "rectified" and isinstance(self.rectification, Rectification):
            return self.points(frame="rectified", colors=False).valid
        z = self.depth() if self.rectification is not None else self.disparity
        return np.isfinite(z) if isinstance(z, np.ndarray) else torch.isfinite(z)

    def points(self, *, frame: Frame = "camera", colors: bool = True) -> PointCloud:
        """Per-pixel 3D points in this camera's own frame (the two views' clouds are not merged).

        Args:
            frame: `"camera"`: the original camera (what the calibration describes), or `"rectified"`: its
                rectified frame (the same for a `RectifiedRig`).
            colors: Return the rectified image of this view as `PointCloud.colors` `[H, W, C]` (None for `run_many`
                results, which keep no images).

        Returns:
            `PointCloud(points [H, W, 3], valid [H, W], colors)` with NaN points where invalid, on the rectified
            grid (index with `cloud.valid` or `valid(grid="rectified")`, not `valid()`); same kind as `disparity`.

        Raises:
            InputError: No `rig` was given, or an unknown `frame`.
        """
        if self.rectification is None:
            msg = "point clouds need a calibration: pass rig="
            raise InputError(msg)
        if frame not in ("camera", "rectified"):
            msg = f"frame must be 'camera' or 'rectified', got {frame!r}"
            raise InputError(msg)
        image = self.__dict__.get("_image")
        rgb = None if not colors or image is None else image.movedim(-3, -1)
        geometry = self.rectification
        d = self.disparity
        if isinstance(d, np.ndarray):  # the tensor of the read-only map, on the CPU (no copy); numpy out
            d = self.__dict__["_source"]
        if isinstance(geometry, Rectification):
            cloud = geometry.point_cloud(d, colors=rgb, frame=frame, view=self.name)  # ty: ignore[invalid-argument-type]
        else:
            cloud = geometry.point_cloud(d, colors=rgb, view=self.name)  # ty: ignore[invalid-argument-type]
        if isinstance(self.disparity, np.ndarray):
            cloud = PointCloud(cloud.points.numpy(), cloud.valid.numpy(), cloud.colors)  # ty: ignore[unresolved-attribute]
        return cloud

    def numpy(self) -> dict[str, Any]:
        """The maps as read-only arrays: `disparity`, `confidence`, the extras, and `depth` with a `rig`.

        Views of CPU maps (no copy), so read-only; `.copy()` one to edit it.
        """
        out = {"disparity": _array(self.disparity)}
        if self.confidence is not None:
            out["confidence"] = _array(self.confidence)
        out.update({k: _array(v) for k, v in self.extras.items()})
        if self.rectification is not None:
            out["depth"] = _array(self.depth())
        return out

    def _map(self, key: str | None) -> np.ndarray:
        """The map `key` as `[H, W]`; depth is only computed when it is asked for."""
        key = key or ("depth" if self.rectification is not None else "disparity")
        if key == "depth":
            value: Map | None = self.depth()
        else:
            value = {"disparity": self.disparity, "confidence": self.confidence}.get(key, self.extras.get(key))
        if value is None:
            msg = f"no output {key!r} to save (available: {sorted(self.numpy())})"
            raise InputError(msg)
        a = _array(value)
        return a[0] if a.ndim == 3 and a.shape[0] == 1 else a  # noqa: PLR2004 - [1, H, W]

    def save(
        self,
        path: str | Path,
        key: str | None = None,
        bits: int | None = None,
        *,
        max_depth: float | None = None,
        axes: str = "opencv",
    ) -> None:
        """Write a map or this view's point cloud; the suffix selects the format.

        - `.ply` / `.las` / `.laz`: `points()`, invalid points dropped, coloured by the rectified image; LAS / LAZ
          keep 0.1 mm. `max_depth=` (metres) drops points with z beyond it (default: every valid point). The header
          names units, axes and software.
        - `.tif` / `.tiff` / `.npy` / `.pfm`: one `[H, W]` map as in `MapOutputs.save`; `key` is `"depth"`,
          `"disparity"`, `"confidence"` or an extras name (default: depth with a `rig`, else disparity).
        - `.png`: KITTI uint16 of the disparity in this view's rectified frame (x 256, 0 = invalid), as
          `StereoResult.save`; a PNG cannot hold depth, so any other `key` raises `OptionsError`.

        Args:
            path: Output file.
            key: The map to write (not for a cloud).
            bits: TIFF mantissa bits kept (`None`: lossless).
            max_depth: Point clouds only: drop points whose z is larger than this (`None`: keep all).
            axes: Point clouds only: `"opencv"` (default: x right, y down, z forward), `"opengl"` / `"blender"` (RUB),
                `"ros"` (FLU), `"pytorch3d"` (LUF) or a 3-letter code (one of R/L, U/D, F/B each).

        Raises:
            OptionsError: A `.png` with a `key` other than `"disparity"`, or unknown `axes`.
            InputError: A missing map or an unknown suffix.
        """
        suffix = check_suffix(path, self._save_suffixes, type(self).__name__)
        if suffix in CLOUD_SUFFIXES:
            cloud = self.points()
            write_cloud(path, cloud.points, cloud.colors, max_depth=max_depth, axes=axes, frame="camera")
            return
        if max_depth is not None:
            msg = "max_depth= is for point clouds (.ply / .las / .laz); a map keeps every pixel"
            raise InputError(msg)
        if suffix == ".png":
            if key not in (None, "disparity"):
                msg = (
                    f"a .png is a KITTI disparity image and only holds key='disparity', not {key!r}; use .tif/.npy/.pfm"
                )
                raise OptionsError(msg)
            write_disparity(path, self._map("disparity"))
            return
        super().save(path, key, bits)


def make_view(  # noqa: PLR0913
    name: str,
    maps: Mapping[str, Any],
    geometry: Geometry | None,
    *,
    image: torch.Tensor | None = None,
    device: torch.device | None = None,
    depth: Map | None = None,
) -> StereoView:
    """A `StereoView` of `{"disparity", "confidence", **extras}`, with its image (colours) and a precomputed depth."""
    source = maps["disparity"]
    # numpy maps (run_many) are read-only, so no edit leaves `depth()` stale; `_source` is the tensor of the same memory
    maps = {k: _readonly(v) if isinstance(v, np.ndarray) else v for k, v in maps.items()}
    rest = {k: v for k, v in maps.items() if k not in ("disparity", "confidence")}
    view = StereoView(name, maps["disparity"], maps.get("confidence"), rest, geometry)
    object.__setattr__(view, "_image", image)  # not fields: numpy() and repr skip them
    object.__setattr__(view, "_device", device or torch.device("cpu"))
    if isinstance(source, np.ndarray):
        object.__setattr__(view, "_source", torch.from_numpy(source if source.flags.writeable else source.copy()))
    if depth is not None:
        depth = _readonly(depth) if isinstance(depth, np.ndarray) else depth
        object.__setattr__(view, "_depths", {False: (depth, _version(view.disparity), _version(depth))})
    return view


class PairViews:
    """The view access shared by `StereoPairResult` and `run_many`'s `PairResult`.

    `left` and `right` are the `StereoView`s; `disparity`, `confidence`, `extras`, `depth()`, `valid()`,
    `points()`, `save()` and `numpy()` are those of the main view `view`: the one that was run, or the left one
    for `views="both"`. Asking for a view that was not run raises `InputError`.
    """

    if TYPE_CHECKING:  # a field of the frozen subclasses

        @property
        def views(self) -> str:
            """`"left"`, `"right"` or `"both"`."""

    def _view(self, name: str) -> StereoView | None:
        by_view = self.__dict__.get("_by_view")
        if by_view is None:  # a run_many result without maps (written to a file, skipped or failed)
            return None
        if name not in by_view:
            msg = f"view {name!r} was not run (views={self.views!r}); run with views='both' or '{name}'"
            raise InputError(msg)
        return by_view[name]

    def _main(self) -> StereoView:
        found = self._view(self.view)
        if found is None:
            msg = "this result holds no maps (they were written with save_to, or the pair failed)"
            raise InputError(msg)
        return found

    @property
    def view(self) -> str:
        """The main view: `"right"` for `views="right"`, else `"left"`."""
        return "right" if self.views == "right" else "left"

    @property
    def left(self) -> StereoView | None:
        """The left camera's maps; raises `InputError` if the left view was not run (None: no maps kept)."""
        return self._view("left")

    @property
    def right(self) -> StereoView | None:
        """The right camera's maps (from the mirrored pair); raises `InputError` if the right view was not run."""
        return self._view("right")

    @property
    def disparity(self) -> Map | None:
        """The main view's disparity (`StereoView.disparity`), None if no maps were kept."""
        found = self._view(self.view)
        return None if found is None else found.disparity

    @property
    def confidence(self) -> Map | None:
        """The main view's confidence, or None."""
        found = self._view(self.view)
        return None if found is None else found.confidence

    @property
    def extras(self) -> Mapping[str, Map]:
        """The main view's extras (empty if no maps were kept)."""
        found = self._view(self.view)
        return {} if found is None else found.extras

    def depth(self, *, euclidean: bool = False) -> Map:
        """The main view's depth (`StereoView.depth`)."""
        return self._main().depth(euclidean=euclidean)

    def valid(self, *, grid: Grid = "original") -> Map:
        """The main view's validity mask (`StereoView.valid`: `"original"` grid of `depth()`, or `"rectified"`)."""
        return self._main().valid(grid=grid)

    def points(self, *, frame: Frame = "camera", colors: bool = True) -> PointCloud:
        """The main view's point cloud (`StereoView.points`)."""
        return self._main().points(frame=frame, colors=colors)

    def to(self, device: str | torch.device) -> Self:
        """The result on another device (same type): every view moves as `StereoView.to`."""
        return self._moved_views(lambda v: v.to(device))

    def cpu(self) -> Self:
        """The result on the host (same type); shorthand for `.to("cpu")`."""
        return self._moved_views(lambda v: v.cpu())

    def _moved_views(self, move: Callable[[StereoView], StereoView]) -> Self:
        new = dataclasses.replace(cast("Any", self))
        by_view = self.__dict__.get("_by_view")
        if by_view is not None:
            object.__setattr__(new, "_by_view", {n: move(v) for n, v in by_view.items()})
        return new

    def numpy(self) -> dict[str, Any]:
        """The main view's maps as arrays (`StereoView.numpy`); `result.right.numpy()` for the other one."""
        return self._main().numpy()

    def save(
        self,
        path: str | Path,
        key: str | None = None,
        bits: int | None = None,
        *,
        max_depth: float | None = None,
        axes: str = "opencv",
    ) -> None:
        """Write the main view's map or point cloud (`StereoView.save`; `result.right.save` for the right one)."""
        self._main().save(path, key, bits, max_depth=max_depth, axes=axes)


@dataclasses.dataclass(frozen=True)
class StereoPairResult(PairViews):
    """Output of a calibrated (or two-view) stereo call: one `StereoView` per view that was run.

    ```python
    r = model(left, right, rig="calib.yaml", views="both")
    r.depth()  # the left camera's depth (the main view), as r.left.depth()
    r.right.disparity, r.right.points()  # the right camera's maps and cloud
    ```

    Attributes:
        rectification: The `Rectification` (or `RectifiedRig`) used, or None.
        views: `"left"`, `"right"` or `"both"`.
        model: Versioned model id, e.g. `"s2m2-s@1"`.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults.
    """

    rectification: Geometry | None
    views: str
    model: str
    backend: Backend
    options: Mapping[str, Any] = dataclasses.field(default_factory=dict)


def build_result(
    model: StereoModel,
    geometry: Geometry | None,
    images: tuple[torch.Tensor, torch.Tensor],
    maps: dict[str, dict[str, Any]],
    views: Views,
) -> StereoPairResult:
    """The `StereoPairResult` of `run_views` outputs (maps and images already without a batch axis if needed)."""
    result = StereoPairResult(geometry, views, model.variant.ref, model.backend, model.options)
    by_view = {v: make_view(v, m, geometry, image=images[side_of(v)], device=model.device) for v, m in maps.items()}
    object.__setattr__(result, "_by_view", by_view)
    return result
