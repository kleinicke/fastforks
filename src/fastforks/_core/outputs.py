"""`.numpy()`, `.to()` and the NumPy containers of the typed results.

A result's outputs are torch tensors or NumPy arrays (`fastforks.set_outputs`, `arrays`); every method here takes both.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

import numpy as np

from fastforks._core import arrays
from fastforks._core.errors import InputError
from fastforks._core.io import MAP_SUFFIXES, atomic_write, check_suffix, write_map
from fastforks._core.lazy import after_import

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

_array = arrays.to_numpy  # numpy has no bfloat16: those become float32


def _is_out(v: object) -> bool:
    """Whether `v` is an output: a tensor or an array (numbers, not the 0-d metadata NumPy hands out)."""
    return arrays.is_tensor(v) or (isinstance(v, np.ndarray) and v.dtype.kind in "biufc")


def _short(v: object) -> str:  # noqa: PLR0911 - one return per kind of value
    """Tensors as `Tensor[shape dtype device]`, containers of them recursively, anything else as `repr`."""
    if arrays.is_tensor(v):
        return f"Tensor[{', '.join(map(str, v.shape))}; {str(v.dtype).removeprefix('torch.')}; {v.device}]"
    if type(v).__module__ == "numpy" and hasattr(v, "shape") and getattr(v, "ndim", 0) > 0:
        return f"ndarray[{', '.join(map(str, v.shape))}; {v.dtype}]"  # e.g. `VolumeMeta.affine`  # ty: ignore[invalid-argument-type, unresolved-attribute]
    if hasattr(v, "keys") and type(v).__module__.startswith("nibabel"):
        return f"<{type(v).__name__}>"  # a NIfTI header
    if isinstance(v, tuple | list) and any(_is_out(x) for x in v):
        return f"({', '.join(_short(x) for x in v)})"
    if isinstance(v, Mapping) and any(_is_out(x) for x in v.values()):
        return "{" + ", ".join(f"{k!r}: {_short(x)}" for k, x in v.items()) + "}"
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return short_repr(v)
    return repr(v)


def short_repr(self: object) -> str:
    """`__repr__` of the results: the shapes of the tensors instead of their values; None fields are left out."""
    values = ((f.name, getattr(self, f.name)) for f in dataclasses.fields(cast("Any", self)) if f.repr)
    fields = ", ".join(f"{name}={_short(v)}" for name, v in values if v is not None)
    return f"{type(self).__name__}({fields})"


class NumpyOutputs:
    """Mixin of the result dataclasses: their tensors as NumPy arrays."""

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        """Allow `torch.load(path)` (`weights_only=True`) of a `torch.save`d result: its class holds only tensors.

        (Registered once torch is imported: defining a result class does not import it.)
        """
        super().__init_subclass__(**kwargs)
        after_import("torch", lambda: sys.modules["torch"].serialization.add_safe_globals([cls]))
        if "__repr__" not in cls.__dict__:  # in the class dict, so the dataclass decorator keeps it
            cls.__repr__ = short_repr  # type: ignore[assignment, method-assign]

    def numpy(self) -> dict[str, Any]:
        """The tensor outputs as NumPy arrays by field name.

        Fields that are `None` are left out, `extras` are listed by their own names, and a tuple of tensors (e.g.
        `hidden_states`, entries may be `None`) becomes a tuple of arrays. Tensors move to the CPU; bfloat16 becomes
        float32 because NumPy has no bfloat16. Metadata (`model`, `backend`, `options`, `class_names`) is not included.

        The arrays of CPU tensors share their memory (no copy) and are writable: editing one in place also changes
        the result's tensor. `.copy()` an array before editing it if the result is still used. A result that holds
        NumPy arrays already (`outputs="numpy"`) gives those arrays.

        Returns:
            Array per output, e.g. `{"disparity": ..., "confidence": ..., "occlusion": ...}`.
        """
        return {
            k: tuple(None if t is None else _array(t) for t in v) if isinstance(v, tuple) else _array(v)
            for k, v in self._tensors().items()
        }

    def to(self, device: str | torch.device) -> Self:
        """The result on another device (same type, so `save`, `top`, ... keep working), or as NumPy arrays.

        Every tensor field moves, including tuples of tensors (entries may be `None`), tensors in `extras` and the
        tensors of dataclass fields (`MultiViewResult.gaussians`); other fields are unchanged. Like `torch.Tensor.to`,
        tensors already on `device` are not copied. NumPy outputs (`outputs="numpy"`) become tensors on `device` (on
        the CPU they share the arrays' memory).

        Args:
            device: Target, e.g. `"cpu"`, `"cuda:0"` or a `torch.device`; `"numpy"` gives NumPy arrays instead (as
                `outputs="numpy"` would have).

        Returns:
            A new result.
        """
        if isinstance(device, str) and device == arrays.HOST.type:
            return self._converted("numpy")
        return self._converted("torch", device)

    def cpu(self) -> Self:
        """The result on the host (same type); `.to("cpu")` for tensors, NumPy outputs stay arrays."""
        return self._moved(lambda t: t if isinstance(t, np.ndarray) else t.cpu())

    def _converted(self, kind: arrays.Outputs, device: Any = None) -> Self:  # noqa: ANN401 - torch.device or str
        """The outputs as `kind` (`arrays.convert`): tensors on `device` (`None`: where they are), or host arrays."""
        if kind == "numpy":
            return self._moved(arrays.to_numpy)
        return self._moved(
            lambda t: arrays.to_torch(t, device) if isinstance(t, np.ndarray) else t if device is None else t.to(device)
        )

    def _moved(self, move: Callable[[Any], Any]) -> Self:
        def one(v: object) -> object:  # tensor, tuple of tensors/None, dataclass of tensors, else unchanged
            if _is_out(v):
                return move(v)
            if isinstance(v, tuple) and v and all(x is None or _is_out(x) for x in v):
                return tuple(None if x is None else move(x) for x in v)
            if dataclasses.is_dataclass(v) and not isinstance(v, type):  # e.g. `MultiViewResult.gaussians`
                sub = {f.name: getattr(v, f.name) for f in dataclasses.fields(v)}
                if any(arrays.is_tensor(x) for x in sub.values()):  # not metadata (`VolumeMeta.affine`)
                    return dataclasses.replace(v, **{k: one(x) for k, x in sub.items()})
            return v

        changes = {}
        for f in dataclasses.fields(cast("Any", self)):
            value = getattr(self, f.name)
            if f.name == "extras" and isinstance(value, Mapping):
                changes[f.name] = {k: one(x) for k, x in value.items()}
            else:
                changes[f.name] = one(value)
        new = dataclasses.replace(cast("Any", self), **changes)
        image = self.__dict__.get("_image")  # the input image kept for point-cloud colours (not a field)
        if image is not None:
            object.__setattr__(new, "_image", move(image))
        return new

    def _tensors(self) -> dict[str, Any]:
        """The outputs by name (tensors / arrays or tuples of them), not yet moved to the CPU."""
        out: dict[str, Any] = {}
        for f in dataclasses.fields(cast("Any", self)):
            value = getattr(self, f.name)
            if _is_out(value) or (isinstance(value, tuple) and value and all(v is None or _is_out(v) for v in value)):
                out[f.name] = value
            elif f.name == "extras":
                out.update({k: v for k, v in value.items() if _is_out(v)})
        return out


class MapOutputs(NumpyOutputs):
    """Mixin of the results with a primary dense float map: adds `save`."""

    _save_fields: ClassVar[tuple[str, ...]]  # fields `save` writes by default; the first one that is set wins
    _save_suffixes: ClassVar[tuple[str, ...]] = MAP_SUFFIXES  # what `save` writes (named in its errors)

    def _map(self, key: str | None) -> np.ndarray:
        tensors = self._tensors()
        key = key or next((k for k in self._save_fields if k in tensors), None)
        t = tensors.get(key)
        if not _is_out(t):
            msg = f"no output {key or self._save_fields[0]!r} to save (available: {sorted(tensors)})"
            raise InputError(msg)
        a = _array(t)  # only this map goes to the CPU
        while a.ndim > 2 and a.shape[0] == 1:  # noqa: PLR2004 - [1, ..., H, W] -> [H, W]
            a = a[0]
        if a.ndim != 2:  # noqa: PLR2004 - [H, W]
            msg = (
                f"save writes one map, which must be [H, W]; {key!r} has shape {list(a.shape)}: select one and "
                f"write it with fastforks.save(path, result.{key}[...])"
            )
            raise InputError(msg)
        return a

    def _description(self, key: str | None) -> str:
        """Provenance text of a saved TIFF (its ImageDescription tag): model, map, what it means."""
        tensors = self._tensors()
        name = key or next((k for k in self._save_fields if k in tensors), self._save_fields[0])
        kind = getattr(self, "kind", None)  # depth results: metric, affine-invariant inverse, ...
        return f"fastforks {getattr(self, 'model', '?')}: {name}" + (
            " (pixels)" if name == "disparity" else f", kind {kind}" if kind else ""
        )

    def save(self, path: str | Path, key: str | None = None, bits: int | None = None) -> None:
        """Write one `[H, W]` float map to a file; the suffix selects the format.

        - `.tif` / `.tiff`: float32 TIFF with deflate and the floating-point predictor (about 1.9x smaller than raw,
          bit-exact; opens in tifffile, OpenCV, Pillow, GDAL).
        - `.npy`: float32 (integer and bool maps keep their dtype). `.pfm`: float32, bottom row first (Middlebury
          convention).
        - `.png`: integer and bool maps (labels, masks), exact: 1-bit for a bool mask, 8-bit for values in 0..255,
          16-bit in 0..65535; a float map is refused (except a stereo disparity: KITTI's 16-bit format).
        - `bits`: for TIFF, round the float32 mantissa to this many bits (round to nearest, relative error at most
          `2**-(bits+1)`), which makes the file 7-10x smaller than raw; `bits=10` keeps 0.05 %. `None` is lossless.

        Batches are not split: pass one image per call, or select one map yourself and write it with
        `fastforks.save(path, map)`. Leading dimensions of size 1 are dropped.

        Args:
            path: Output file.
            key: Output to write: a field name or an `extras` name (default: the primary map of the task).
            bits: Mantissa bits kept in a TIFF, 1..23, or `None`.

        Raises:
            InputError: The output is missing or not a single `[H, W]` map, the suffix is unknown or `bits` is invalid.
        """
        check_suffix(path, self._save_suffixes, type(self).__name__)
        write_map(Path(path), self._map(key), bits, self._description(key))


def save_images(
    path: str | os.PathLike[str],
    t: torch.Tensor | np.ndarray,
    what: str,
    suffixes: tuple[str, ...],
    write: Callable[[Path, np.ndarray], None],
) -> None:
    """`write` one `[C, H, W]` tensor to `path`, or each of a batch `[B, C, H, W]` to `path` with `{i}` = the index.

    Raises:
        InputError: Unknown suffix, or a batch and a path without `{i}`.
    """
    check_suffix(path, suffixes, what)
    a = _array(t)
    if a.ndim == 3:  # noqa: PLR2004 - one image
        atomic_write(path, lambda tmp: write(tmp, a))
        return
    if "{i}" not in os.fspath(path):
        msg = f"{what} holds a batch of {a.shape[0]}: put {{i}} in the path (e.g. 'out_{{i}}.png'), one file per image"
        raise InputError(msg)
    for i, item in enumerate(a):
        atomic_write(os.fspath(path).replace("{i}", str(i)), lambda tmp, item=item: write(tmp, item))
