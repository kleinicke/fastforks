"""`.numpy()` for the typed results."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

import torch

from fastforks._core.errors import InputError
from fastforks._core.io import MAP_SUFFIXES, check_suffix, write_map

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np


def _array(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu()
    return (t.float() if t.dtype is torch.bfloat16 else t).numpy()  # numpy has no bfloat16


class NumpyOutputs:
    """Mixin of the result dataclasses: their tensors as NumPy arrays."""

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        """Allow `torch.load(path)` (`weights_only=True`) of a `torch.save`d result: its class holds only tensors."""
        super().__init_subclass__(**kwargs)
        torch.serialization.add_safe_globals([cls])

    def numpy(self) -> dict[str, Any]:
        """The tensor outputs as NumPy arrays by field name.

        Fields that are `None` are left out, `extras` are listed by their own names, and a tuple of tensors (e.g.
        `hidden_states`, entries may be `None`) becomes a tuple of arrays. Tensors move to the CPU; bfloat16 becomes
        float32 because NumPy has no bfloat16. Metadata (`model`, `backend`, `options`, `class_names`) is not included.

        The arrays of CPU tensors share their memory (no copy) and are writable: editing one in place also changes
        the result's tensor. `.copy()` an array before editing it if the result is still used.

        Returns:
            Array per output, e.g. `{"disparity": ..., "confidence": ..., "occlusion": ...}`.
        """
        return {
            k: tuple(None if t is None else _array(t) for t in v) if isinstance(v, tuple) else _array(v)
            for k, v in self._tensors().items()
        }

    def to(self, device: str | torch.device) -> Self:
        """The result on another device (same type, so `save`, `top`, ... keep working).

        Every tensor field moves, including tuples of tensors (entries may be `None`) and tensors in `extras`; other
        fields are unchanged. Like `torch.Tensor.to`, tensors already on `device` are not copied.

        Args:
            device: Target, e.g. `"cpu"`, `"cuda:0"` or a `torch.device`.

        Returns:
            A new result.
        """
        return self._moved(lambda t: t.to(device))

    def cpu(self) -> Self:
        """The result on the host (same type); shorthand for `.to("cpu")`."""
        return self._moved(lambda t: t.cpu())

    def _moved(self, move: Callable[[torch.Tensor], torch.Tensor]) -> Self:
        def one(v: object) -> object:  # tensor, tuple of tensors/None, or anything else (unchanged)
            if isinstance(v, torch.Tensor):
                return move(v)
            if isinstance(v, tuple) and v and all(isinstance(x, torch.Tensor | None) for x in v):
                return tuple(None if x is None else move(x) for x in v)
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
        """The tensor outputs by name (tensors or tuples of tensors), not yet moved to the CPU."""
        out: dict[str, Any] = {}
        for f in dataclasses.fields(cast("Any", self)):
            value = getattr(self, f.name)
            if isinstance(value, torch.Tensor) or (
                isinstance(value, tuple) and value and all(isinstance(v, torch.Tensor | None) for v in value)
            ):
                out[f.name] = value
            elif f.name == "extras":
                out.update({k: v for k, v in value.items() if isinstance(v, torch.Tensor)})
        return out


class MapOutputs(NumpyOutputs):
    """Mixin of the results with a primary dense float map: adds `save`."""

    _save_fields: ClassVar[tuple[str, ...]]  # fields `save` writes by default; the first one that is set wins
    _save_suffixes: ClassVar[tuple[str, ...]] = MAP_SUFFIXES  # what `save` writes (named in its errors)

    def _map(self, key: str | None) -> np.ndarray:
        tensors = self._tensors()
        key = key or next((k for k in self._save_fields if k in tensors), None)
        t = tensors.get(key)
        if not isinstance(t, torch.Tensor):
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
