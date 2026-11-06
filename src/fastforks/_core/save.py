"""`fastforks.save(path, thing)`: one writer for results, maps and point clouds; the format follows the suffix."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastforks._core.errors import InputError

CLOUD_SUFFIXES = (".ply", ".las", ".laz")


def save(path: str | Path, thing: Any, **kwargs: Any) -> None:  # noqa: ANN401 - results, maps and clouds
    """Write `thing` to `path`; the suffix selects the format.

    - A result with a `save` method (stereo, depth, multi-view, segmentation and promptable segmentation results):
      `thing.save(path, **kwargs)`.
    - A `PointCloud` (or an array / tensor `[..., 3]` of points) to `.ply` (binary PLY) or `.las` / `.laz` (LAS 1.2,
      0.1 mm integer coordinates); invalid points are dropped, `colors=` is
      optional; `max_depth=` drops points with z beyond it (metres; default: all finite points); `axes=` writes the
      points in another convention (default `"opencv"`: x right, y down, z forward; `"opengl"`, `"blender"`, `"ros"`,
      `"pytorch3d"` or a 3-letter code such as `"RUB"`); the header names units, axes and software.
    - A `[H, W]` array or tensor to `.tif` / `.tiff` (lossless float32 deflate TIFF, `bits=` rounds the mantissa),
      `.npy` (float32; integer and bool maps keep their dtype), `.pfm` or `.png` (integer and bool maps only, exact:
      e.g. one detection or SAM 3 mask, `fastforks.save("m.png", result.masks[i])`).

    Results without a `save` method (detection, concept segmentation, classification, features) are refused: write
    their arrays with `numpy.save`, or one mask as above.

    Args:
        path: Output file.
        thing: What to write.
        **kwargs: Passed on (`key=`, `bits=` for results, also `result.right`; `bits=` for maps; `colors=`,
            `max_depth=`, `axes=` for point clouds; results: see their `save`).

    Raises:
        InputError: A format the object cannot be written as.
    """
    if hasattr(thing, "save"):
        thing.save(path, **kwargs)
        return
    path = Path(path)
    if path.suffix.lower() in CLOUD_SUFFIXES:
        from fastforks.stereo.geometry.ply import write_cloud  # noqa: PLC0415 - torch stays out of `import fastforks`

        points, colors = (thing.points, thing.colors) if hasattr(thing, "points") else (thing, None)
        colors = kwargs.pop("colors", colors)
        max_depth = kwargs.pop("max_depth", None)
        axes = kwargs.pop("axes", "opencv")
        _no_extra(kwargs, "a point cloud", "colors, max_depth, axes")
        write_cloud(path, points, colors, max_depth=max_depth, axes=axes)
        return
    from fastforks._core.io import write_map  # noqa: PLC0415

    if not hasattr(thing, "shape"):
        name = type(thing).__name__
        if hasattr(thing, "_tensors"):  # a result without a writer
            arrays = sorted(thing._tensors())  # noqa: SLF001 - names only, nothing is copied
            first = arrays[0] if arrays else "..."
            msg = (
                f"a {name} has no save(): write its arrays with numpy.save, e.g."
                f" np.save(path, result.numpy()[{first!r}]) (arrays: {', '.join(arrays)})"
            )
        else:
            msg = f"cannot save a {name}: pass a result with save(), a PointCloud or an [H, W] array"
        raise InputError(msg)
    bits = kwargs.pop("bits", None)
    _no_extra(kwargs, "a map", "bits")
    write_map(path, thing, bits)


def _no_extra(kwargs: dict[str, Any], what: str, allowed: str) -> None:
    if kwargs:
        msg = f"save of {what} takes no {', '.join(map(repr, sorted(kwargs)))} (only {allowed}=)"
        raise InputError(msg)
