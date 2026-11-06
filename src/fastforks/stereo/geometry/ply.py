"""Binary little-endian PLY writer for point clouds (`PointCloud` of a calibrated stereo call)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from fastforks._core.errors import InputError, MissingDependencyError, OptionsError
from fastforks._core.io import atomic_write

if TYPE_CHECKING:
    import os


AXES_PRESETS = {"opencv": "RDF", "opengl": "RUB", "blender": "RUB", "ros": "FLU", "pytorch3d": "LUF"}
"""Axis conventions of `axes=`: the directions of x, y, z (Right / Left, Up / Down, Forward / Back). Default: opencv."""
_AXIS_OF = {"R": (0, 1), "L": (0, -1), "D": (1, 1), "U": (1, -1), "F": (2, 1), "B": (2, -1)}  # RDF axis, sign


def axes_code(axes: str) -> str:
    """The 3-letter code of an `axes=` preset or code (one of R/L, one of U/D, one of F/B, in any order).

    Raises:
        OptionsError: Neither a preset nor a valid code.
    """
    code = AXES_PRESETS.get(axes.lower(), axes.upper()) if isinstance(axes, str) else ""
    if len(code) != len("RDF") or sorted(_AXIS_OF[c][0] for c in code if c in _AXIS_OF) != [0, 1, 2]:
        presets = ", ".join(f"{k} = {v}" for k, v in AXES_PRESETS.items())
        msg = (
            f"axes must be a preset ({presets}) or a 3-letter code with one of R/L, one of U/D and one of F/B "
            f"(the directions of x, y, z), got {axes!r}"
        )
        raise OptionsError(msg)
    return code


def _meta(code: str, frame: str | None, units: str) -> list[str]:
    """The header lines shared by the PLY comments and the LAS record: software, units, axes, frame, pixel grid."""
    from fastforks import __version__  # noqa: PLC0415 - lazy: `fastforks` imports this module

    lines = [f"fastforks {__version__}", f"units: {units}", f"axes: {code} (x, y, z point {', '.join(code)})"]
    return lines + ([f"frame: {frame}"] if frame else []) + ["pixel centres: integer coordinates (OpenCV)"]


def _finite(
    points: np.ndarray | torch.Tensor,
    colors: np.ndarray | torch.Tensor | None,
    code: str = "RDF",
    dtype: type = np.float32,
) -> tuple[np.ndarray, np.ndarray | None]:
    """The finite points `[N, 3]` as `dtype` in the axes `code`, and their uint8 colours `[N, 3]` (None: none)."""
    p = points.detach().cpu().numpy() if isinstance(points, torch.Tensor) else np.asarray(points)
    if p.shape[-1:] != (3,):
        msg = f"points must have shape [..., 3], got {p.shape}"
        raise InputError(msg)
    p = p.reshape(-1, 3).astype(dtype, copy=False)
    keep = np.isfinite(p).all(1)
    if code != "RDF":  # permute and flip the OpenCV axes (exact in float32)
        p = p[:, [_AXIS_OF[c][0] for c in code]] * np.array([_AXIS_OF[c][1] for c in code], dtype)
    rgb = None
    if colors is not None:
        c = colors.detach().cpu().numpy() if isinstance(colors, torch.Tensor) else np.asarray(colors)
        c = c.reshape(-1, c.shape[-1]) if c.ndim > 1 and c.shape[-1] in (1, 3) else c.reshape(-1, 1)
        if len(c) != len(p):
            msg = f"colors {colors.shape} do not match points {points.shape}"
            raise InputError(msg)
        if c.dtype != np.uint8:
            c = np.clip(np.rint(c.astype(np.float32) * 255), 0, 255).astype(np.uint8)
        rgb = np.broadcast_to(c, (len(c), 3))[keep]
    return p[keep], rgb


def write_ply(  # noqa: PLR0913
    path: str | os.PathLike[str],
    points: np.ndarray | torch.Tensor,
    colors: np.ndarray | torch.Tensor | None = None,
    *,
    axes: str = "opencv",
    frame: str | None = None,
    units: str = "metres",
) -> int:
    """Write the finite points of `points` `[..., 3]` as a binary PLY (float32 xyz, optional uint8 rgb).

    Args:
        path: Output file (written atomically).
        points: `[..., 3]` points; rows with a NaN or infinite coordinate are dropped (holes of a point cloud).
        colors: Optional `[..., 3]` or `[..., 1]` per-point colours matching `points` (uint8, or floats in
            `[0, 1]`).
        axes: Axes of the written points (`points` are OpenCV: x right, y down, z forward): `"opencv"` (default, RDF),
            `"opengl"` / `"blender"` (RUB), `"ros"` (FLU), `"pytorch3d"` (LUF) or any 3-letter code, e.g. `"RFU"`.
        frame: Coordinate frame named in the header (e.g. `"camera"`), or `None`.
        units: Unit named in the header.

    Returns:
        The number of points written. The header comments name the software version, units, axes, frame and pixel
        convention.

    Raises:
        InputError: Shapes that do not match.
        OptionsError: Unknown `axes`.
    """
    code = axes_code(axes)
    p, rgb = _finite(points, colors, code)
    dtype = [("xyz", "<f4", 3)] + ([("rgb", "u1", 3)] if rgb is not None else [])
    data = np.empty(len(p), dtype=dtype)
    data["xyz"] = p
    if rgb is not None:
        data["rgb"] = rgb
    header = ["ply", "format binary_little_endian 1.0", *(f"comment {m}" for m in _meta(code, frame, units))]
    header.append(f"element vertex {len(p)}")
    header += [f"property float {a}" for a in "xyz"]
    if rgb is not None:
        header += [f"property uchar {a}" for a in ("red", "green", "blue")]
    header.append("end_header\n")

    def write(tmp: Path) -> None:
        with Path(tmp).open("wb") as f:
            f.write("\n".join(header).encode("ascii"))
            f.write(data.tobytes())

    atomic_write(Path(path), write)
    return len(p)


def write_las(  # noqa: PLR0913
    path: str | os.PathLike[str],
    points: np.ndarray | torch.Tensor,
    colors: np.ndarray | torch.Tensor | None = None,
    *,
    axes: str = "opencv",
    frame: str | None = None,
    units: str = "metres",
) -> int:
    """Write the finite points as a LAS file, or a compressed LAZ file for the suffix `.laz`.

    The coordinates are stored as integers with a scale of 0.1 mm (in metres; the unit of the points) relative to the
    minimum corner, coarser only if the extent would overflow 32 bits. Colours are 16-bit (8-bit values times 257).

    Args:
        path: Output file (`.las` or `.laz`, written atomically).
        points: `[..., 3]` points; rows with a NaN or infinite coordinate are dropped.
        colors: Optional per-point colours as for `write_ply`.
        axes: As for `write_ply`.
        frame: As for `write_ply`.
        units: As for `write_ply`.

    Returns:
        The number of points written. System id "fastforks"; a VLR (user id "fastforks") holds the software version,
        units, axes, frame and pixel convention. A local frame: no CRS.

    Raises:
        InputError: Shapes that do not match.
        OptionsError: Unknown `axes`.
        MissingDependencyError: `laspy` (and `lazrs` for `.laz`) is not installed.
    """
    try:
        import laspy  # noqa: PLC0415 - optional
    except ImportError as err:
        msg = f"writing {Path(path).suffix} point clouds needs laspy (a fastforks dependency): pip install laspy lazrs"
        raise MissingDependencyError(msg) from err
    if Path(path).suffix.lower() == ".laz":
        try:
            import lazrs  # noqa: F401, PLC0415 - optional compression backend
        except ImportError as err:
            msg = "writing .laz point clouds needs the lazrs backend (no wheel for this platform): pip install lazrs"
            raise MissingDependencyError(msg) from err
    code = axes_code(axes)
    p, rgb = _finite(points, colors, code, np.float64)  # float64 input keeps its precision (UTM-sized coordinates)
    header = laspy.LasHeader(point_format=2 if rgb is not None else 0, version="1.2")
    meta = _meta(code, frame, units)
    header.system_identifier = "fastforks"
    header.generating_software = meta[0]
    header.vlrs.append(laspy.VLR("fastforks", 1, "units, axes, frame", "\n".join(meta).encode()))
    lo = p.min(0) if len(p) else np.zeros(3)
    extent = float(np.max(p.max(0) - lo)) if len(p) else 0.0
    header.offsets = lo
    header.scales = np.full(3, max(1e-4, extent / 2e9))
    las = laspy.LasData(header)
    las.x, las.y, las.z = p[:, 0], p[:, 1], p[:, 2]
    if rgb is not None:
        las.red, las.green, las.blue = (rgb[:, i].astype(np.uint16) * 257 for i in range(3))

    def write(tmp: Path) -> None:
        with tmp.open("wb") as f:
            las.write(f, do_compress=Path(path).suffix.lower() == ".laz")

    atomic_write(Path(path), write)
    return len(p)


def check_max_depth(max_depth: float | None) -> None:
    """Raise `OptionsError` for a NaN `max_depth` (it would keep every point silently)."""
    if max_depth is not None and math.isnan(max_depth):
        msg = "max_depth must be a number (metres) or None, got NaN"
        raise OptionsError(msg)


def limit_depth(points: np.ndarray | torch.Tensor, max_depth: float | None) -> np.ndarray | torch.Tensor:
    """`points` `[..., 3]` with the points beyond `max_depth` (their z) set to NaN, so the writers drop them.

    `None` returns `points` unchanged (the same object).
    """
    if max_depth is None:
        return points
    check_max_depth(max_depth)
    if isinstance(points, np.ndarray):
        return np.where((points[..., 2] > max_depth)[..., None], np.float32("nan"), points)
    return points.masked_fill((points[..., 2] > max_depth)[..., None], math.nan)


def write_cloud(  # noqa: PLR0913
    path: str | os.PathLike[str],
    points: np.ndarray | torch.Tensor,
    colors: np.ndarray | torch.Tensor | None = None,
    *,
    max_depth: float | None = None,
    axes: str = "opencv",
    frame: str | None = None,
    units: str = "metres",
) -> int:
    """`write_ply` for `.ply`, `write_las` for `.las` / `.laz`; the suffix selects the format.

    `max_depth` drops the points whose z (OpenCV axes, before `axes`) is larger (`limit_depth`); the default keeps
    every finite point. `axes`, `frame` and `units` as for `write_ply`.
    """
    axes_code(axes)  # before any work
    points = limit_depth(points, max_depth)
    writer = write_ply if Path(path).suffix.lower() == ".ply" else write_las
    return writer(path, points, colors, axes=axes, frame=frame, units=units)
