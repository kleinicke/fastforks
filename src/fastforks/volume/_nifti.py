"""NIfTI input and output (`nibabel`, the `nifti` extra): volumes in the axis order the models use, affine kept."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.errors import InputError, MissingDependencyError

if TYPE_CHECKING:
    import torch

__all__ = ["VolumeMeta", "read_nifti", "write_nifti"]


@dataclass(frozen=True)
class VolumeMeta:
    """Where a volume of `read_nifti` came from; `write_nifti` writes labels back onto that image's grid.

    Attributes:
        spacing: Voxel size in mm along the array's `D`, `H`, `W` axes.
        affine: The file's voxel-to-world affine (4 x 4).
        header: The file's NIfTI header.
        shape: The array's `(D, H, W)`.
    """

    spacing: tuple[float, float, float]
    affine: np.ndarray
    header: Any
    shape: tuple[int, int, int]


def _nib() -> Any:  # noqa: ANN401 - the nibabel module
    try:
        import nibabel  # noqa: PLC0415 - optional  # ty: ignore[unresolved-import]
    except ImportError as e:
        msg = "reading and writing NIfTI files needs nibabel: pip install 'fastforks-core[nifti]'"
        raise MissingDependencyError(msg) from e
    return nibabel


def read_nifti(path: str | os.PathLike[str]) -> tuple[np.ndarray, VolumeMeta]:
    """A NIfTI volume as float32 `[D, H, W]` (`[C, D, H, W]` for 4D files) and its `VolumeMeta`.

    The array is the RAS-canonical image (`nibabel.as_closest_canonical`) with the axes reversed, the `(z, y, x)` order
    nnU-Net and TotalSegmentator give their networks; `meta.spacing` follows it. Intensities are scaled by the header
    (`get_fdata`).
    """
    nib = _nib()
    img = nib.load(os.fspath(path))
    can = nib.as_closest_canonical(img)
    data = np.asarray(can.get_fdata(dtype=np.float32))
    if data.ndim == 4:  # noqa: PLR2004 - [x, y, z, c]
        data = data.transpose(3, 2, 1, 0)
    elif data.ndim == 3:  # noqa: PLR2004
        data = data.transpose(2, 1, 0)
    else:
        msg = f"{path}: a {data.ndim}D image; volumes are 3D (or 4D with channels)"
        raise InputError(msg)
    zooms = can.header.get_zooms()[:3]
    spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
    shape = tuple(int(s) for s in data.shape[-3:])
    return np.ascontiguousarray(data), VolumeMeta(spacing, img.affine, img.header, shape)  # ty: ignore[invalid-argument-type]


def write_nifti(path: str | os.PathLike[str], labels: np.ndarray | torch.Tensor, meta: VolumeMeta) -> None:
    """Write labels `[D, H, W]` of a `read_nifti` volume as a NIfTI file on that file's grid (affine and header kept).

    Raises:
        InputError: `labels` is not on the grid of `meta`.
    """
    nib = _nib()
    arr = labels.detach().cpu().numpy() if hasattr(labels, "detach") else np.asarray(labels)  # ty: ignore[call-non-callable]
    if tuple(arr.shape) != meta.shape:
        msg = f"labels have shape {tuple(arr.shape)}, the volume of meta has {meta.shape}"
        raise InputError(msg)
    orn = nib.orientations
    back = orn.ornt_transform(orn.axcodes2ornt("RAS"), orn.io_orientation(meta.affine))
    data = orn.apply_orientation(arr.transpose(2, 1, 0), back)
    dtype = np.uint8 if arr.max(initial=0) <= 255 and arr.min(initial=0) >= 0 else np.int16  # noqa: PLR2004
    header = meta.header.copy()
    header.set_data_dtype(dtype)
    out = nib.Nifti1Image(np.ascontiguousarray(data.astype(dtype)), meta.affine, header)
    out.header.set_slope_inter(1, 0)
    nib.save(out, os.fspath(path))
