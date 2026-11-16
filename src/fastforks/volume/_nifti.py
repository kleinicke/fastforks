"""NIfTI input and output (`nibabel`, the `nifti` extra): volumes in the axis order the models use, affine kept."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.errors import InputError, MissingDependencyError
from fastforks._core.outputs import short_repr

if TYPE_CHECKING:
    import torch

__all__ = ["VolumeMeta", "read_dicom", "read_nifti", "write_nifti"]


@dataclass(frozen=True)
class VolumeMeta:
    """Where a volume of `read_nifti` / `read_dicom` came from; `write_nifti` writes labels back onto that image's grid.

    Attributes:
        spacing: Voxel size in mm along the array's `D`, `H`, `W` axes.
        affine: The image's voxel-to-world affine (4 x 4, RAS mm).
        header: The NIfTI header (built from the series for DICOM).
        shape: The array's `(D, H, W)`.
    """

    __repr__ = short_repr

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
    name = os.fspath(path)
    if os.path.isdir(name) or name.lower().endswith((".dcm", ".dicom")):  # noqa: PTH112
        msg = f"{name}: read_nifti reads NIfTI files (.nii, .nii.gz); for a DICOM series folder use read_dicom(folder)"
        raise InputError(msg)
    return _canonical(nib.load(name), name)


def _canonical(img: Any, name: str) -> tuple[np.ndarray, VolumeMeta]:  # noqa: ANN401 - a nibabel image
    nib = _nib()
    can = nib.as_closest_canonical(img)
    data = np.asarray(can.get_fdata(dtype=np.float32))
    if data.ndim == 4:  # noqa: PLR2004 - [x, y, z, c]
        data = data.transpose(3, 2, 1, 0)
    elif data.ndim == 3:  # noqa: PLR2004
        data = data.transpose(2, 1, 0)
    else:
        msg = f"{name}: a {data.ndim}D image; volumes are 3D (or 4D with channels)"
        raise InputError(msg)
    zooms = can.header.get_zooms()[:3]
    spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
    shape = tuple(int(s) for s in data.shape[-3:])
    return np.ascontiguousarray(data), VolumeMeta(spacing, img.affine, img.header, shape)  # ty: ignore[invalid-argument-type]


def read_dicom(path: str | os.PathLike[str]) -> tuple[np.ndarray, VolumeMeta]:
    """A DICOM series (folder of slice files) as float32 `[D, H, W]` and its `VolumeMeta`, like `read_nifti`.

    Same convention as `read_nifti`: RAS-canonical axes `(z, y, x)`, so a DICOM CT needs no flipping. Slices are sorted
    by their position along the slice normal (`ImagePositionPatient`, not file names), pixel values are scaled to
    Hounsfield units (`RescaleSlope` / `RescaleIntercept`), spacing and affine come from the tags. Non-DICOM files in
    the folder are skipped; `write_nifti` writes labels back onto the series' grid.

    Raises:
        InputError: no readable slices, mixed slice sizes, or duplicate slice positions (several series in one folder).
    """
    nib = _nib()
    try:
        import pydicom  # noqa: PLC0415 - optional  # ty: ignore[unresolved-import]
    except ImportError as e:
        msg = "reading DICOM needs pydicom: pip install 'fastforks-core[dicom]'"
        raise MissingDependencyError(msg) from e
    root = os.fspath(path)
    files = sorted(str(f) for f in Path(root).iterdir()) if os.path.isdir(root) else [root]  # noqa: PTH112
    ds = []
    for f in files:
        try:
            d = pydicom.dcmread(f, force=False)
            if hasattr(d, "PixelData") and hasattr(d, "ImagePositionPatient"):
                ds.append(d)
        except Exception:  # noqa: BLE001, S112 - other files in the folder (DICOMDIR, readme, ...)
            continue
    if not ds:
        msg = f"{root}: no DICOM image slices (with PixelData and ImagePositionPatient) found; give the series folder"
        raise InputError(msg)
    iop = np.asarray(ds[0].ImageOrientationPatient, dtype=np.float64)
    row_dir, col_dir = iop[:3], iop[3:]  # direction of increasing column index / of increasing row index
    normal = np.cross(row_dir, col_dir)
    pos = np.array([np.asarray(d.ImagePositionPatient, dtype=np.float64) @ normal for d in ds])
    ds = [ds[i] for i in np.argsort(pos, kind="stable")]
    pos = np.sort(pos)
    if len({d.pixel_array.shape for d in ds}) != 1:
        msg = f"{root}: slices differ in size (several series in one folder?); give a folder with one series"
        raise InputError(msg)
    if len(ds) > 1 and float(np.diff(pos).min()) < 1e-3:  # noqa: PLR2004
        msg = f"{root}: two slices share one position (several series in one folder?); give a folder with one series"
        raise InputError(msg)
    arr = np.stack(
        [
            d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1))
            + float(getattr(d, "RescaleIntercept", 0))
            for d in ds
        ]
    ).astype(np.float32)
    dr, dc = (float(v) for v in ds[0].PixelSpacing)  # row spacing (between rows), column spacing
    dz = float(np.diff(pos).mean()) if len(ds) > 1 else float(getattr(ds[0], "SliceThickness", 1.0))
    p0 = np.asarray(ds[0].ImagePositionPatient, dtype=np.float64)
    lps = np.eye(4)
    lps[:3, 0], lps[:3, 1], lps[:3, 2], lps[:3, 3] = row_dir * dc, col_dir * dr, normal * dz, p0
    affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps  # DICOM is LPS, NIfTI is RAS
    img = nib.Nifti1Image(np.ascontiguousarray(arr.transpose(2, 1, 0)), affine)  # [x=col, y=row, z=slice]
    img.header.set_zooms((dc, dr, dz))
    return _canonical(img, root)


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
