"""Disparity file output."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fastforks._core.io import write_map, write_png_x256

if TYPE_CHECKING:
    import torch


def write_disparity(path: str | Path, disparity: torch.Tensor | np.ndarray, bits: int | None = None) -> None:
    """Write an `[H, W]` disparity map; the format follows the suffix.

    - `.tif` / `.tiff`: float32 TIFF, deflate + floating-point predictor; `bits` rounds the mantissa first (see
      `StereoResult.save`).
    - `.npy`: float32.
    - `.pfm`: float32 (Middlebury / SceneFlow convention, bottom row first).
    - `.png`: uint16, disparity x 256, 0 = invalid (KITTI convention); values above 255.996 px are clipped,
      non-finite values are written as 0 (invalid).

    Args:
        path: Output file; the suffix selects the format.
        disparity: Disparity in pixels, `[H, W]`.
        bits: Mantissa bits kept in a TIFF (`None` = lossless); ignored by the other formats.

    Raises:
        InputError: Unknown suffix or not a 2-D map.
    """
    path = Path(path)
    if path.suffix.lower() != ".png":
        write_map(path, disparity, bits)
        return
    d = disparity.detach().float().cpu().numpy() if not isinstance(disparity, np.ndarray) else disparity
    write_png_x256(path, d)
