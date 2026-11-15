"""TotalSegmentator runners: TotalSegmentator 2.18.0's `nnUNet_predict_image` on the GPU.

The pipeline of `totalsegmentator.nnunet.nnUNet_predict_image` for the open tasks: the RAS-canonical image (the caller's
`[D, H, W]` from `fastforks.volume.read_nifti`) resampled to the task spacing as `change_spacing` does it (scipy
`zoom`, order 1, corner-aligned, truncated to int32), nnU-Net v2 (`fastforks.volume._nnunet_runner`, fold 0, step 0.8,
no mirroring) per part, the parts merged into one label map (later parts win), and an order-0 corner-aligned zoom back
to the input grid. Not reproduced: the split of > 512 x 512 x 900 voxel volumes into three z chunks (it bounds
upstream's CPU memory for the full logits, which are never held here), and the optional crops / statistics / extra
outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from fastforks.volume import _nnunet_runner as nn_runner
from fastforks.volume._nnunet_runner import weight_names

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = ["RUNNERS", "Params", "weight_names", "zoom_linear", "zoom_nearest"]


@dataclass(frozen=True)
class Params(nn_runner.Params):
    """`fastforks.volume._nnunet_runner.Params` + TotalSegmentator's resampling and class list.

    Attributes:
        spacing: TotalSegmentator's resampling to this isotropic spacing (mm) first, or `None` (nnU-Net's own).
        classes: Label file (one name per line, line k = label k) of the merged map, or `None`: the single part's.
    """

    spacing: float | None = None
    classes: str | None = None


def zoom_linear(x: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """`scipy.ndimage.zoom(x, order=1)` to `shape` (corner-aligned) in float64, truncated to integers like `int32`."""
    src = x.cpu() if x.device.type == "mps" else x  # MPS has no float64
    y = F.interpolate(src[None, None].double(), size=tuple(shape), mode="trilinear", align_corners=True)[0, 0]
    return y.trunc_().float().to(x.device)


def zoom_nearest(x: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """`scipy.ndimage.zoom(x, order=0)` to `shape`: index `floor(o * (n - 1) / (m - 1) + 0.5)` per axis."""
    for axis, m in enumerate(shape):
        n = x.shape[axis]
        if n == m:
            continue
        scale = (n - 1) / (m - 1) if m > 1 else 0.0
        idx = torch.floor(torch.arange(m, dtype=torch.float64) * scale + 0.5).long().clamp_(0, n - 1)
        x = x.index_select(axis, idx.to(x.device))
    return x


class Portable(nn_runner.Portable):
    """`fastforks.volume._nnunet_runner.Portable` with TotalSegmentator's spacing change and class list."""

    Params = Params

    def _names(self, labels: Sequence[Mapping[int, str]]) -> dict[int, str]:
        p: Params = self.ctx.params
        if p.classes is None:
            return super()._names(labels)
        lines = (Path(__file__).parent / p.classes).read_text().splitlines()
        return {k: n for k, n in enumerate(lines, start=1) if n}

    def __call__(self, volume: torch.Tensor, spacing: Sequence[float]) -> dict[str, torch.Tensor]:
        """Labels of one volume on its grid."""
        p: Params = self.ctx.params
        shape = tuple(volume.shape[1:])
        x, sp = volume, tuple(spacing)
        if p.spacing is not None:
            sp32 = np.asarray(spacing, dtype=np.float32)
            if not np.array_equal(sp32, np.full(3, p.spacing)):
                zoom = sp32.astype(np.float64) / p.spacing
                new = tuple(round(n * float(z)) for n, z in zip(shape, zoom, strict=True))
                x = torch.stack([zoom_linear(c, new) for c in volume])
                sp = (p.spacing,) * 3
        out = self.segment(x, sp)
        if tuple(out.shape) != shape:
            out = zoom_nearest(out, shape)
        return {"labels": out}


class Kernels(Portable, nn_runner.Kernels):
    """`fastforks.volume._nnunet_runner.Kernels` with TotalSegmentator's spacing change and class list."""


RUNNERS = {"portable": Portable, "kernels": Kernels}
