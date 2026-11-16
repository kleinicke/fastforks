"""TotalSegmentator goldens from upstream's `totalsegmentator` package (2.18.0 @ `COMMIT`, nnU-Net v2 2.8.1, unpatched).

Reference: `totalsegmentator.python_api.totalsegmentator(image, ml=True, fast=..., device="cpu")` (CPU float32) on a
96 x 128 x 128 voxel crop (1.5 mm) of TotalSegmentator dataset case s0013 (CC-BY-4.0; `CT_SHA256`), RAS-canonical
`(z, y, x)` as `fastforks.volume.read_nifti` gives it. Stored: `input` (Hounsfield units), `spacing`, `labels`.

    FASTFORKS_GOLDEN_PATH=... TOTALSEG_HOME_DIR=... <TotalSegmentator venv>/bin/python \
        tools/golden/totalsegmentator.py CT
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

# this file's name hides upstream's package: import it without the script directory (also in nnU-Net's spawned workers)
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _HERE]
from totalsegmentator.python_api import totalsegmentator  # noqa: E402 - upstream, reference venv only

sys.path.insert(0, str(_HERE))
from _common import write  # noqa: E402

UPSTREAM = "https://github.com/wasserth/TotalSegmentator"
COMMIT = "184fa4765456492b2ab81a981cc387465906b003"  # v2.18.0
CT_SHA256 = "33a235934cfee4e8534121ca9bfc2e81ca8d4ee62039ed34349d5b753160cba9"  # dataset v2 s0013/ct.nii.gz
CROP = (96, 128, 128)
VARIANTS = {"totalsegmentator-total-3mm@1": True, "totalsegmentator-total@1": False}  # ref: fast


def crop(path: Path) -> tuple[np.ndarray, tuple[float, float, float], np.ndarray]:
    """The centre crop `[D, H, W]` (float32), its spacing (D, H, W) and the RAS affine of the crop."""
    img = nib.as_closest_canonical(nib.load(path))
    vol = np.asarray(img.dataobj, dtype=np.float32).transpose(2, 1, 0)
    start = [(n - c) // 2 for n, c in zip(vol.shape, CROP, strict=True)]
    vol = np.ascontiguousarray(vol[tuple(slice(s, s + c) for s, c in zip(start, CROP, strict=True))])
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return vol, (zooms[2], zooms[1], zooms[0]), np.diag([*zooms, 1.0])


def main() -> None:
    ct = Path(sys.argv[1])
    if hashlib.sha256(ct.read_bytes()).hexdigest() != CT_SHA256:
        msg = f"{ct} is not the TotalSegmentator dataset's s0013/ct.nii.gz"
        raise SystemExit(msg)
    vol, spacing, affine = crop(ct)
    for ref, fast in VARIANTS.items():
        seg = totalsegmentator(nib.Nifti1Image(vol.transpose(2, 1, 0), affine), None, ml=True, fast=fast, device="cpu")
        labels = np.asarray(seg.dataobj).transpose(2, 1, 0).astype(np.float32)
        write(
            ref,
            {"input": vol, "spacing": np.asarray(spacing, dtype=np.float32), "labels": labels},
            family="totalsegmentator",
            upstream=f"{UPSTREAM}@{COMMIT}",
            patch=None,
            settings=(
                f"CPU float32, totalsegmentator(ml=True, fast={fast}, device='cpu') on s0013 ct.nii.gz, centre crop "
                f"{CROP} (z, y, x) RAS, torch {torch.__version__}"
            ),
            task="volume",
        )
        print(f"  {ref}: {len(np.unique(labels)) - 1} classes")


if __name__ == "__main__":
    main()
