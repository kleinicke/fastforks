"""Generate references for the six additional whole-body CT checkpoints.

Install upstream at COMMIT and nnU-Net 2.8.1 and set FASTFORKS_GOLDEN_PATH. Pass the
upstream tests/reference_files/aorta_report/example_ct.nii.gz and optionally one or more model IDs.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _HERE]
from totalsegmentator.python_api import totalsegmentator  # noqa: E402

sys.path.insert(0, str(_HERE))
from _common import write  # noqa: E402

COMMIT = "88060518cc3d556fbcf1894d05f1d1e65de52277"
CT_SHA256 = "372d98723e3283a8d4a0ae2f0ef1e581c0ad17b673b30b2e8d3e215180cc2a12"
VARIANTS = {
    "totalsegmentator-total-6mm": ("total", False, True, "big"),
    "totalsegmentator-v3-total-6mm": ("total_v3", False, True, "big"),
    "totalsegmentator-v3-total-3mm": ("total_v3", True, False, "big"),
    "totalsegmentator-v3-total-small-3mm": ("total_v3", True, False, "small"),
    "totalsegmentator-v3-total-small": ("total_v3", False, False, "small"),
    "totalsegmentator-v3-total": ("total_v3", False, False, "big"),
}


def main() -> None:
    ct = Path(sys.argv[1])
    if hashlib.sha256(ct.read_bytes()).hexdigest() != CT_SHA256:
        msg = "Expected example_ct.nii.gz from the pinned upstream checkout"
        raise ValueError(msg)
    img = nib.as_closest_canonical(nib.load(ct))
    starts = [(n - 96) // 2 for n in img.shape]
    img = img.slicer[tuple(slice(i, i + 96) for i in starts)]
    vol = np.asarray(img.dataobj, dtype=np.float32).T
    spacing = np.asarray(img.header.get_zooms()[::-1], dtype=np.float32)
    for name in sys.argv[2:] or VARIANTS:
        task, fast, fastest, size = VARIANTS[name]
        seg = totalsegmentator(img, task=task, fast=fast, fastest=fastest, model_size=size, ml=True, device="cpu")
        write(
            f"{name}@1",
            {"input": vol, "spacing": spacing, "labels": np.asarray(seg.dataobj).T},
            family="totalsegmentator",
            upstream=f"https://github.com/wasserth/TotalSegmentator@{COMMIT}",
            patch=None,
            settings=f"example_ct centre 96^3 RAS, CPU float32, torch {torch.__version__}, "
            f"task={task}, fast={fast}, fastest={fastest}, model_size={size}",
            task="volume",
        )


if __name__ == "__main__":
    main()
