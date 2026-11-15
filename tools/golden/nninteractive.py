"""nnInteractive goldens from upstream's own session (`nnInteractiveInferenceSession`, v1.1.5 @ `COMMIT`, unpatched).

Reference: the session on the CPU (float32, `do_autozoom=True`) with the `MIC-DKFZ/nnInteractive` weights
(`nnInteractive_v1.0`, fold 0), on a 96 x 128 x 128 voxel crop of TotalSegmentator dataset case s0013 (CC-BY-4.0;
`CT_SHA256`) around the liver, RAS-canonical `(z, y, x)`: one positive point (the liver voxel deepest inside the organ,
crop coordinates `POINT`). Stored: `input` (Hounsfield units), `point`, `mask` (the session's target buffer, 0 / 1).

    FASTFORKS_GOLDEN_PATH=... <reference venv>/bin/python tools/golden/nninteractive.py SRC CT MODEL_DIR

`SRC` is the nnInteractive-1.1.5 source tree (its package metadata reads version 1.1.5).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import write

UPSTREAM = "https://github.com/MIC-DKFZ/nnInteractive"
COMMIT = "6bf57adf42cae092e01957a81228d929dc03c489"  # v1.1.5
CT_SHA256 = "33a235934cfee4e8534121ca9bfc2e81ca8d4ee62039ed34349d5b753160cba9"  # dataset s0013/ct.nii.gz
START, CROP = (199, 24, 73), (96, 128, 128)  # the whole liver; its deepest voxel is (255, 65, 170) in the volume
POINT = (56, 41, 97)


def crop(path: Path) -> np.ndarray:
    img = nib.as_closest_canonical(nib.load(path))
    vol = np.asarray(img.dataobj, dtype=np.float32).transpose(2, 1, 0)
    return np.ascontiguousarray(vol[tuple(slice(s, s + c) for s, c in zip(START, CROP, strict=True))])


def main() -> None:
    src, ct, model = (Path(a) for a in sys.argv[1:4])
    if hashlib.sha256(ct.read_bytes()).hexdigest() != CT_SHA256:
        msg = f"{ct} is not the TotalSegmentator dataset's s0013/ct.nii.gz"
        raise SystemExit(msg)
    version = importlib.metadata.version
    importlib.metadata.version = lambda n: "1.1.5" if n.lower() == "nninteractive" else version(n)  # no metadata
    sys.path.insert(0, str(src))
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession  # noqa: PLC0415 - from SRC

    vol = crop(ct)
    torch.set_num_threads(16)
    sess = nnInteractiveInferenceSession(
        device=torch.device("cpu"), use_torch_compile=False, verbose=False, torch_n_threads=16, do_autozoom=True,
        use_pinned_memory=False,
    )  # fmt: skip
    sess.initialize_from_trained_model_folder(str(model), 0, "checkpoint_final.pth")
    np.random.seed(0)  # noqa: NPY002 - upstream draws its box covers from the legacy global generator
    sess.set_image(vol[None])
    target = torch.zeros(vol.shape, dtype=torch.uint8)
    sess.set_target_buffer(target)
    sess.add_point_interaction(POINT, include_interaction=True)
    mask = target.numpy().astype(np.float32)
    write(
        "nninteractive@1",
        {"input": vol, "point": np.asarray(POINT, dtype=np.float32), "mask": mask},
        family="nninteractive",
        upstream=f"{UPSTREAM}@{COMMIT}",
        patch=None,
        settings=(
            f"CPU float32, nnInteractiveInferenceSession(do_autozoom=True), fold 0, on s0013 ct.nii.gz, crop {CROP} at "
            f"{START} (z, y, x) RAS, one positive point at {POINT}, torch {torch.__version__}"
        ),
        task="volume",
    )
    print(f"  nninteractive@1: {int(mask.sum())} mask voxels of {mask.size}")


if __name__ == "__main__":
    main()
