"""CREStereo goldens from the reference implementation (the MegEngine original).

Reference: upstream megvii-research/CREStereo (`REF_COMMIT`, MegEngine), CPU float32, upstream `test.py` inference
(two passes: half resolution from zero flow, then full resolution from its flow; 20 iterations each). Upstream reads
the images with `cv2.imread`, so the network sees BGR: the synthetic RGB pair is flipped to BGR. Upstream `test.py`
resizes instead of padding; the pair is padded to a multiple of 32 (centre, replicate), as the card's `[io]` does.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/crestereo.py MGE_PYTHON CRESTEREO_CHECKOUT WEIGHTS_DIR

`MGE_PYTHON` is a Python <= 3.10 with `megengine` and `opencv-python-headless` (outside this repo); it runs
`crestereo_mge.py` on an export of `REF_COMMIT`. `WEIGHTS_DIR` holds the authors' `crestereo_eth3d.mge`.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from _common import pair, reference_tree, write

UPSTREAM = "https://github.com/megvii-research/CREStereo"
REF_COMMIT = "ad3a1613bdedd88b93247e5f002cb7c80799762d"
MULTIPLE = 32


def main() -> None:
    mge_python, repo, weights = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    tree = reference_tree(repo, REF_COMMIT)
    left, right = (x[..., ::-1].astype(np.float32) for x in pair())  # RGB -> BGR (cv2.imread order)
    h, w = left.shape[:2]
    ph, pw = (-h) % MULTIPLE, (-w) % MULTIPLE
    pads = ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0))
    with tempfile.TemporaryDirectory(prefix="ff-cre-") as tmp:
        inp, out = Path(tmp) / "in.npz", Path(tmp) / "out.npz"
        np.savez(inp, left=np.pad(left, pads, mode="edge"), right=np.pad(right, pads, mode="edge"))
        script = Path(__file__).with_name("crestereo_mge.py")
        cmd = [mge_python, "-I", script, tree, weights / "crestereo_eth3d.mge", inp, out]
        subprocess.run(cmd, check=True)
        with np.load(out) as res:
            disp, version = res["disparity"], str(res["megengine"])
    disp = disp[pads[0][0] : pads[0][0] + h, pads[1][0] : pads[1][0] + w]
    write(
        "crestereo-eth3d@1",
        {"disparity": disp},
        family="crestereo",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=None,
        settings=f"CPU float32, test.py inference (2 passes x 20 iters), BGR input, megengine {version}",
    )
    print(f"  median disparity {np.median(disp):.4f}")


if __name__ == "__main__":
    main()
