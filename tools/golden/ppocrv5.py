"""PP-OCRv5 goldens from PaddleOCR 3 (the PaddlePaddle original).

Reference: PaddleOCR's `PaddleOCR` pipeline (PaddleX `OCR`, `REF_COMMIT` of PaddleOCR) on the CPU in float32, without
the document preprocessor and the text-line orientation classifier, as the card. Input: PaddleOCR's
`tests/test_files/table.jpg` (Apache-2.0, 551 x 132, 12 text lines; stored as `input`, RGB), read with OpenCV as
upstream. Outputs: the quads, the texts as code points (`goldens.text_codes`) and the scores.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/ppocrv5.py PADDLE_PYTHON PADDLEOCR_CHECKOUT WEIGHTS_DIR

`PADDLE_PYTHON` is a Python with `paddlepaddle` (3.3) and `paddleocr` (3.3, with its `paddlex`), outside this repo;
it runs `ppocrv5_paddle.py`. `WEIGHTS_DIR` holds the card's `PP-OCRv5_*` directories.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from _common import write

UPSTREAM = "https://github.com/PaddlePaddle/PaddleOCR"
REF_COMMIT = "dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf"
IMAGE = "tests/test_files/table.jpg"


def main() -> None:
    paddle_python, repo, weights = (Path(a) for a in sys.argv[1:4])
    with tempfile.TemporaryDirectory(prefix="ff-ocr-") as tmp:
        image = Path(tmp) / "table.jpg"
        archive = subprocess.run(["git", "-C", repo, "show", f"{REF_COMMIT}:{IMAGE}"], check=True, capture_output=True)
        image.write_bytes(archive.stdout)
        rgb = cv2.imread(str(image))[..., ::-1].astype(np.float32)
        for kind in ("server", "mobile"):
            out = Path(tmp) / f"{kind}.npz"
            script = Path(__file__).with_name("ppocrv5_paddle.py")
            subprocess.run([paddle_python, "-I", script, weights, kind, image, out], check=True)
            with np.load(out) as res:
                maps = {k: res[k] for k in ("boxes", "texts", "scores")}
                versions = str(res["versions"])
            write(
                f"ppocrv5-{kind}@1",
                maps | {"input": rgb},
                family="ppocrv5",
                upstream=f"{UPSTREAM}@{REF_COMMIT}",
                patch=None,
                settings=f"CPU float32 (no oneDNN), pipeline without orientation / unwarping, {IMAGE}, {versions}",
                task="ocr",
            )
            print(f"  {kind}: {len(maps['scores'])} lines")


if __name__ == "__main__":
    main()
