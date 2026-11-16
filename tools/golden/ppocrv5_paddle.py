"""Reference side of `ppocrv5.py`: PaddleOCR 3's `PaddleOCR` pipeline in a PaddlePaddle venv.

    PADDLE_PYTHON -I tools/golden/ppocrv5_paddle.py WEIGHTS_DIR KIND IMAGE OUTPUT.npz

`WEIGHTS_DIR` holds `PP-OCRv5_<KIND>_det/` and `PP-OCRv5_<KIND>_rec/` (the card's files); KIND is `server` or
`mobile`. The pipeline runs as the card describes (no document orientation / unwarping, no text-line orientation
classifier), on the CPU in float32 without oneDNN. Writes the recognised quads (`rec_polys`), the texts as code points
and the scores, plus the package versions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    weights, kind, image, output = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"  # no model hoster lookups: the models are local
    import paddle  # noqa: PLC0415  # reference-only dependency
    import paddleocr  # noqa: PLC0415
    import paddlex  # noqa: PLC0415

    names = {f"text_{t}_model_name": f"PP-OCRv5_{kind}_{s}" for t, s in (("detection", "det"), ("recognition", "rec"))}
    dirs = {k.replace("_name", "_dir"): str(weights / v) for k, v in names.items()}
    ocr = paddleocr.PaddleOCR(
        **names,
        **dirs,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
        enable_mkldnn=False,
    )
    r = ocr.predict(image)[0]
    texts = list(r["rec_texts"])
    codes = np.zeros((len(texts), max(map(len, texts), default=0)), np.float32)
    for i, t in enumerate(texts):
        codes[i, : len(t)] = [ord(c) for c in t]
    np.savez(
        output,
        boxes=np.asarray(r["rec_polys"], np.float32).reshape(-1, 4, 2),
        texts=codes,
        scores=np.asarray(r["rec_scores"], np.float32),
        versions=np.array(
            f"paddlepaddle {paddle.__version__}, paddleocr {paddleocr.__version__}, paddlex {paddlex.__version__}"
        ),
    )


if __name__ == "__main__":
    main()
