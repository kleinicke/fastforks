"""CLIPSeg goldens from the Hugging Face transformers port (the authors' converted `CIDAS/clipseg-rd64-refined`).

Reference: transformers `CLIPSegForImageSegmentation` (`REF_VERSION`), CPU float32, `CLIPSegProcessor` input on
scikit-image's `coins.png` (`IMAGE_SHA256`, converted to RGB) with the text prompts of `synth.concepts()` (CLIPSeg
takes no exemplar boxes). Stored: the input image (`input`, RGB) and per prompt (key prefix) the one instance the
runner returns: the highest pixel probability (`scores`), the bounding box of the logits > 0 normalised by 352
(`boxes`, `x0, y0, x1, y1`), the mask logits `[1, 352, 352]` (`pred_masks`, query `mask_query` = 0). Run this with a
Python that has transformers and torchvision.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/clipseg.py CLIPSEG_DIR COINS_PNG
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from _common import write
from PIL import Image

from synth import concepts

UPSTREAM = "https://github.com/huggingface/transformers"
REF_VERSION = "v5.19.0"
IMAGE_SHA256 = "f8d773fc9cfa6f4d8e5942dc34d0a0788fcaed2a4fefbbed0aef5398d7ef4cba"  # skimage 0.26 data/coins.png


def box(mask: np.ndarray) -> np.ndarray:
    """Bounding box `(x0, y0, x1, y1)` of a boolean `[h, w]` mask, normalised by its size (zeros if empty)."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return np.zeros(4, np.float32)
    return np.array([xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h], np.float32)


def main() -> None:
    weights, png = Path(sys.argv[1]), Path(sys.argv[2])
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's coins.png"
        raise SystemExit(msg)
    import transformers  # noqa: PLC0415 - reference environment

    if f"v{transformers.__version__}" != REF_VERSION:
        msg = f"transformers {transformers.__version__}, expected {REF_VERSION}"
        raise SystemExit(msg)
    rgb = np.asarray(Image.open(png).convert("RGB"))
    proc = transformers.CLIPSegProcessor.from_pretrained(weights)
    model = transformers.CLIPSegForImageSegmentation.from_pretrained(weights, dtype=torch.float32).eval()
    maps: dict[str, np.ndarray] = {"input": rgb}
    for prefix, p in concepts().items():
        if "input_boxes" in p:  # text prompts only
            continue
        inputs = proc(images=Image.fromarray(rgb), text=[p["text"]], return_tensors="pt", padding=True)
        with torch.inference_mode():
            logits = model(**inputs).logits.reshape(1, 352, 352)[0].numpy()
        maps |= {
            f"{prefix}scores": 1 / (1 + np.exp(-logits.max()[None])),
            f"{prefix}boxes": box(logits > 0)[None],
            f"{prefix}pred_masks": logits[None],
            f"{prefix}mask_query": np.zeros(1, np.int64),
        }
        print(f"  {prefix or 'text_'}: max prob {maps[f'{prefix}scores'][0]:.4f}, {(logits > 0).mean():.4f} of pixels")
    write(
        "clipseg@1",
        maps,
        family="clipseg",
        upstream=f"{UPSTREAM}@{REF_VERSION}",
        patch=None,
        settings=f"CPU float32, transformers CLIPSegProcessor + CLIPSegForImageSegmentation, skimage coins.png, "
        f"tests/synth.py concepts() (text prompts), torch {torch.__version__}",
        task="promptable_segmentation",
    )


if __name__ == "__main__":
    main()
