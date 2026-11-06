"""SAM 3 (image) goldens from the Hugging Face transformers port (Meta's checkpoint conversion, `facebook/sam3`).

Reference: transformers `Sam3Model` (`REF_VERSION`), CPU float32, `Sam3Processor` input on scikit-image's `coins.png`
(public domain, Greek coins from Pompeii; `IMAGE_SHA256`, converted to RGB) with the concept prompts
`synth.concepts()` (text, exemplar boxes, both; `Sam3Processor` `text`, `input_boxes`, `input_boxes_labels`).
The geometry encoder always runs, as in upstream `Sam3Image._encode_prompt` (transformers skips it without boxes;
patched here by replacing its `has_geometry_prompts` test in `Sam3Model.forward`: with no boxes it encodes its CLS
token only). Stored: the input image (`input`, RGB) and per prompt (key prefix) every query's score
(sigmoid(logit) * sigmoid(presence)) and normalised `x0, y0, x1, y1` box, and the mask logits of the `TOP` best
queries (`pred_masks`, their indices in `mask_query`). Run this with a Python that has transformers and torchvision.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/sam3.py SAM3_DIR COINS_PNG
"""

from __future__ import annotations

import hashlib
import inspect
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from _common import write
from PIL import Image

from synth import concepts

if TYPE_CHECKING:
    from collections.abc import Callable

UPSTREAM = "https://github.com/huggingface/transformers"
REF_VERSION = "v5.19.0"
IMAGE_SHA256 = "f8d773fc9cfa6f4d8e5942dc34d0a0788fcaed2a4fefbbed0aef5398d7ef4cba"  # skimage 0.26 data/coins.png
TOP = 8  # fixed: queries whose mask logits are stored
KEEP = 0.5  # fixed: upstream score threshold (printed count only)


def geometry_forward(ms: object) -> Callable[..., Any]:
    """`Sam3Model.forward` with the geometry encoder always on (upstream behaviour)."""
    src = textwrap.dedent(inspect.getsource(ms.Sam3Model.forward))  # module
    src = src[src.index("def forward") :]
    old = "has_geometry_prompts = input_boxes is not None and input_boxes.numel() > 0"
    if src.count(old) != 1:
        msg = "transformers Sam3Model.forward changed: update the geometry-encoder patch"
        raise SystemExit(msg)
    ns: dict[str, object] = {}
    exec(compile(src.replace(old, "has_geometry_prompts = True"), "sam3-geometry", "exec"), vars(ms), ns)  # noqa: S102 - patched reference source
    return ms.can_return_tuple(ns["forward"])  # module


def main() -> None:
    weights, png = Path(sys.argv[1]), Path(sys.argv[2])
    if hashlib.sha256(png.read_bytes()).hexdigest() != IMAGE_SHA256:
        msg = f"{png} is not scikit-image's coins.png"
        raise SystemExit(msg)
    import transformers  # noqa: PLC0415 - reference environment
    import transformers.models.sam3.modeling_sam3 as ms  # noqa: PLC0415 - reference environment

    if f"v{transformers.__version__}" != REF_VERSION:
        msg = f"transformers {transformers.__version__}, expected {REF_VERSION}"
        raise SystemExit(msg)
    rgb = np.asarray(Image.open(png).convert("RGB"))
    proc = transformers.Sam3Processor.from_pretrained(weights)
    model = ms.Sam3Model.from_pretrained(weights, dtype=torch.float32).eval()
    forward = geometry_forward(ms)
    maps: dict[str, np.ndarray] = {"input": rgb}
    for prefix, p in concepts().items():
        kw: dict[str, Any] = {"text": p.get("text")}
        if "input_boxes" in p:
            kw |= {"input_boxes": [p["input_boxes"]], "input_boxes_labels": [p["input_boxes_labels"]]}
        inputs = proc(images=Image.fromarray(rgb), return_tensors="pt", **kw)
        with torch.inference_mode():
            o = forward(model, **inputs)
        scores = (o.pred_logits.sigmoid() * o.presence_logits.sigmoid())[0]
        top = torch.argsort(scores, descending=True)[:TOP]
        maps |= {
            f"{prefix}scores": scores.numpy(),
            f"{prefix}boxes": o.pred_boxes[0].numpy(),
            f"{prefix}pred_masks": o.pred_masks[0, top].numpy(),
            f"{prefix}mask_query": top.numpy(),
        }
        s = np.sort(scores.numpy())[::-1]
        print(f"  {prefix or 'text_'}: {(s > KEEP).sum()} above {KEEP}; scores {np.round(s[:30], 3)}")
    write(
        "sam3@1",
        maps,
        family="sam3",
        upstream=f"{UPSTREAM}@{REF_VERSION}",
        patch=None,
        settings=f"CPU float32, transformers Sam3Processor + Sam3Model (geometry encoder on), skimage coins.png, "
        f"tests/synth.py concepts() (text, exemplar boxes, both), torch {torch.__version__}",
        task="promptable_segmentation",
    )


if __name__ == "__main__":
    main()
