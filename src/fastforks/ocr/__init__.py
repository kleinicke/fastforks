"""Text recognition (OCR): `load(model_id)` returns a callable `OCRModel` producing an `OCRResult` per image.

Detector + recogniser pipelines (PP-OCRv5: text lines of any orientation, Chinese / English / Japanese and more) and
recognisers of single text-line crops (PARSeq: English).

```python
from fastforks import ocr

model = ocr.load("ppocrv5-server")
result = model("receipt.jpg")
result.texts  # ["TOTAL", "12.50", ...], reading order (top to bottom, then left to right)
result.boxes  # [N, 4, 2] float32 quads (x, y) in input pixels, clockwise from the top-left corner
result.scores  # [N] float32 mean character confidence
ocr.load("parseq")("word.png").texts[0]  # a recogniser reads the whole image as one text line
```
"""

from fastforks._core.arguments import renamed_attributes
from fastforks._core.shared_types import OLD_TASK_NAMES
from fastforks.ocr._api import OCRModel, OCRResult, list_models, load, model_card

__all__ = [
    "OCRModel",
    "OCRResult",
    "list_models",
    "load",
    "model_card",
]

__getattr__ = renamed_attributes(__name__, OLD_TASK_NAMES)  # the shared types: `fastforks.<Name>`
