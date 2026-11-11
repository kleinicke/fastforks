"""Dense matching: `load(model_id)` returns a callable `MatchingModel` producing a `MatchingResult`.

```python
from fastforks import matching

model = matching.load("romav2")
result = model("view_a.jpg", "view_b.jpg")  # the two images may differ in size
result.warp  # [2, H1, W1] float32: (x, y) in pixels of the second image for every pixel of the first
result.certainty  # [H1, W1] float32 in [0, 1]: how likely the pixel is visible in the second image
```
"""

from fastforks._core.cards import ModelCard
from fastforks._core.export import TensorRTEngine
from fastforks._core.image import ImageLike
from fastforks._core.sizes import SizeRange
from fastforks.matching._api import Backend, MatchingModel, MatchingResult, list_models, load, model_card

__all__ = [
    "Backend",
    "ImageLike",
    "MatchingModel",
    "MatchingResult",
    "ModelCard",
    "SizeRange",
    "TensorRTEngine",
    "list_models",
    "load",
    "model_card",
]
