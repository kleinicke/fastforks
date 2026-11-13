"""Matting / background removal: `load(model_id)` returns a callable `MattingModel` producing a `MattingResult`.

```python
from fastforks import matting

model = matting.load("birefnet", mode="kernels")
result = model("photo.jpg")
result.alpha  # [H, W] float32 foreground alpha in [0, 1]
```
"""

from fastforks._core.cards import ModelCard
from fastforks._core.export import TensorRTEngine
from fastforks._core.image import ImageLike
from fastforks._core.sizes import SizeRange
from fastforks.matting._api import Backend, MattingModel, MattingResult, list_models, load, model_card

__all__ = [
    "Backend",
    "ImageLike",
    "MattingModel",
    "MattingResult",
    "ModelCard",
    "SizeRange",
    "TensorRTEngine",
    "list_models",
    "load",
    "model_card",
]
