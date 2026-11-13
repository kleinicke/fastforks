"""Frame interpolation: `load(model_id)` returns a callable `InterpolationModel` producing an `InterpolationResult`.

```python
from fastforks import interpolation

model = interpolation.load("rife-4.25", mode="kernels")
result = model("frame_0001.png", "frame_0002.png")  # the frame half-way between them
result.frame  # [3, H, W] float32 RGB in [0, 1]
```
"""

from fastforks._core.cards import ModelCard
from fastforks._core.export import TensorRTEngine
from fastforks._core.image import ImageLike
from fastforks._core.sizes import SizeRange
from fastforks.interpolation._api import (
    Backend,
    InterpolationModel,
    InterpolationResult,
    list_models,
    load,
    model_card,
)

__all__ = [
    "Backend",
    "ImageLike",
    "InterpolationModel",
    "InterpolationResult",
    "ModelCard",
    "SizeRange",
    "TensorRTEngine",
    "list_models",
    "load",
    "model_card",
]
