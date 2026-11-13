"""Image restoration (super-resolution): `load(model_id)` returns a callable `RestorationModel`.

```python
from fastforks import restoration

model = restoration.load("realesrgan-x4plus", mode="kernels")
result = model("photo.png")  # tile=512 for large images
result.image  # [3, 4H, 4W] float32 RGB in [0, 1]

model = restoration.load_file("4x-UltraSharp.pth")  # a checkpoint of a supported architecture (spandrel formats)
```
"""

from fastforks._core.cards import ModelCard
from fastforks._core.export import TensorRTEngine
from fastforks._core.image import ImageLike
from fastforks._core.sizes import SizeRange
from fastforks.restoration._api import (
    Backend,
    RestorationModel,
    RestorationResult,
    checkpoint_family,
    list_models,
    load,
    load_file,
    model_card,
)

__all__ = [
    "Backend",
    "ImageLike",
    "ModelCard",
    "RestorationModel",
    "RestorationResult",
    "SizeRange",
    "TensorRTEngine",
    "checkpoint_family",
    "list_models",
    "load",
    "load_file",
    "model_card",
]
