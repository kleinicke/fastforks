"""Image classification: `load(model_id)` returns a callable `ClassificationModel` producing a `ClassificationResult`.

```python
from fastforks import classification

model = classification.load("resnet50", mode="kernels")
result = model("cat.jpg")
result.top(3)  # [("tabby", 0.62), ("tiger cat", 0.21), ("Egyptian cat", 0.09)]
```
"""

from fastforks._core.cards import ModelCard
from fastforks._core.export import TensorRTEngine
from fastforks._core.image import ImageLike
from fastforks._core.sizes import SizeRange
from fastforks.classification._api import (
    Backend,
    ClassificationModel,
    ClassificationResult,
    list_models,
    load,
    model_card,
)

__all__ = [
    "Backend",
    "ClassificationModel",
    "ClassificationResult",
    "ImageLike",
    "ModelCard",
    "SizeRange",
    "TensorRTEngine",
    "list_models",
    "load",
    "model_card",
]
