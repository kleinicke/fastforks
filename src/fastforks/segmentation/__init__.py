"""Semantic segmentation: `load(model_id)` returns a callable `SegmentationModel` producing a `SegmentationResult`.

```python
from fastforks import segmentation

model = segmentation.load("segformer-b0-ade20k", mode="kernels")
result = model("street.jpg")
result.labels  # [H, W] int64 class indices; result.class_names[i] names class i
```
"""

from fastforks._core.cards import ModelCard
from fastforks._core.export import TensorRTEngine
from fastforks._core.image import ImageLike
from fastforks._core.sizes import SizeRange
from fastforks.segmentation._api import (
    Backend,
    SegmentationModel,
    SegmentationResult,
    list_models,
    load,
    model_card,
)

__all__ = [
    "Backend",
    "ImageLike",
    "ModelCard",
    "SegmentationModel",
    "SegmentationResult",
    "SizeRange",
    "TensorRTEngine",
    "list_models",
    "load",
    "model_card",
]
