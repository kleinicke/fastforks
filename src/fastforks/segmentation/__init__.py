"""Semantic segmentation: `load(model_id)` returns a callable `SegmentationModel` producing a `SegmentationResult`.

```python
from fastforks import segmentation

model = segmentation.load("segformer-b0-ade20k", mode="kernels")
result = model("street.jpg")
result.labels  # [H, W] int64 class indices; result.class_names[i] names class i
```
"""

from fastforks._core.arguments import renamed_attributes
from fastforks._core.shared_types import OLD_TASK_NAMES
from fastforks.segmentation._api import (
    SegmentationModel,
    SegmentationResult,
    list_models,
    load,
    model_card,
)

__all__ = [
    "SegmentationModel",
    "SegmentationResult",
    "list_models",
    "load",
    "model_card",
]

__getattr__ = renamed_attributes(__name__, OLD_TASK_NAMES)  # the shared types: `fastforks.<Name>`
