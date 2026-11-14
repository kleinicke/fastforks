"""Image classification: `load(model_id)` returns a callable `ClassificationModel` producing a `ClassificationResult`.

```python
from fastforks import classification

model = classification.load("resnet50", mode="kernels")
result = model("cat.jpg")
result.top(3)  # [("tabby", 0.62), ("tiger cat", 0.21), ("Egyptian cat", 0.09)]
```
"""

from fastforks._core.arguments import renamed_attributes
from fastforks._core.shared_types import OLD_TASK_NAMES
from fastforks.classification._api import (
    ClassificationModel,
    ClassificationResult,
    list_models,
    load,
    model_card,
)

__all__ = [
    "ClassificationModel",
    "ClassificationResult",
    "list_models",
    "load",
    "model_card",
]

__getattr__ = renamed_attributes(__name__, OLD_TASK_NAMES)  # the shared types: `fastforks.<Name>`
