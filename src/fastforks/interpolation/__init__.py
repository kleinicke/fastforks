"""Frame interpolation: `load(model_id)` returns a callable `InterpolationModel` producing an `InterpolationResult`.

```python
from fastforks import interpolation

model = interpolation.load("rife-4.25", mode="kernels")
result = model("frame_0001.png", "frame_0002.png")  # the frame half-way between them
result.frame  # [3, H, W] float32 RGB in [0, 1]
```
"""

from fastforks._core.arguments import renamed_attributes
from fastforks._core.shared_types import OLD_TASK_NAMES
from fastforks.interpolation._api import (
    InterpolationModel,
    InterpolationResult,
    list_models,
    load,
    model_card,
)

__all__ = [
    "InterpolationModel",
    "InterpolationResult",
    "list_models",
    "load",
    "model_card",
]

__getattr__ = renamed_attributes(__name__, OLD_TASK_NAMES)  # the shared types: `fastforks.<Name>`
