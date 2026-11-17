"""Image restoration (super-resolution): `load(model_id)` returns a callable `RestorationModel`.

```python
from fastforks import restoration

model = restoration.load("realesrgan-x4plus", mode="kernels")
result = model("photo.png")  # tile=512 for large images
result.image  # [3, 4H, 4W] float32 RGB in [0, 1]

model = restoration.load("4x-UltraSharp.pth")  # a checkpoint file of a supported architecture (spandrel formats)
```
"""

from fastforks._core.arguments import renamed_attributes
from fastforks._core.shared_types import OLD_TASK_NAMES
from fastforks.restoration._api import (
    RestorationModel,
    RestorationResult,
    list_models,
    load,
    model_card,
)

__all__ = [
    "RestorationModel",
    "RestorationResult",
    "list_models",
    "load",
    "model_card",
]

__getattr__ = renamed_attributes(__name__, OLD_TASK_NAMES)  # the shared types: `fastforks.<Name>`
