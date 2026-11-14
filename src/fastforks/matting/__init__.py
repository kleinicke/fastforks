"""Matting / background removal: `load(model_id)` returns a callable `MattingModel` producing a `MattingResult`.

```python
from fastforks import matting

model = matting.load("birefnet", mode="kernels")
result = model("photo.jpg")
result.alpha  # [H, W] float32 foreground alpha in [0, 1]
```
"""

from fastforks._core.arguments import renamed_attributes
from fastforks._core.shared_types import OLD_TASK_NAMES
from fastforks.matting._api import (
    MattingModel,
    MattingResult,
    VideoMattingResult,
    list_models,
    load,
    model_card,
)

__all__ = [
    "MattingModel",
    "MattingResult",
    "VideoMattingResult",
    "list_models",
    "load",
    "model_card",
]

__getattr__ = renamed_attributes(__name__, OLD_TASK_NAMES)  # the shared types: `fastforks.<Name>`
