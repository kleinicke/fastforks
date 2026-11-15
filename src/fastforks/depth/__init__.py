"""Monocular depth: `load(model_id)` returns a callable `DepthModel` producing a `DepthResult`.

```python
from fastforks import depth

model = depth.load("depth-anything-v2-s")  # mode="auto": the kernels on CUDA, portable elsewhere
result = model("room.jpg")
result.inverse_depth  # [H, W] float32, relative (result.kind says what the maps mean)
```
"""

from fastforks._core.arguments import renamed_attributes
from fastforks._core.shared_types import OLD_TASK_NAMES
from fastforks.depth._api import DepthKind, DepthModel, DepthResult, list_models, load, model_card

__all__ = [
    "DepthKind",
    "DepthModel",
    "DepthResult",
    "list_models",
    "load",
    "model_card",
]

__getattr__ = renamed_attributes(__name__, OLD_TASK_NAMES)  # the shared types: `fastforks.<Name>`
