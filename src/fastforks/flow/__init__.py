"""Optical flow: `load(model_id)` returns a callable `FlowModel` producing a `FlowResult`.

```python
from fastforks import flow

model = flow.load("raft-large")
result = model("frame_0001.png", "frame_0002.png")
result.flow  # [2, H, W] float32 tensor: (dx, dy) in input pixels, from the first image to the second
```
"""

from fastforks._core.arguments import renamed_attributes
from fastforks._core.shared_types import OLD_TASK_NAMES
from fastforks.flow._api import FlowModel, FlowResult, list_models, load, model_card

__all__ = [
    "FlowModel",
    "FlowResult",
    "list_models",
    "load",
    "model_card",
]

__getattr__ = renamed_attributes(__name__, OLD_TASK_NAMES)  # the shared types: `fastforks.<Name>`
