"""Optical flow: `load(model_id)` returns a callable `FlowModel` producing a `FlowResult`.

```python
from fastforks import flow

model = flow.load("raft-large")
result = model("frame_0001.png", "frame_0002.png")
result.flow  # [2, H, W] float32 tensor: (dx, dy) in input pixels, from the first image to the second
```
"""

from fastforks._core.cards import ModelCard
from fastforks._core.export import TensorRTEngine
from fastforks._core.image import ImageLike
from fastforks._core.sizes import SizeRange
from fastforks.flow._api import Backend, FlowModel, FlowResult, list_models, load, model_card

__all__ = [
    "Backend",
    "FlowModel",
    "FlowResult",
    "ImageLike",
    "ModelCard",
    "SizeRange",
    "TensorRTEngine",
    "list_models",
    "load",
    "model_card",
]
