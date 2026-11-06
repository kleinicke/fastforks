"""Monocular depth: `load(model_id)` returns a callable `DepthModel` producing a `DepthResult`.

```python
from fastforks import depth

model = depth.load("depth-anything-v2-s")  # mode="auto": the kernels on CUDA, portable elsewhere
result = model("room.jpg")
result.inverse_depth  # [H, W] float32, relative (result.kind says what the maps mean)
```
"""

from fastforks._core.cards import ModelCard
from fastforks._core.export import TensorRTEngine
from fastforks._core.image import ImageLike
from fastforks._core.sizes import SizeRange
from fastforks.depth._api import Backend, DepthKind, DepthModel, DepthResult, list_models, load, model_card

__all__ = [
    "Backend",
    "DepthKind",
    "DepthModel",
    "DepthResult",
    "ImageLike",
    "ModelCard",
    "SizeRange",
    "TensorRTEngine",
    "list_models",
    "load",
    "model_card",
]
