"""Edge / line maps: `load(model_id)` returns a callable `EdgesModel` producing an `EdgesResult`.

The line and edge preprocessors of ControlNet (controlnet_aux, ComfyUI's ControlNet preprocessors): line art, anime
line art, HED, PiDiNet, TEED / Anyline and M-LSD straight line segments.

```python
from fastforks import edges

model = edges.load("lineart", options={"resolution": 1024})
result = model("photo.jpg")
result.edges  # [H, W] float32 in [0, 1], 1 = line (white lines on black, as ControlNet takes them)
edges.load("mlsd")("room.jpg").lines  # [N, 4] float32 segments (x0, y0, x1, y1) in input pixels
```
"""

from fastforks._core.cards import ModelCard
from fastforks._core.export import TensorRTEngine
from fastforks._core.image import ImageLike
from fastforks._core.sizes import SizeRange
from fastforks.edges._api import Backend, EdgesModel, EdgesResult, list_models, load, model_card

__all__ = [
    "Backend",
    "EdgesModel",
    "EdgesResult",
    "ImageLike",
    "ModelCard",
    "SizeRange",
    "TensorRTEngine",
    "list_models",
    "load",
    "model_card",
]
