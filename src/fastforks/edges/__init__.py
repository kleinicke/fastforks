"""Edge / line maps: `load(model_id)` returns a callable `EdgesModel` producing an `EdgesResult`.

The line and edge preprocessors of ControlNet (controlnet_aux, ComfyUI's ControlNet preprocessors): line art, anime
line art, HED, PiDiNet, TEED / Anyline and M-LSD straight line segments.

```python
from fastforks import edges

model = edges.load("lineart", short_side=1024)
result = model("photo.jpg")
result.edges  # [H, W] float32 in [0, 1], 1 = line (white lines on black, as ControlNet takes them)
edges.load("mlsd")("room.jpg").lines  # [N, 4] float32 segments (x0, y0, x1, y1) in input pixels
```
"""

from fastforks._core.arguments import renamed_attributes
from fastforks._core.shared_types import OLD_TASK_NAMES
from fastforks.edges._api import EdgesModel, EdgesResult, list_models, load, model_card

__all__ = [
    "EdgesModel",
    "EdgesResult",
    "list_models",
    "load",
    "model_card",
]

__getattr__ = renamed_attributes(__name__, OLD_TASK_NAMES)  # the shared types: `fastforks.<Name>`
