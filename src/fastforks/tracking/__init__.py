"""Point tracking: `load(model_id)` returns a callable `TrackingModel` giving a `TrackingResult`.

```python
from fastforks import tracking

model = tracking.load("tapnext")
result = model(frames, queries)  # frames: [T, H, W, 3] uint8 (or a list of images); queries: [Q, 3] (t, x, y) pixels
result.tracks  # [T, Q, 2] float32: (x, y) of every query point in every frame, in pixels
result.visibility  # [T, Q] float32 in [0, 1]: probability that the point is visible in the frame
result.visible  # [T, Q] bool: visibility > 0.5
```

A query is a point in one frame (`t`, pixel `x`, `y`); the model tracks it through the whole video. Models that run
online (TAPNext) report nothing before a query's frame: their tracks there are the model's guess with a low visibility.
"""

from fastforks._core.cards import ModelCard
from fastforks._core.export import TensorRTEngine
from fastforks._core.image import ImageLike
from fastforks._core.sizes import SizeRange
from fastforks.tracking._api import Backend, TrackingModel, TrackingResult, list_models, load, model_card

__all__ = [
    "Backend",
    "ImageLike",
    "ModelCard",
    "SizeRange",
    "TensorRTEngine",
    "TrackingModel",
    "TrackingResult",
    "list_models",
    "load",
    "model_card",
]
