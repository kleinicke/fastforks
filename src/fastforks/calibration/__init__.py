"""Single-image camera calibration: `load(model_id)` returns a callable `CalibrationModel` (a `CalibrationResult`).

```python
from fastforks import calibration

model = calibration.load("geocalib-pinhole", mode="kernels")
cam = model("photo.jpg")
cam.intrinsics  # [3, 3] float32 pinhole matrix in pixels of the image
cam.roll, cam.pitch, cam.vfov  # radians; cam.gravity: unit vector in the camera frame
```
"""

from fastforks._core.cards import ModelCard
from fastforks._core.export import TensorRTEngine
from fastforks._core.image import ImageLike
from fastforks._core.sizes import SizeRange
from fastforks.calibration._api import (
    Backend,
    CalibrationModel,
    CalibrationResult,
    list_models,
    load,
    model_card,
)

__all__ = [
    "Backend",
    "CalibrationModel",
    "CalibrationResult",
    "ImageLike",
    "ModelCard",
    "SizeRange",
    "TensorRTEngine",
    "list_models",
    "load",
    "model_card",
]
