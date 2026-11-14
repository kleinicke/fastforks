"""Single-image camera calibration: `load(model_id)` returns a callable `CalibrationModel` (a `CalibrationResult`).

```python
from fastforks import calibration

model = calibration.load("geocalib-pinhole", mode="kernels")
cam = model("photo.jpg")
cam.intrinsics  # [3, 3] float32 pinhole matrix in pixels of the image
cam.roll, cam.pitch, cam.vfov  # radians; cam.gravity: unit vector in the camera frame
```
"""

from fastforks._core.arguments import renamed_attributes
from fastforks._core.shared_types import OLD_TASK_NAMES
from fastforks.calibration._api import (
    CalibrationModel,
    CalibrationResult,
    list_models,
    load,
    model_card,
)

__all__ = [
    "CalibrationModel",
    "CalibrationResult",
    "list_models",
    "load",
    "model_card",
]

__getattr__ = renamed_attributes(__name__, OLD_TASK_NAMES)  # the shared types: `fastforks.<Name>`
