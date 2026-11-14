"""Stereo matching: `load(model_id)` returns a callable `StereoModel` producing a `StereoResult`.

```python
from fastforks import stereo

model = stereo.load("s2m2-s", mode="kernels")
result = model("left.png", "right.png")
result.disparity  # [H, W] float32 tensor, pixels

for r in model.run_many("left/", "right/"):  # many pairs, pipelined, in input order
    r.disparity  # [H, W] float32 numpy

r = model("left.png", "right.png", rig="calib.yaml", views="both")  # unrectified pair + calibration
r.left.depth(), r.right.points()  # StereoView per camera
```
"""

from fastforks._core.arguments import renamed_attributes
from fastforks._core.cards import ModelCard
from fastforks._core.export import TensorRTEngine
from fastforks._core.image import ImageLike
from fastforks._core.sizes import SizeRange
from fastforks.stereo import io as _io
from fastforks.stereo._api import Backend, StereoModel, StereoResult, list_models, load, model_card
from fastforks.stereo.calibrated import StereoView

__all__ = [
    "Backend",
    "ImageLike",
    "ModelCard",
    "SizeRange",
    "StereoModel",
    "StereoResult",
    "StereoView",
    "TensorRTEngine",
    "list_models",
    "load",
    "model_card",
]

_RESULT = ("fastforks.stereo.StereoResult", StereoResult)
__getattr__ = renamed_attributes(
    __name__,
    {
        "PairResult": _RESULT,
        "StereoPairResult": _RESULT,
        "write_disparity": ("StereoResult.save(path) (a .png is the KITTI format)", _io.write_disparity),
    },
)
