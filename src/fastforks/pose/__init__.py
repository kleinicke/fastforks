"""2D human pose estimation: `load(model_id)` returns a callable `PoseModel` producing a `PoseResult` per image.

```python
from fastforks import detection, pose

people = detection.load("rtdetrv2-s-coco")("street.jpg")
people = people[people.labels == 0]  # the "person" class of coco-80
model = pose.load("vitpose-plus-base", mode="kernels")
result = model("street.jpg", boxes=people.boxes)
result.keypoints, result.keypoint_scores  # [P, 17, 2] input pixels, [P, 17]
result.keypoint_names, result.skeleton  # COCO names and the limbs (pairs of keypoint indices)
```

Top-down models (ViTPose) estimate one person per box: pass the boxes from a detector (or ground truth). A batch
(BHWC numpy / BCHW torch) takes a list of box arrays, one per image, and gives a list of results.
"""

from fastforks._core.cards import ModelCard
from fastforks._core.export import TensorRTEngine
from fastforks._core.image import ImageLike
from fastforks._core.sizes import SizeRange
from fastforks.pose._api import (
    Backend,
    PoseModel,
    PoseResult,
    list_models,
    load,
    model_card,
)

__all__ = [
    "Backend",
    "ImageLike",
    "ModelCard",
    "PoseModel",
    "PoseResult",
    "SizeRange",
    "TensorRTEngine",
    "list_models",
    "load",
    "model_card",
]
