"""Object detection: `load(model_id)` returns a callable `DetectionModel` producing a `DetectionResult` per image.

```python
from fastforks import detection

model = detection.load("mask-rcnn-r50-fpn-3x-coco", mode="kernels")
result = model("street.jpg")
for d in result.to_list():  # best first
    print(d["class_name"], d["score"], d["box"])  # d["mask"]: [H, W] bool (mask models)
```

A batch (BHWC numpy / BCHW torch) gives a list with one result per image. The result fields are the keys of the
Hugging Face transformers `post_process_object_detection` dicts (`scores`, `labels`, `boxes`, + `masks`). Detectors
need torchvision's NMS / ROIAlign operators: `pip install 'fastforks-core[torchvision]'`.
"""

from fastforks._core.cards import ModelCard
from fastforks._core.export import TensorRTEngine
from fastforks._core.image import ImageLike
from fastforks._core.sizes import SizeRange
from fastforks.detection._api import (
    Backend,
    DetectionModel,
    DetectionResult,
    list_models,
    load,
    model_card,
)

__all__ = [
    "Backend",
    "DetectionModel",
    "DetectionResult",
    "ImageLike",
    "ModelCard",
    "SizeRange",
    "TensorRTEngine",
    "list_models",
    "load",
    "model_card",
]
