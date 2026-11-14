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

Multi-object tracking by detection (ByteTrack, on any detector; CPU):

```python
detector = detection.load("dfine-x-coco", score_threshold=0.1)  # ByteTrack also uses low-score boxes
per_frame = detection.track_objects(video, detector, classes=["person"])  # one ObjectTracks per frame
per_frame[10].ids, per_frame[10].boxes  # [K] int64 track ids, [K, 4] (x0, y0, x1, y1) pixels

tracker = detection.ByteTracker(frame_rate=30)  # online: one update per frame
tracks = tracker.update(detector(frame))  # a DetectionResult, or [N, 5] (x0, y0, x1, y1, score)
```
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
from fastforks.detection._tracks import ByteTracker, ObjectTracks, track_objects

__all__ = [
    "Backend",
    "ByteTracker",
    "DetectionModel",
    "DetectionResult",
    "ImageLike",
    "ModelCard",
    "ObjectTracks",
    "SizeRange",
    "TensorRTEngine",
    "list_models",
    "load",
    "model_card",
    "track_objects",
]
