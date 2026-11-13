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

Multi-object tracking by detection (ByteTrack on any `fastforks.detection` model; details in `tracking.objects`):

```python
from fastforks import detection, tracking

detector = detection.load("dfine-x-coco", options={"score_threshold": 0.1})  # ByteTrack also uses low-score boxes
per_frame = tracking.track_objects(video, detector, labels=[0])  # one ObjectTracks per frame (COCO class 0: person)
per_frame[10].ids, per_frame[10].boxes  # [K] int64 track ids, [K, 4] (x0, y0, x1, y1) pixels

tracker = tracking.ByteTracker(frame_rate=30)  # online: one update per frame
tracks = tracker.update(detector(frame))  # a DetectionResult, or [N, 5] (x0, y0, x1, y1, score)
```
"""

from fastforks._core.cards import ModelCard
from fastforks._core.export import TensorRTEngine
from fastforks._core.image import ImageLike
from fastforks._core.sizes import SizeRange
from fastforks.tracking._api import Backend, TrackingModel, TrackingResult, list_models, load, model_card
from fastforks.tracking.objects import ByteTracker, ObjectTracks, track_objects

__all__ = [
    "Backend",
    "ByteTracker",
    "ImageLike",
    "ModelCard",
    "ObjectTracks",
    "SizeRange",
    "TensorRTEngine",
    "TrackingModel",
    "TrackingResult",
    "list_models",
    "load",
    "model_card",
    "track_objects",
]
