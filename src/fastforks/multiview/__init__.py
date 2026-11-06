"""Multi-view depth: `load(model_id)` returns a callable `MultiViewModel` producing a `MultiViewResult`.

```python
from fastforks import multiview

model = multiview.load("vggt-1b", mode="kernels")
result = model(["a.jpg", "b.jpg", "c.jpg"])  # one scene; cameras estimated in the frame of view 0
result.depth, result.cam_to_world  # [3, H, W], [3, 4, 4]

model = multiview.load("mvsanywhere", mode="kernels")  # calibrated MVS: depth of the reference view
result = model(images, intrinsics=k, cam_to_world=c2w, reference=0)
```

Cameras: `intrinsics` are OpenCV pixel matrices (pixel centres at integer coordinates), `cam_to_world` camera-to-world
matrices with OpenCV axes (x right, y down, z forward); `MultiViewResult.extrinsics` gives world-to-camera `[R | t]`
(the VGGT / COLMAP convention).
"""

from fastforks._core.cards import ModelCard
from fastforks._core.export import TensorRTEngine
from fastforks._core.image import ImageLike
from fastforks._core.sizes import SizeRange
from fastforks.multiview._api import (
    Backend,
    MultiViewKind,
    MultiViewModel,
    MultiViewResult,
    list_models,
    load,
    model_card,
    unproject,
)

__all__ = [
    "Backend",
    "ImageLike",
    "ModelCard",
    "MultiViewKind",
    "MultiViewModel",
    "MultiViewResult",
    "SizeRange",
    "TensorRTEngine",
    "list_models",
    "load",
    "model_card",
    "unproject",
]
