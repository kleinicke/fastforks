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

Feed-forward splatting models (e.g. `anysplat`) also return `result.gaussians` (`Gaussians`; `.save("scene.ply")`).
"""

from fastforks._core.arguments import renamed_attributes
from fastforks._core.shared_types import OLD_TASK_NAMES
from fastforks.multiview._api import (
    MultiViewKind,
    MultiViewModel,
    MultiViewResult,
    list_models,
    load,
    model_card,
    unproject,
)
from fastforks.multiview._gaussians import Gaussians

__all__ = [
    "Gaussians",
    "MultiViewKind",
    "MultiViewModel",
    "MultiViewResult",
    "list_models",
    "load",
    "model_card",
    "unproject",
]

__getattr__ = renamed_attributes(__name__, OLD_TASK_NAMES)  # the shared types: `fastforks.<Name>`
