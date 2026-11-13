"""LightGlue, the sparse feature matcher (cvg/LightGlue, vendored in `net/`).

`LightGlue(features=None, **conf)` builds the matcher; `load_weights(state_dict)` loads a checkpoint (the upstream
renames, strict apart from the unused `confidence_thresholds` buffer). The call takes upstream's dict of `image0` /
`image1` features (`keypoints` in pixels, `descriptors`, `image_size`) and returns its `matches` / `scores` lists.
`resize_long(image, long_side)` is upstream's image resize before feature detection.
"""

from .net.lightglue import LightGlue
from .resize import resize_long

__all__ = ["LightGlue", "resize_long"]
