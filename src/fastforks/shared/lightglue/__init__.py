"""LightGlue, the sparse feature matcher (cvg/LightGlue, vendored in `net/`).

`LightGlue(features=None, **conf)` builds the matcher; `load_weights(state_dict)` loads a checkpoint (the upstream
renames, strict apart from the unused `confidence_thresholds` buffer). The call takes upstream's dict of `image0` /
`image1` features (`keypoints` in pixels, `descriptors`, `image_size`) and returns its `matches` / `scores` lists.
`match_many(net, features)` matches many pairs in one packed forward (`batched.py`). `resize_long(image, long_side)`
is upstream's image resize before feature detection.
"""

from .batched import match_many
from .net.lightglue import LightGlue
from .resize import resize_long

__all__ = ["LightGlue", "match_many", "resize_long"]
