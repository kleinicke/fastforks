"""Matching: `load(model_id)` returns a callable `MatchingModel` giving a `MatchingResult` or `SparseMatchingResult`.

Dense models give a warp of every pixel, keypoint matchers (LightGlue) a `SparseMatchingResult`.

```python
from fastforks import matching

model = matching.load("romav2")
result = model("view_a.jpg", "view_b.jpg")  # the two images may differ in size
result.warp  # [2, H1, W1] float32: (x, y) in pixels of the second image for every pixel of the first
result.certainty  # [H1, W1] float32 in [0, 1]: how likely the pixel is visible in the second image

points1, points2 = result.matches(5000)  # [N, 2] pixel keypoints in each image, sampled where certain
pose = matching.estimate_pose(points1, points2, K1, K2)  # relative pose on the GPU: RANSAC on the essential matrix
pose.R, pose.t, pose.inliers  # camera 1 -> camera 2, unit translation, [N] inlier mask
fund = matching.estimate_fundamental(points1, points2)  # unknown intrinsics: 7-point RANSAC, fund.F, fund.inliers
pose = matching.estimate_absolute_pose(points2d, points3d, K)  # 2D-3D matches: P3P RANSAC, x_cam = R X + t

sparse = matching.load("aliked-lightglue")("view_a.jpg", "view_b.jpg")
sparse.keypoints1, sparse.keypoints2  # [N, 2], [M, 2] detected keypoints (x, y) in pixels of each image
sparse.pairs, sparse.scores  # [K, 2] int64 index pairs (as COLMAP / pycolmap take them), [K] confidence
points1, points2 = sparse.matches()  # the matched points, for estimate_pose
```

`estimate_pose` is batched (`[P, N, 2]` points, `[P, 3, 3]` intrinsics: all pairs in one call) and has a `portable`
(plain PyTorch) and a `kernels` (CUDA) backend like the models; `mode="auto"` picks the kernels when they load.
"""

from fastforks._core.cards import ModelCard
from fastforks._core.export import TensorRTEngine
from fastforks._core.image import ImageLike
from fastforks._core.sizes import SizeRange
from fastforks.matching._api import (
    Backend,
    MatchingModel,
    MatchingResult,
    SparseMatchingResult,
    list_models,
    load,
    model_card,
)
from fastforks.matching.absolute_pose import AbsolutePose, estimate_absolute_pose
from fastforks.matching.relative_pose import FundamentalMatrix, RelativePose, estimate_fundamental, estimate_pose

__all__ = [
    "AbsolutePose",
    "Backend",
    "FundamentalMatrix",
    "ImageLike",
    "MatchingModel",
    "MatchingResult",
    "ModelCard",
    "RelativePose",
    "SizeRange",
    "SparseMatchingResult",
    "TensorRTEngine",
    "estimate_absolute_pose",
    "estimate_fundamental",
    "estimate_pose",
    "list_models",
    "load",
    "model_card",
]
