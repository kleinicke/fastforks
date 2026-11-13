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

Geometric verification for structure from motion, in one call on the result of any matcher (nothing is computed
before it is called), and many pairs of an image set:

```python
v = result.verify()  # sampled matches + RANSAC; verify(K1, K2) estimates the relative pose instead
v.points1, v.points2, v.scores  # [N, 2], [N, 2] pixels of the two input images, [N] confidence
v.inliers, v.geometry.F  # [N] bool inlier mask, the fundamental matrix (geometry.R, .t with intrinsics)

verified = {}
for (i, j), result in zip(pairs, model.match_pairs(paths, pairs)):  # pairs: (i, j) indices into paths
    v = result.verify()
    verified[i, j] = v.points1[v.inliers], v.points2[v.inliers], v.geometry.F  # keep these, drop the result
```

Keep only what the pipeline needs (points, mask, geometry) and let the dense result go: it holds the warp and the
certainty of every pixel on the GPU (about 144 MB for a 3000x4000 image), `v` holds neither them nor the model.

`match_pairs` computes each image's own part once (RoMa v2: its DINOv3 descriptor) and gives the same results as
single calls; for a stream of frames without a known pair list, RoMa v2 has the option `keep_descriptors=N`.

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
    VerifiedMatches,
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
    "VerifiedMatches",
    "estimate_absolute_pose",
    "estimate_fundamental",
    "estimate_pose",
    "list_models",
    "load",
    "model_card",
]
