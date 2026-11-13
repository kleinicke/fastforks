# Structure from motion

`fastforks.matching.sfm.colmap_database` turns an image folder into a COLMAP database: image retrieval picks the pairs,
a sparse matcher matches them, a batched fundamental-matrix RANSAC on the GPU verifies them. The mapper (COLMAP or
GLOMAP) then builds the reconstruction.

```bash
fastforks sfm scene/images scene/database.db            # MegaLoc pairs (k=20), ALIKED + LightGlue, GPU RANSAC
colmap mapper --database_path scene/database.db --image_path scene/images --output_path scene/sparse
```

```python
from fastforks.matching import sfm

stats = sfm.colmap_database("scene/images", "scene/database.db", matcher="disk-lightglue", k=30)
print(stats.verified, stats.seconds)  # per stage: retrieval, matching (incl. decoding and verification), database
```

- `--retrieval none` matches all pairs (exhaustive); `--k` sets the pairs per image.
- `--single-camera` shares one camera (intrinsics) among the images of the same size; by default every image gets its
  own camera, as in COLMAP.
- Only sparse matchers fit (keypoints per image); dense ones (RoMa, LoFTR) are refused.
- The database uses the COLMAP 3 schema; COLMAP 4 / pycolmap 4 read it.

## Speed (RTX 5060 Ti, front end only)

Mip-NeRF 360 garden (185 images, 1/4 resolution), image folder to database, incl. model loading:

| pipeline | front end | then incremental mapper (registered, AUC@3) |
|---|---|---|
| fastforks (MegaLoc, ALIKED + LightGlue, GPU RANSAC) | 38 s | 185/185, 95.2 |
| hloc original (NetVLAD, SuperPoint + LightGlue, COLMAP verification) | 81 s | 185/185, 97.6 |
| COLMAP original (CPU SIFT, exhaustive) | 589 s | 185/185, 97.8 |
