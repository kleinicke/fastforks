# Structure from motion

`fastforks.matching.sfm.colmap_database` turns an image folder into a COLMAP database: image retrieval picks the pairs,
a sparse matcher matches them, a batched fundamental-matrix RANSAC on the GPU verifies them. The mapper (COLMAP or
GLOMAP) then builds the reconstruction.

```bash
fastforks sfm scene/images scene/database.db            # MegaLoc pairs (k=20), ALIKED + LightGlue, GPU RANSAC
colmap mapper --database_path scene/database.db --image_path scene/images --output_path scene/sparse
fastforks sfm scene/images scene/out --reconstruct      # both steps: out/database.db + out/sparse/0 (pycolmap)
```

```python
from fastforks.matching import sfm

stats = sfm.colmap_database("scene/images", "scene/database.db", matcher="disk-lightglue", k=30)
print(stats.verified, stats.seconds)  # per stage: retrieval, matching (incl. decoding and verification), database
rec = sfm.reconstruct("scene/images", "scene/out", k=30)  # + pycolmap's mapper; rec.model = scene/out/sparse/0
print(rec.registered, rec.points, rec.seconds)
```

`reconstruct` and `--reconstruct` need pycolmap (`pip install 'fastforks-core[sfm]'`); `mapper="global"` /
`--mapper global` uses GLOMAP's global mapper (pycolmap >= 3.12) instead of the incremental one.

- `--retrieval none` matches all pairs (exhaustive); `--k` sets the pairs per image.
- `--single-camera` shares one camera (intrinsics) among the images of the same size; by default every image gets its
  own camera, as in COLMAP.
- `--threshold` (default 4 px, as COLMAP's verification) is the inlier threshold of the RANSAC. Against 1 px it
  costs <= 0.6 AUC@3 on Mip-NeRF 360 garden/counter and gains 6.6 on ETH3D facade (6048 x 4032 images).
- `matcher_options` passes load options to the matcher: `{"long_side": 1600}` detects features at up to 1600 px
  instead of 1024 (garden AUC@3 95.1 -> 96.3, counter 95.4 -> 96.2; +16 % front-end time, 2x matching memory).
- Only sparse matchers fit (keypoints per image); dense ones (RoMa, LoFTR) are refused.
- The database uses the COLMAP 3 schema; COLMAP 4 / pycolmap 4 read it.

## Speed (RTX 5060 Ti, front end only)

Mip-NeRF 360 garden (185 images, 1/4 resolution), image folder to database, incl. model loading:

| pipeline | front end | then incremental mapper (registered, AUC@3) |
|---|---|---|
| fastforks (MegaLoc, ALIKED + LightGlue, GPU RANSAC) | 30 s | 185/185, 95.1 |
| fastforks, `matcher_options={"long_side": 1600}` | 35 s | 185/185, 96.3 |
| hloc original (NetVLAD, SuperPoint + LightGlue, COLMAP verification) | 81 s | 185/185, 97.6 |
| COLMAP original (CPU SIFT, exhaustive) | 589 s | 185/185, 97.8 |
