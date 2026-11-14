# Structure from motion

`fastforks sfm` turns an image folder into camera poses and a sparse point cloud, the way hloc or COLMAP's feature
pipeline do: image retrieval picks the image pairs, a matcher finds corresponding points, a GPU RANSAC verifies them,
and the COLMAP mapper (pycolmap) builds the reconstruction.

```bash
fastforks sfm scene/images scene/database.db              # COLMAP database: MegaLoc pairs (k=20), ALIKED + LightGlue
colmap mapper --database_path scene/database.db --image_path scene/images --output_path scene/sparse
fastforks sfm scene/images scene/out --reconstruct        # both steps: out/database.db + out/sparse/0
fastforks sfm scene/images scene/out --reconstruct --matcher loma-b   # the more accurate matcher (below)
```

```python
from fastforks.matching import sfm

stats = sfm.colmap_database("scene/images", "scene/database.db", matcher="loma-b", k=30)
print(stats.verified, stats.seconds)        # per stage: retrieval, matching (incl. verification), database
rec = sfm.reconstruct("scene/images", "scene/out", k=30)   # + pycolmap's mapper; rec.model = scene/out/sparse/0
print(rec.registered, rec.points, rec.seconds)
```

`reconstruct` and `--reconstruct` need pycolmap (`pip install 'fastforks-core[sfm]'`); `--mapper global` uses
GLOMAP's global mapper (pycolmap >= 3.12) instead of the incremental one. `--retrieval none` matches all pairs,
`--k` sets the pairs per image, `--single-camera` shares one set of intrinsics among images of the same size,
`--threshold` is the RANSAC inlier threshold (4 px, as COLMAP). The database uses the COLMAP 3 schema; COLMAP 4 and
pycolmap 4 read it. Dense matchers (`romav2`, `romav1`, `loftr`) also work (`dense_matches` most certain pixels
per pair, like hloc's `match_dense`) but take 10x longer and are not in the tables below.

## Which matcher

Nine scenes, each run from the image folder to the reconstruction on an RTX 5060 Ti with the pycolmap incremental
mapper (16 CPU threads). AUC@3 is the share of camera pairs whose relative pose is within 3 degrees of the reference
(100 = perfect), total time = matching front end + mapper incl. model loading, peak memory = GPU memory reserved by
the front end. Where a scene was run more than once the range is given in parentheses; differences inside such a
range are ties.

### Mip-NeRF 360 (125-311 photos of one object or room, 1/4 resolution)

Reference poses: the scenes' COLMAP reconstructions.

| scene | front end | AUC@3 | registered images | total time | peak memory | commercial use |
|---|---|---|---|---|---|---|
| garden | `aliked-lightglue` (default) | 95.1 | 185/185 | 126 s | 1370 MB | yes |
| garden | `disk-lightglue` | 97.6 | 185/185 | 121 s | 2506 MB | yes |
| garden | `loma-b` | 97.7 | 185/185 | 197 s | 3320 MB | yes |
| garden | `aliked-flexglue` | 95.2 | 185/185 | 279 s | 2418 MB | yes, DINOv3 licence terms |
| counter | `aliked-lightglue` (default) | 95.6 | 240/240 | 223 s | 1426 MB | yes |
| counter | `disk-lightglue` | 96.1 | 240/240 | 250 s | 2438 MB | yes |
| counter | `loma-b` | 98.2 | 240/240 | 286 s | 2376 MB | yes |
| counter | `aliked-flexglue` | 95.7 | 240/240 | 398 s | 2100 MB | yes, DINOv3 licence terms |
| stump | `aliked-lightglue` (default) | 93.7 | 125/125 | 67 s | 1384 MB | yes |
| stump | `disk-lightglue` | 96.1 | 125/125 | 67 s | 2530 MB | yes |
| stump | `loma-b` | 97.1 | 125/125 | 108 s | 2376 MB | yes |
| stump | `aliked-flexglue` | 94.5 | 125/125 | 181 s | 2140 MB | yes, DINOv3 licence terms |
| bicycle | `aliked-lightglue` (default) | 93.1 | 194/194 | 135 s | 1422 MB | yes |
| bicycle | `disk-lightglue` | 92.8 | 194/194 | 143 s | 2568 MB | yes |
| bicycle | `loma-b` | 94.4 | 194/194 | 197 s | 2392 MB | yes |
| bicycle | `aliked-flexglue` | 92.8 | 194/194 | 284 s | 2492 MB | yes, DINOv3 licence terms |
| kitchen | `aliked-lightglue` (default) | 91.4 | 279/279 | 287 s | 1432 MB | yes |
| kitchen | `disk-lightglue` | 87.3 | 279/279 | 290 s | 2446 MB | yes |
| kitchen | `loma-b` | 92.8 | 279/279 | 380 s | 2376 MB | yes |
| kitchen | `aliked-flexglue` | 92.0 | 279/279 | 482 s | 2124 MB | yes, DINOv3 licence terms |
| room | `aliked-lightglue` (default) | 92.9 | 311/311 | 214 s | 1530 MB | yes |
| room | `disk-lightglue` | 94.1 | 311/311 | 282 s | 2580 MB | yes |
| room | `loma-b` | 95.4 | 311/311 | 365 s | 2680 MB | yes |
| room | `aliked-flexglue` | 93.6 | 311/311 | 494 s | 2454 MB | yes, DINOv3 licence terms |

### ETH3D high-res (38-76 DSLR photos, 6048 x 4032, laser-scan reference poses)

| scene | front end | AUC@3 | registered images | total time | peak memory | commercial use |
|---|---|---|---|---|---|---|
| facade | `aliked-lightglue` (default) | 62.6 | 76/76 | 121 s | 2324 MB | yes |
| facade | `disk-lightglue` | 44.2 | 76/76 | 132 s | 2974 MB | yes |
| facade | `loma-b` | 59.0 (58.8-59.3) | 76/76 | 233 s | 3418 MB | yes |
| facade | `aliked-flexglue` | 71.0 | 76/76 | 261 s | 3100 MB | yes, DINOv3 licence terms |
| courtyard | `aliked-lightglue` (default) | 25.4 | 38/38 | 58 s | 2286 MB | yes |
| courtyard | `disk-lightglue` | 28.4 | 38/38 | 62 s | 2936 MB | yes |
| courtyard | `loma-b` | 77.0 | 38/38 | 85 s | 3416 MB | yes |
| courtyard | `aliked-flexglue` | 58.5 (58.0-59.1) | 38/38 | 126 s | 3098 MB | yes, DINOv3 licence terms |
| delivery area | `aliked-lightglue` (default) | 37.9 (35.3-40.5) | 40/44 | 53 s | 2292 MB | yes |
| delivery area | `disk-lightglue` | 46.6 | 40/44 | 61 s | 2944 MB | yes |
| delivery area | `loma-b` | 82.0 (80.8-83.2) | 44/44 | 101 s | 3416 MB | yes |
| delivery area | `aliked-flexglue` | 52.1 (50.1-54.1) | 44/44 | 118 s | 3098 MB | yes, DINOv3 licence terms |

### Mean over the scenes

| front end | Mip-NeRF 360 AUC@3 (6 scenes) | ETH3D AUC@3 (3 scenes) | total time Mip / ETH3D | peak memory | commercial use |
|---|---|---|---|---|---|
| `aliked-lightglue` (default) | 93.6 | 41.9 | 175 s / 77 s | 1.5 / 2.3 GB | yes |
| `disk-lightglue` | 94.0 | 39.7 | 192 s / 85 s | 2.6 / 3.0 GB | yes |
| `loma-b` | 95.9 | 72.7 | 256 s / 140 s | 2.7 / 3.4 GB | yes |
| `aliked-flexglue` | 93.9 | 60.5 | 353 s / 169 s | 2.5 / 3.1 GB | yes, DINOv3 licence terms |

Recommendation:

- **`aliked-lightglue`** (the default) is the fastest and smallest. It registers every image on 8 of the 9 scenes and
  reaches 91-96 AUC@3 on the Mip-NeRF 360 scenes.
- **`loma-b`** is the most accurate: better than the default on 8 of 9 scenes (+1 to +3 AUC@3 on Mip-NeRF 360, +15 to
  +52 on the two wide-baseline ETH3D scenes, where it is the only one with correct poses; its one loss, facade, is
  3.6 points). The cost is about 2x the front-end time (1.5x the total with the mapper), 1.5-1.8x the GPU memory and
  2.9 GB of weights. Use it when accuracy matters more than a minute of extra time.
- `disk-lightglue` ties the default overall (better on 6 scenes, worse on 3, by -18 to +2.5). `aliked-flexglue` ties the default on
  Mip-NeRF 360 and is the best on facade, but 5x slower; its DINOv3 backbone weights carry Meta's DINOv3 licence
  terms.
- These are 6 indoor / garden scenes and 3 outdoor building scenes; other image sets can rank differently. Six of
  the nine scenes were run once; on the repeated ones the results move by up to 1 (facade) to 5 (delivery area)
  AUC@3 between runs, so differences of that size on the ETH3D scenes are not meaningful.

The default compute dtype (bfloat16 for `loma-b`) loses no SfM accuracy: in float32 the poses are the same within
the run-to-run range (garden 97.7 vs 97.7, facade 59.0 vs 59.8) at 2x the front-end time.

Details, every single run and the noise measurements: `results/2026-11-14_sfm_scenes.md` and
`results/2026-11-13_sfm_lightglue.md` in the experiment repository.
