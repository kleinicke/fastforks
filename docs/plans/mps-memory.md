# MPS memory on a 16 GB Mac

The November 24 pass fitted 53 survey rows that ran out of memory on an M1 Pro 16 GB, at the survey size, with the
goldens passing: HAT, DRCT, GMFlow scale 2, FlowFormer small, FlowSeek-T, FILM, MegaFlow fp32, S2M2-XL fp32,
GIMM-VFI fp16, GREAT-IGEV and FoundationStereo ViT-L fp32. The tools are `fastforks._core.mps.release` (synchronize
plus `empty_cache` between big stages), `mps.attention` (query-blocked SDPA) and batch-blocked `grid_sample` in
`_core/sampling.py`. All of them are MPS, eager and no-grad only.

## Still out of memory or too slow

| Ref | dtype | Where | Lead |
|---|---|---|---|
| depthfm-v1 | both | completes, but 243-257 s (fp16) and 537 s (fp32) per call at 994x1482 | 32 MiB score blocks in the SD2 UNet gave 162 s and 8.43 GiB in fp16; `unet.MPS_SCORE_BYTES` is shared with the other SD2 models, so set it per model |
| gimm-vfi-r, gimm-vfi-r-p | fp32 | 12.1 GiB live in the full-resolution decoder; the fp32 volume pair is about 6.5 GiB | on-the-fly correlation lookups instead of volumes |
| memfof-kitti, memfof-sintel | fp32 | two all-pairs volumes with pyramids, about 10.4 GiB; OOM at `CorrBlock.corr` | on-the-fly lookups, or the volume in fp16 storage with fp32 accumulation |
| c-foundationstereo-s | fp32 | `corr_stem`: 10.81 GiB allocated, 1.06 GiB per conv3d tap | two attempts reverted; needs a disparity-blocked stem |
| waft-stereo-b | fp16 | three calls pass (12.9 GiB footprint), the survey OOMs on a 3.12 GiB request | locate the allocation; borderline |
| resplat-base-re10k | fp32 | the survey now runs 256x256, the committed rows are 480x640; the tool rejects a file with both sizes | drop the old rows and re-survey both dtypes at the new size |

cfnet-sceneflow was left out on purpose.

## Not checked yet

- MLX rows after the change for megaflow, s2m2, gimm-vfi and depthfm. These are vendored edits gated on MPS, so
  no change is expected.
- CUDA: nothing ran there; the claim of no change rests on the `device.type == "mps"` gates.
