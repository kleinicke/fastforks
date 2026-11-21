# Group B, Windows 11 portable mode: test ledger

RTX 5060 Ti 16 GB (sm_120), torch 2.14.1+cu130 (no flash SDPA). One row per test in `groupB_windows_portable.jsonl` (last run wins); failures carry `category` and `note`.

| task | pass | fail | error | skip |
|---|---|---|---|---|
| (family test files) | 321 | 51 | 0 | 18 |
| classification | 541 | 13 | 0 | 106 |
| detection | 169 | 6 | 0 | 52 |
| interpolation | 49 | 0 | 0 | 10 |
| matting | 74 | 2 | 0 | 16 |
| ocr | 20 | 0 | 0 | 7 |
| pose | 139 | 19 | 0 | 44 |
| promptable_segmentation | 77 | 5 | 0 | 32 |
| restoration | 158 | 1 | 0 | 30 |
| segmentation | 111 | 6 | 0 | 25 |

| failure category | count |
|---|---|
| kernels-not-on-windows | 71 |
| pre-existing | 12 |
| numeric | 10 |
| fixed | 8 |
| golden-file | 1 |
| skipped-big | 1 |

## Failures other than kernels mode

- `tests.test_weights_arg::test_every_family_loads_user_weights_strictly[dinov2_vitl14_lc@1-add]` (fixed): OSError: [WinError 1314] Dem Client fehlt ein erforderliches Recht: 'C:\\Users\\neuer\\fastforks-cache\\blobs\\sha256\\04\\0424a5d1b515278cba3c6640ccbeaacc41de5
- `tests.test_weights_arg::test_every_family_loads_user_weights_strictly[dinov2_vitl14_lc@1-drop]` (fixed): OSError: [WinError 1314] Dem Client fehlt ein erforderliches Recht: 'C:\\Users\\neuer\\fastforks-cache\\blobs\\sha256\\04\\0424a5d1b515278cba3c6640ccbeaacc41de5
- `tests.test_weights_arg::test_every_family_loads_user_weights_strictly[dinov2_vitl14_lc@1-shape]` (fixed): OSError: [WinError 1314] Dem Client fehlt ein erforderliches Recht: 'C:\\Users\\neuer\\fastforks-cache\\blobs\\sha256\\04\\0424a5d1b515278cba3c6640ccbeaacc41de5
- `tests.test_export::test_module_compile_export[efficientformer_l1.snap_dist_in1k@1]` (fixed): IndexError: list index out of range
- `tests.test_export::test_module_compile_export[efficientformerv2_s0.snap_dist_in1k@1]` (fixed): IndexError: list index out of range
- `tests.test_export::test_module_compile_export[efficientvit_b2.r288_in1k@1]` (fixed): IndexError: list index out of range
- `tests.test_export::test_module_compile_export[shvit_s1.in1k@1]` (fixed): IndexError: list index out of range
- `tests.test_golden::test_cpu_portable[wd-vit-tagger-v3@1]` (numeric): AssertionError: wd-vit-tagger-v3@1 vs golden:
- `tests.test_golden::test_gpu[grounding-dino-base@1-portable]` (golden-file): AssertionError: C:\Users\neuer\fastforks-goldens\grounding-dino-base@1.npz does not match golden.toml
- `tests.test_export::test_module_compile_export[lw-detr-tiny-coco@1]` (pre-existing): RuntimeError: shape '[1, 4, 1, 4, 2, 192]' is invalid for input of size 9216
- `tests.test_golden::test_cpu_portable[rf-detr-seg-n-coco@1]` (numeric): AssertionError: rf-detr-seg-n-coco@1 vs golden:
- `tests.test_golden::test_gpu[rf-detr-n-coco@1-portable]` (numeric): AssertionError: call 1: boxes: p99 |diff| 0.2300
- `tests.test_golden::test_gpu[rf-detr-seg-2xl-coco@1-portable]` (numeric): AssertionError: call 1: boxes: p99 |diff| 2.1559
- `tests.test_weights_arg::test_every_family_loads_user_weights_strictly[rtmdet-nano-person@1-shape]` (pre-existing): RuntimeError: The size of tensor a (257) must match the size of tensor b (256) at non-singleton dimension 0
- `tests.test_golden::test_cpu_portable[ben2@1]` (numeric): AssertionError: ben2@1 vs golden:
- `tests.test_export::test_module_compile_export[vitmatte-s@1]` (pre-existing): TypeError: _Matting.module.<locals>.<lambda>() missing 1 required positional argument: 'trimap'
- `tests.test_weights_arg::test_every_family_loads_user_weights_strictly[hrnet-w32@1-add]` (pre-existing): AssertionError: Regex pattern did not match.
- `tests.test_weights_arg::test_every_family_loads_user_weights_strictly[hrnet-w32@1-shape]` (pre-existing): RuntimeError: The size of tensor a (257) must match the size of tensor b (256) at non-singleton dimension 0
- `tests.test_golden::test_cpu_portable[rf-detr-keypoint-preview@1]` (numeric): AssertionError: rf-detr-keypoint-preview@1 vs golden:
- `tests.test_golden::test_gpu[rf-detr-keypoint-preview@1-portable]` (numeric): AssertionError: call 1: keypoints: mean |diff| 0.0749
- `tests.test_export::test_module_compile_export[rtmo-t@1]` (pre-existing): AssertionError: rtmo-t@1: ['keypoints_0', 'keypoints_1', 'keypoints_2', 'keypoints_3'] are not declared by module()
- `tests.test_weights_arg::test_every_family_loads_user_weights_strictly[rtmo-t@1-shape]` (pre-existing): RuntimeError: The size of tensor a (385) must match the size of tensor b (384) at non-singleton dimension 0
- `tests.test_export::test_module_compile_export[rtmpose-t@1]` (pre-existing): RuntimeError: mat1 and mat2 shapes cannot be multiplied (17x12 and 48x256)
- `tests.test_weights_arg::test_every_family_loads_user_weights_strictly[rtmpose-t@1-shape]` (pre-existing): RuntimeError: The size of tensor a (385) must match the size of tensor b (384) at non-singleton dimension 0
- `tests.test_weights_arg::test_every_family_loads_user_weights_strictly[insid3-vitl16@1-add]` (pre-existing): Failed: DID NOT RAISE CheckpointMismatchError
- `tests.test_weights_arg::test_every_family_loads_user_weights_strictly[insid3-vitl16@1-drop]` (pre-existing): Failed: DID NOT RAISE CheckpointMismatchError
- `tests.test_weights_arg::test_every_family_loads_user_weights_strictly[insid3-vitl16@1-shape]` (pre-existing): Failed: DID NOT RAISE CheckpointMismatchError
- `tests.test_golden::test_cpu_portable[sam-vit-b@1]` (numeric): AssertionError: sam-vit-b@1 vs golden:
- `tests.test_golden::test_cpu_portable[sam3@1]` (numeric): AssertionError: sam3@1 vs golden:
- `tests.test_golden::test_cpu_portable[dat-light-x4@1]` (numeric): AssertionError: dat-light-x4@1 vs golden:
- `tests.test_golden::test_gpu[sapiens-seg-1b-goliath@1-portable]` (skipped-big): MemoryError
- `tests.test_export::test_module_compile_export[segnext-mscan-t-ade20k@1]` (fixed): UserWarning: A model attribute `net.backbone.block1.0.attn.res_scale` requires gradient but it's not properly registered as a parameter. torch.export will detac

## Numeric deviations (goldens and tolerances unchanged)

- GPU fp16, rf-detr: this Windows torch has no flash SDPA (mem-efficient / cuDNN kernels instead).
  rf-detr-n-coco boxes p99 0.2300 (tol 0.23); rf-detr-seg-2xl-coco boxes p99 2.156 (tol 1.8);
  rf-detr-keypoint-preview keypoints mean 0.0749 (tol 0.072). With only the math SDPA backend n and seg-2xl pass.
- CPU float32 just over atol: rf-detr-seg-n-coco boxes max 2.14e-4 (atol 1e-4, 3 of 16, values to 597);
  rf-detr-keypoint-preview keypoints max 2.14e-4 (1 of 34); sam-vit-b pred_masks max 2.68e-3 (atol 2e-3, 23 px);
  sam3 mixed_boxes max 1.38e-3 (atol 1e-3, 2 of 800); ben2 alpha max 7.15e-6 (atol 3.5e-6, 10 px);
  dat-light-x4 image max 3.86e-6 (atol 3.6e-6, 4 px); wd-vit-tagger-v3 logits max 2.1e-5 (atol 1.5e-5, 1 of 10861).

## Not covered

- seedvr2 (22.7 GB, over the 15 GB staging cap), siglip (not on the SSD, 1.6 GB), sapiens-seg-1b (4.5 GB download).
- grounding-dino-base: the local golden npz does not match the sha in golden.toml (re-fetch the golden).

## Bench smoke

`groupB_windows_bench.jsonl`: `fastforks bench <smallest ref> --mode portable --runs 3`, one per family, all exit 0
(gfpgan / codeformer at `--size 512x512`: the default 375x1242 is below their 512 px minimum).
