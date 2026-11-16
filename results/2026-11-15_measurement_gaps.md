# Measurement gaps, 2026-11-15

Matrix of every model ref of the library (452 refs, 143 families, from `registry.cards`) against `docs/data/speed.jsonl` and `docs/data/speed_compile.jsonl` (side `lib`). Per-ref csv: `results/raw/2026-11-15_gaps/gaps.csv` (made by `gapmatrix.py`; plan by `plan.py`). Commit of the data: `git log -1 -- docs/data` at the time of writing.

## Counts

| column | ok | missing | error / OOM |
|---|---|---|---|
| upstream | 447 | 5 | 0 |
| upstream-default | 387 | 65 | 0 |
| portable | 448 | 4 | 0 |
| kernels | 384 | 68 | 0 |
| compile | 402 | 48 | 2 |
| tensorrt | 367 | 49 | 36 |

## (a) No original (upstream / upstream-default) speed + peak-memory row

Every ref has an `upstream` OR `upstream-default` row except 5 (and every ok original row carries peak memory). The 5 need an upstream clone that is not on disk (a download, not allowed inside the queue lock):

- `memfof-kitti` (memfof): needs `FF_MEMFOF_REPO` (clone of msu-video-group/memfof)
- `memfof-sintel` (memfof): needs `FF_MEMFOF_REPO` (clone of msu-video-group/memfof); portable row added (375x1242 ms, 1080x1920 OOM; see results/2026-11-15_original_gaps.md)
- `memfof-spring` (memfof): needs `FF_MEMFOF_REPO` (clone of msu-video-group/memfof); also no portable row yet
- `tapnext-pp` (tapnext): needs the `tapnet` repo; also no portable row yet
- `video-depth-anything-metric-s` (video_depth_anything): no original adapter in `tools/_original.py`; also no portable row yet

Four of them (memfof-sintel, memfof-spring, tapnext-pp, video-depth-anything-metric-s) have no portable row either: `tools/speed_check.py` builds its jobs from the floor's portable rows, so they cannot be queued before their owner records portable / kernels rows (`tools/speed_survey.py`-style).

## (b) No TensorRT row: 49 refs

Of these, 11 have cached weights (queued), 38 do not (weights would have to be downloaded: `weights_not_cached.txt` lists all 224 refs without cached weights). Missing:

`aanet-plus-sceneflow`, `bridgedepth-rvc-pretrain`, `clip-h14-378-dfn5b`, `clip-h14-dfn5b`, `clip-l14-dfn2b`, `dedode-v2-g`, `depth-anything-v2-b`, `detr-r101-coco`, `dinov2-vitb14-reg`, `dinov2-vitg14`, `dinov2-vitg14-reg`, `dinov2-vitl14-reg`, `dinov2-vits14-reg`, `dinov3-convnext-base`, `dinov3-convnext-small`, `dinov3-vitb16`, `dinov3-vith16plus`, `dinov3-vitl16`, `dlnr-middlebury`, `fast-acvnet-kitti12`, `fast-acvnet-kitti15`, `fast-acvnet-plus-generalization`, `fast-acvnet-plus-kitti12`, `fast-acvnet-plus-kitti15`, `fast-acvnet-plus-sceneflow`, `foundationstereo-vit-l`, `great-igev-middlebury`, `gwcnet-gc-sceneflow`, `igev-stereo-kitti12`, `igev-stereo-kitti15`, `loftr-indoor`, `memfof-sintel`, `memfof-spring`, `mocha-v1-sceneflow`, `moge-3-vitl`, `rtdetrv2-l-coco`, `rtdetrv2-m-coco`, `rtdetrv2-ms-coco`, `rtdetrv2-x-coco`, `s2m2-m`, `selective-igev-eth3d-train`, `selective-igev-kitti12`, `selective-igev-kitti15`, `selective-igev-middlebury-finetune`, `selective-igev-middlebury-train`, `selective-igev-sceneflow`, `tapnext-pp`, `video-depth-anything-metric-s`, `waft-stereo-b`

## (c) No compile row: 48 refs

10 with cached weights (queued), 38 without. Missing:

`aanet-plus-sceneflow`, `bridgedepth-rvc-pretrain`, `clip-h14-378-dfn5b`, `clip-h14-dfn5b`, `clip-l14-dfn2b`, `dedode-v2-g`, `depth-anything-v2-b`, `detr-r101-coco`, `dinov2-vitb14-reg`, `dinov2-vitg14`, `dinov2-vitg14-reg`, `dinov2-vitl14-reg`, `dinov2-vits14-reg`, `dinov3-convnext-base`, `dinov3-convnext-small`, `dinov3-vitb16`, `dinov3-vith16plus`, `dinov3-vitl16`, `dlnr-middlebury`, `fast-acvnet-kitti12`, `fast-acvnet-kitti15`, `fast-acvnet-plus-generalization`, `fast-acvnet-plus-kitti12`, `fast-acvnet-plus-kitti15`, `fast-acvnet-plus-sceneflow`, `foundationstereo-vit-l`, `gwcnet-gc-sceneflow`, `igev-stereo-kitti12`, `igev-stereo-kitti15`, `loftr-indoor`, `memfof-sintel`, `memfof-spring`, `mocha-v1-sceneflow`, `moge-3-vitl`, `rtdetrv2-l-coco`, `rtdetrv2-m-coco`, `rtdetrv2-ms-coco`, `rtdetrv2-x-coco`, `s2m2-m`, `selective-igev-eth3d-train`, `selective-igev-kitti12`, `selective-igev-kitti15`, `selective-igev-middlebury-finetune`, `selective-igev-middlebury-train`, `selective-igev-sceneflow`, `tapnext-pp`, `video-depth-anything-metric-s`, `waft-stereo-b`

## (d) Rows that are errors / OOM / NaN

No OOM rows exist in the original columns (all 447 `upstream` rows have ms + memory). Failed rows are all in the TRT / compile files:

TensorRT, 36 refs (real failures of the export / engine build, recorded as `fails`; none is a by-design reason such as original-fp32 OOM; they are TRT-export limits of the model and stay documented as failures, not retried):

| ref | reason |
|---|---|
| `aliked-lightglue` | TorchExportError: Failed to export the model with torch.export. [96mThis is step 1/3[0m  |
| `alltracker` | ValueError: not enough values to unpack (expected 4, got 2) |
| `alltracker-tiny` | ValueError: not enough values to unpack (expected 4, got 2) |
| `anycalib-pinhole` | ConversionError: Failed to decompose the FX graph for ONNX compatibility. [96mThis is ste |
| `bootstapir` | ValueError: not enough values to unpack (expected 4, got 2) |
| `bridgedepth-sf` | RuntimeError: TensorRT engine build failed for /tmp/ff-trt-971_3_p2/model.onnx (see the Te |
| `dedode-v2-b` | timeout after 298 s |
| `defom-stereo-vitl-sceneflow` | RuntimeError: TensorRT cannot parse /tmp/ff-trt-oyg7bm3t/model.onnx: In node 1299 with nam |
| `dfine-x-coco` | wrong output: an engine output is all NaN |
| `disk-lightglue` | TorchExportError: Failed to export the model with torch.export. [96mThis is step 1/3[0m  |
| `geocalib-pinhole` | TorchExportError: Failed to export the model with torch.export. [96mThis is step 1/3[0m  |
| `gmflow-scale2-mixdata` | BackendUnavailableError: TensorRT engine build failed: IBuilder::buildSerializedNetwork: E |
| `gmflow-scale2-regrefine6-kitti` | BackendUnavailableError: TensorRT engine build failed: IBuilder::buildSerializedNetwork: E |
| `gmflow-scale2-regrefine6-mixdata` | BackendUnavailableError: TensorRT engine build failed: IBuilder::buildSerializedNetwork: E |
| `gmflow-scale2-regrefine6-sintelft` | BackendUnavailableError: TensorRT engine build failed: IBuilder::buildSerializedNetwork: E |
| `gmflow-scale2-regrefine6-things` | BackendUnavailableError: TensorRT engine build failed: IBuilder::buildSerializedNetwork: E |
| `gmflow-scale2-sintel` | BackendUnavailableError: TensorRT engine build failed: IBuilder::buildSerializedNetwork: E |
| `gmflow-scale2-things` | BackendUnavailableError: TensorRT engine build failed: IBuilder::buildSerializedNetwork: E |
| `grounding-dino-base` | ConversionError: Failed to convert the exported program to an ONNX model. [96mThis is ste |
| `grounding-dino-tiny` | ConversionError: Failed to convert the exported program to an ONNX model. [96mThis is ste |
| `las2-h` | BackendUnavailableError: TensorRT engine build failed: Error Code: 9: Skipping tactic 0x00 |
| `loftr-outdoor` | TorchExportError: Failed to export the model with torch.export. [96mThis is step 1/3[0m  |
| `memfof-kitti` | BackendUnavailableError: TensorRT engine build failed: [defaultAllocator.cpp::allocate::62 |
| `mm-grounding-dino-base` | ConversionError: Failed to convert the exported program to an ONNX model. [96mThis is ste |
| `mm-grounding-dino-tiny` | ConversionError: Failed to convert the exported program to an ONNX model. [96mThis is ste |
| `mvsanywhere` | RuntimeError: TensorRT cannot parse /tmp/ff-trt-_iwkthez/model.onnx: In node 0 with name:  |
| `raft-large` | BackendUnavailableError: TensorRT engine build failed: [defaultAllocator.cpp::allocate::62 |
| `raft-large-kitti` | BackendUnavailableError: TensorRT engine build failed: [defaultAllocator.cpp::allocate::62 |
| `raft-large-things` | BackendUnavailableError: TensorRT engine build failed: [defaultAllocator.cpp::allocate::62 |
| `raft-small` | BackendUnavailableError: TensorRT engine build failed: [defaultAllocator.cpp::allocate::62 |
| `romav1` | ConversionError: Failed to convert the exported program to an ONNX model. [96mThis is ste |
| `stablenormal-turbo` | timeout after 290 s |
| `tapir` | ValueError: not enough values to unpack (expected 4, got 2) |
| `tapnext` | ValueError: not enough values to unpack (expected 4, got 2) |
| `xfeat` | ConversionError: Failed to decompose the FX graph for ONNX compatibility. [96mThis is ste |
| `xfeat-lighterglue` | TorchExportError: Failed to export the model with torch.export. [96mThis is step 1/3[0m  |

Compile, 2 refs:

- `c-foundationstereo-s`: run: AssertionError: Guard failed on the same frame it was created. This is a bug - please (torch dynamo guard bug; real failure)
- `romav1`: worker rc=124 (cold compile exceeded the 290 s step limit; a real gap, needs a longer unlocked-lock slot)

## (e) Kernels

`kernels` row missing for 68 refs. 15 families (64 refs) declare `[modes.kernels] status = "none"` with a reason in their card (by design); families without a declared reason: none. Partly missing:

- `memfof`: memfof-sintel, memfof-spring (new ports without a floor row)
- `tapnext`: tapnext-pp (new ports without a floor row)
- `video_depth_anything`: video-depth-anything-metric-s (new ports without a floor row)

By-design families: `coatnet`, `davit`, `dedode`, `efficientformer`, `efficientformerv2`, `flowformer`, `flowseek`, `focalnet`, `gmflow`, `grounding_dino`, `hgnetv2`, `lightglue`, `loftr`, `shvit`, `xfeat`

## Start-cost columns

Cold start (no engine / no cache) exists for all ok rows: TRT `onnx_export_s` + `engine_build_s` (367 rows), compile `compile_s` = first call on empty Inductor / Triton caches (402 rows). The START WITH AN ALREADY COMPILED MODEL did not exist in any row (0 of 367 TRT, 0 of 402 compile rows). Added to `tools/speed_check.py` (138e4308): `engine_load_s` + `engine_first_call_s` (load of the cached engine file and its first call) in every TRT row, `compile_warm_s` (second start on the filled caches, `--warm-start`) in every compile row.

## Queue (unit `gapfill-trt`)

`results/raw/2026-11-15_gaps/queue.sh`: (1) originals skipped (above), (2) 7 TRT, (3) 6 compile gaps with cached weights into `speed_compile_new.jsonl`, merged into `docs/data/speed_compile.jsonl` (new keys only) and pushed after `tests/test_docs.py tests/test_policy.py tests/test_speed_floor.py` pass (`publish.sh`), (4) re-measure of the 193 TRT / 216 compile refs that have cached weights with the start columns, into `backfill_trt.jsonl` / `backfill_compile.jsonl` (existing rows untouched; the owner decides how to merge). Log: `queue.log`, `queue_steps.log`. Every step: `gpulock-verylow timeout 175`, a failed step is logged and the queue goes on.
