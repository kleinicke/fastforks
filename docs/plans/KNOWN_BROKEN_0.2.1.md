# 0.2.0 limitations and 0.2.1 measurement follow-up

## Current measurement status

The November 19 review found valid TensorRT timings for 37 of the 40 variants in the original export follow-up.
Thirty-six of the selected timings use standalone builds. The [benchmark tables](../benchmarks.md) give the input
shape and stage for each measurement. The model cards give the current availability of the public export methods.
The [post-release checklist](postrelease-todo.md) records coverage for the full catalog.

| Variant | Remaining TensorRT accuracy limit |
|---|---|
| `disk-lightglue` | The staged CNN lost 1.92 pose AUC@5 points in default FP32 and 4.13 with TF32 disabled on 50 MegaDepth pairs. |
| `xfeat-lighterglue` | The staged CNN caused a new pose solver failure on one of 300 MegaDepth pairs. Disabling TF32 and running just the first LighterGlue block in TensorRT both reproduced the failure. |
| `romav1` | The staged DINOv2 and fine VGG trials caused new pose solver failures or large individual pose errors on MegaDepth, including with TF32 disabled. |

## Original 0.2.0 tracking notes

The sections below record the symptoms and proposed work at the time of the 0.2.0 release.

Everything here is either gated or fails with a clear error. Gated features raise
`BackendUnavailableError: <ref>: <feature> is not available in 0.2.0, planned for 0.2.1: <reason>`.
They are not advertised in the docs. The model pages carry a "Not in this release" line, and the benchmark cells say `planned for 0.2.1`.

**Mechanism (one for all).** A card gets `planned = { <feature> = "<reason>" }`: a family-level `[planned]` table, or a per-variant entry. Both are merged into `Variant.planned`.
- Features: `load`, `kernels`, `compile`, `onnx`, `tensorrt`. A gate on `onnx` also blocks `tensorrt`.
- One check, `fastforks._core.cards.require_available(variant, feature)`, runs before any work in:
  - `load_variant` (load / kernels);
  - `Model.compile`;
  - `export_onnx` / `export_tensorrt` (before the size check);
  - `check_exportable`.
- `mode="auto"` skips gated kernels silently.
- To activate a feature in 0.2.1: fix it, delete its `planned` line, rerun the row's check, and `tools/gen_docs.py`.

## Gated (card `planned`)

| item | symptom | cause | fix idea | gate location |
|---|---|---|---|---|
| `aliked-lightglue` export_onnx / export_tensorrt | raises 'planned for 0.2.1' | Whole export stops at a data-dependent branch. At 768 x 1024, the staged FP32 encoder took 57.276 ms versus 42.410 ms portable and lost 1.01 pose AUC@5 points across 300 MegaDepth pairs | compare the dense score maps before keypoint selection | `matching/models/lightglue/card.toml` planned.onnx and planned.tensorrt |
| `disk-lightglue` export_onnx / export_tensorrt | raises 'planned for 0.2.1' | Whole export stops at variable keypoint counts. A staged FP32 DISK CNN at 768 x 1024 took 286.993 ms versus 116.955 ms portable and lost 1.92 pose AUC@5 points across 50 MegaDepth pairs | compare heatmap and descriptor outputs before sparse selection | `matching/models/lightglue/card.toml` planned.onnx and planned.tensorrt |
| `alltracker` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | torch.export fails (TorchExportError, step 1/3 of the ONNX export) | run torch.export with strict=False, read the graph break, rewrite that op | `tracking/models/alltracker/card.toml` planned.onnx |
| `alltracker-tiny` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | torch.export fails (TorchExportError, step 1/3 of the ONNX export) | run torch.export with strict=False, read the graph break, rewrite that op | `tracking/models/alltracker/card.toml` planned.onnx |
| `cotracker3-online` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | torch.export fails (TorchExportError, step 1/3 of the ONNX export) | run torch.export with strict=False, read the graph break, rewrite that op | `tracking/models/cotracker/card.toml` planned.onnx |
| `track-on-r` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | was: the export failed in the network call (ValueError: expected 4 values, got 2); now (2026-11-24, torch.export non-strict) it stops at the data-dependent query start frames in `track` (model.py:117), as track-on2 | as track-on2: a fixed-size loop over all frames with the queries masked until their start frame | `tracking/models/track_on/card.toml` planned.onnx |
| `track-on2` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | torch.export stops at a data-dependent branch (the query start frames, model.py:117) | replace the data-dependent branch with a mask / fixed-size path (torch.cond or padding) | `tracking/models/track_on/card.toml` planned.onnx |
| `loftr-indoor` export_tensorrt | raises 'planned for 0.2.1' | the fixed plan loses 5.27 AUC@20 points in FP16 and 2.58 AUC@10 points in FP32 on 50 ScanNet pairs | use the verified FP32 ONNX graph in ONNX Runtime | `matching/models/loftr/card.toml` planned.tensorrt |
| `loftr-outdoor` export_onnx / export_tensorrt | raises 'planned for 0.2.1' | torch.export stops at a data-dependent branch. At fixed 768 x 1024, pose AUC@5 on 50 MegaDepth pairs drops by 4.29 points in FP16 and 2.49 in FP32 | compare fine match coordinates before pose estimation | `matching/models/loftr/card.toml` planned.onnx and planned.tensorrt |
| `loma-b` export_tensorrt | raises 'planned for 0.2.1' | The bundled BF16 build loses 9.87 AUC@5 points on 50 calibrated MegaDepth pairs | Export ONNX and build the engine in a fresh process at 768×1024 | `matching/models/loma/card.toml` planned.tensorrt |
| `romav1` export_onnx / export_tensorrt | raises 'planned for 0.2.1' | Whole-model ONNX conversion fails in step 3/3. The staged FP32 DINOv2 and fine VGG engines each introduce a new pose failure on MegaDepth with TF32 disabled | compare the differences in refined warps and pose inliers on the failing pairs | `matching/models/romav1/card.toml` planned.onnx and planned.tensorrt |
| `xfeat` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | torch.export stops at a data-dependent branch (NMS / keypoint selection) | replace the data-dependent branch with a mask / fixed-size path (torch.cond or padding) | `matching/models/xfeat/card.toml` planned.onnx |
| `xfeat-lighterglue` export_onnx / export_tensorrt | raises 'planned for 0.2.1' | Whole-model export stops at sparse keypoint selection. A staged FP32 CNN produces a new pose failure on MegaDepth pair 95 with or without TF32 | compare sparse correspondence selection on the failing pair | `matching/models/xfeat/card.toml` planned.onnx and planned.tensorrt |
| `dedode-v2-b` export_tensorrt | raises 'planned for 0.2.1' | FP16 loses 2.54 pose AUC@5 points on 300 MegaDepth pairs | partitioned TopK builds successfully. The fixed FP32 plan has paired quality and timing at 768 x 1024 | `matching/models/dedode/card.toml` planned.tensorrt |
| `dedode-v2-g` export_tensorrt | raises 'planned for 0.2.1' | At fixed 768 x 1024, pose AUC@5 on 300 MegaDepth pairs drops by 4.08 points in FP16 and 1.61 in default FP32. FP32 with TF32 disabled gains 1.41 points and introduces a solver failure on a pair that portable solves at 12.88 degrees. | The strict FP32 descriptor-only plan keeps the PyTorch detector, keypoint order and matching. It has a paired 300-case score and timing. | `matching/models/dedode/card.toml` planned.tensorrt |
| `geocalib-pinhole` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | dynamic optimizer convergence and unsupported ONNX Cholesky decomposition | export a convergence-preserving loop and a supported solve | `calibration/models/geocalib/card.toml` planned.onnx |
| `resplat-small-dl3dv` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | torch.export cannot capture data-dependent Gaussian renderer shapes | make Gaussian rendering exportable without data-dependent shapes | `multiview/models/resplat/card.toml` planned.onnx |
| `grounding-dino-base` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the ONNX converter has no decomposition for aten._is_all_true | add a decomposition or rewrite the op the dynamo ONNX converter lacks | `detection/models/grounding_dino/card.toml` planned.onnx |
| `grounding-dino-tiny` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the ONNX converter has no decomposition for aten._is_all_true | add a decomposition or rewrite the op the dynamo ONNX converter lacks | `detection/models/grounding_dino/card.toml` planned.onnx |
| `mm-grounding-dino-base` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the ONNX converter has no decomposition for aten._is_all_true | add a decomposition or rewrite the op the dynamo ONNX converter lacks | `detection/models/grounding_dino/card.toml` planned.onnx |
| `mm-grounding-dino-tiny` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the ONNX converter has no decomposition for aten._is_all_true | add a decomposition or rewrite the op the dynamo ONNX converter lacks | `detection/models/grounding_dino/card.toml` planned.onnx |
| `marigold-v2-depth` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the NF4 transformer expands its weights per call; no ONNX path yet | export a dequantized fp16 transformer (needs > 16 GB) or skip ONNX for this model | `depth/models/marigold_v2/card.toml` planned.onnx |
| `pi3` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the ONNX converter fails on the exported program (ConversionError, step 3/3) | add a decomposition or rewrite the op the dynamo ONNX converter lacks | `multiview/models/pi3/card.toml` planned.onnx |
| `omni-dc` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | torch.export fails (TorchExportError, step 1/3 of the ONNX export) | run torch.export with strict=False, read the graph break, rewrite that op | `depth/models/omni_dc/card.toml` planned.onnx |
| `moge-3-vitl` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | torch.export fails (TorchExportError, step 1/3 of the ONNX export) | run torch.export with strict=False, read the graph break, rewrite that op | `depth/models/moge/card.toml` planned.onnx |
| `nnfoundation-vit` export_onnx / export_tensorrt | raises 'planned for 0.2.1' | the 192³ graph exports and builds, but mean relative L2 error against portable FP16 is 0.0469 on eight CT crops | compare both backends to the original FP32 encoder on the same crops | `volume/models/nnfoundation/card.toml` planned.onnx |
| `bridgedepth-sf` export_tensorrt | raises 'planned for 0.2.1' | TensorRT 11.4 still fails the Myelin tactic check at 994×1482 | isolate the failing attention block | `stereo/models/bridgedepth/card.toml` planned.tensorrt |
| `bridgedepth-rvc-pretrain` export_tensorrt | raises 'planned for 0.2.1' | TensorRT 11.4 still fails the Myelin tactic check at 994×1482 and 375×1242 | isolate the failing attention block | `stereo/models/bridgedepth/card.toml` planned.tensorrt |
| `defom-stereo-vitl-sceneflow` export_tensorrt | raises 'planned for 0.2.1' (was: fails) | TensorRT 11.4 still fails the Myelin tactic check after export at 994 x 1482 | check whether the attention block can be exported as a separate stage | `stereo/models/defom_stereo/card.toml` planned.tensorrt |
| `las2-h` export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the TensorRT 10 engine build fails (Myelin autotuner skips all tactics) | retry on a newer TensorRT 10 release; else split the attention block out of Myelin (fp32 island or plugin) | `stereo/models/las2/card.toml` planned.tensorrt |
| `gmflow-scale2-mixdata` export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the TensorRT 10 engine build fails (Myelin error on the attention graph) | retry on a newer TensorRT 10 release; else split the attention block out of Myelin (fp32 island or plugin) | `flow/models/gmflow/card.toml` planned.tensorrt |
| `gmflow-scale2-regrefine6-kitti` export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the TensorRT 10 engine build fails (Myelin error on the attention graph) | retry on a newer TensorRT 10 release; else split the attention block out of Myelin (fp32 island or plugin) | `flow/models/gmflow/card.toml` planned.tensorrt |
| `gmflow-scale2-regrefine6-mixdata` export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the TensorRT 10 engine build fails (Myelin error on the attention graph) | retry on a newer TensorRT 10 release; else split the attention block out of Myelin (fp32 island or plugin) | `flow/models/gmflow/card.toml` planned.tensorrt |
| `gmflow-scale2-regrefine6-sintelft` export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the TensorRT 10 engine build fails (Myelin error on the attention graph) | retry on a newer TensorRT 10 release; else split the attention block out of Myelin (fp32 island or plugin) | `flow/models/gmflow/card.toml` planned.tensorrt |
| `gmflow-scale2-regrefine6-things` export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the TensorRT 10 engine build fails (Myelin error on the attention graph) | retry on a newer TensorRT 10 release; else split the attention block out of Myelin (fp32 island or plugin) | `flow/models/gmflow/card.toml` planned.tensorrt |
| `gmflow-scale2-sintel` export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the TensorRT 10 engine build fails (Myelin error on the attention graph) | retry on a newer TensorRT 10 release; else split the attention block out of Myelin (fp32 island or plugin) | `flow/models/gmflow/card.toml` planned.tensorrt |
| `gmflow-scale2-things` export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the TensorRT 10 engine build fails (Myelin error on the attention graph) | retry on a newer TensorRT 10 release; else split the attention block out of Myelin (fp32 island or plugin) | `flow/models/gmflow/card.toml` planned.tensorrt |
| `mlsd` export_tensorrt | raises 'planned for 0.2.1' | Splitting the TensorRT 11 FP32 CNN after its first or second feature map still gives raster-map IoU between 0.18 and 0.66 against portable FP32 on five COCO images | isolate the remaining CNN discrepancies; decoding and rasterization agree given the same logits | `edges/models/mlsd/card.toml` planned.tensorrt |
| `waft-stereo-b` export_tensorrt | raises 'planned for 0.2.1' | the fp16 engine outputs all NaN. At 994x1482 the fp32 builder skips a tactic requesting 6.6 GiB under a 6 GiB cap | validate a smaller strongly typed fp32 engine | `stereo/models/waft_stereo/card.toml` planned.tensorrt |

## Not gated: fails with a clear error, or works with limits

| item | symptom | cause | fix idea | where |
|---|---|---|---|---|
| TRT/compile builds marked `deferred` | the benchmark cell says the build is over 5 min | the 5 min GPU lock rule; the build itself works | measure in a long-lock window (one-off), or cache the engines | `docs/data/speed_compile.jsonl` (`deferred`) |
| `memflow*`, `raft-large*`, `memfof-kitti`, `memfof-sintel`, `raft-small` (1080x1920), `hat-s-x4` export_tensorrt | the TRT builder runs out of GPU memory on 16 GB (`memfof-sintel` compile too) | the builder workspace plus large tactics | a smaller `workspace_gb`, or build on a 24 GB card; gate if it also fails there | speed table `fails:` cells |
| `seedvr2-7b` compile() | CUDA OOM while loading the complete graph for compile | 7B weights plus the compile buffers > 16 GB | standalone VAE-only compile with streamed transformer is measured on 25 DIV2K crops | SeedVR2 model page |
| `memflow-kitti` TensorRT at 384×1248 | the FP16 engine fails the labeled accuracy check | on ten KITTI pairs, EPE is 4.672 px versus 0.413 px in portable FP16 | isolate the divergent operators on the same pairs | MemFlow model page |
| `gimm-vfi-r` TensorRT at 480×854 | the FP32 engine returns numerically different frames | mean pixel error against portable FP32 is 0.07687 versus 0.000029 for ONNX Runtime on the same graph | isolate TensorRT operators by comparing intermediate outputs | GIMM-VFI model page |
| `pointdit-b-512` export_tensorrt | not verified | the sweep venv had no `onnx` | verify with the export extra | the 0.2.1 export sweep |
| `seedvr2-3b` export with host streaming | export fails during an in-place update of an inference tensor | the default streamed graph moves weights during inference | use `stream=False` and separate export/build for the validated 64 x 64 BF16 engine | SeedVR2 model page |
| `resplat-base` weights | the third-party HF repo is gated; the error says how to get access (login via `huggingface_hub`) | licence gate upstream | none needed (works after login); mirror if the licence allows | `multiview/models/resplat/card.toml` |
| kernels binaries `raft_flow`, `romav2` | `mode="kernels"` works locally; elsewhere `mode="auto"` falls back to portable | the binaries are not uploaded yet (HF upload at release) | upload at release prep | `kernels/` builds, HF repos |
| `matanyone`, `totalsegmentator` export | `BackendUnavailableError`: no exportable module | by design (recurrent memory / multi-model pipeline) | none planned | family `export` = none |
| families with kernels `status = "none"` | `mode="kernels"` raises with the family's reason; `auto` uses portable | no custom kernels for these yet | the per-family kernel backlog | card `[kernels]` |
| volume NIfTI example | `MissingDependencyError` names `pip install 'fastforks-core[nifti]'` | `nibabel` is an optional extra | none needed | `volume/_nifti.py` |

## API leftovers (proposal v2, deferred: risky or needs a mechanism)

| item | symptom | cause | fix idea | where |
|---|---|---|---|---|
| `@overload` return types | `fastforks.load()` returns a generic `Model` to type checkers | it takes a model id, not a task, so an overload would need a `Literal` of every id (hundreds, changing with each card) | decided (2026-11-24): no; `fastforks.<task>.load` already returns the precise class and the `load` docstring says so | `__init__.py` |
| `save()` keywords | resolved (checked 2026-11-24 against tests/api_snapshot.txt) | every map result (depth, stereo views and pairs, `MapOutputs`, multiview) takes `save(path, key, bits, *, max_depth, axes)`, multiview adds `view=`; flow, interpolation, restoration, volume, instances and Gaussians take `save(path)` alone | none | result classes |
| `labels=` (0.1.1 name of `class_names=`) | still accepted with a FutureWarning | migration rule | remove after 2026-12-15 (with the other renamed names) | `_core/arguments.py` `legacy_labels` |
