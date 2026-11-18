# Known broken in 0.2.0, planned for 0.2.1

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
| `aliked-lightglue` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | torch.export stops at a data-dependent branch (masks.sum() == 0) | replace the data-dependent branch with a mask / fixed-size path (torch.cond or padding) | `matching/models/lightglue/card.toml` planned.onnx |
| `disk-lightglue` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | torch.export fails (TorchExportError, step 1/3 of the ONNX export) | run torch.export with strict=False, read the graph break, rewrite that op | `matching/models/lightglue/card.toml` planned.onnx |
| `alltracker` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | torch.export fails (TorchExportError, step 1/3 of the ONNX export) | run torch.export with strict=False, read the graph break, rewrite that op | `tracking/models/alltracker/card.toml` planned.onnx |
| `alltracker-tiny` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | torch.export fails (TorchExportError, step 1/3 of the ONNX export) | run torch.export with strict=False, read the graph break, rewrite that op | `tracking/models/alltracker/card.toml` planned.onnx |
| `cotracker3-online` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | torch.export fails (TorchExportError, step 1/3 of the ONNX export) | run torch.export with strict=False, read the graph break, rewrite that op | `tracking/models/cotracker/card.toml` planned.onnx |
| `track-on-r` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the export fails in the network call (ValueError: expected 4 values, got 2) | our bug: fix the export wrapper / example inputs (Portable.module) | `tracking/models/track_on/card.toml` planned.onnx |
| `track-on2` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | torch.export stops at a data-dependent branch (model.py:107) | replace the data-dependent branch with a mask / fixed-size path (torch.cond or padding) | `tracking/models/track_on/card.toml` planned.onnx |
| `loftr-indoor` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | torch.export fails (TorchExportError, step 1/3 of the ONNX export) | run torch.export with strict=False, read the graph break, rewrite that op | `matching/models/loftr/card.toml` planned.onnx |
| `loftr-outdoor` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | torch.export stops at a data-dependent branch (loftr.py:190) | replace the data-dependent branch with a mask / fixed-size path (torch.cond or padding) | `matching/models/loftr/card.toml` planned.onnx |
| `loma-b` export_tensorrt | raises 'planned for 0.2.1' | The bundled BF16 build loses 9.87 AUC@5 points on 50 calibrated MegaDepth pairs | Export ONNX and build the engine in a fresh process at 768×1024 | `matching/models/loma/card.toml` planned.tensorrt |
| `romav1` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the ONNX converter fails on the exported program (ConversionError, step 3/3) | add a decomposition or rewrite the op the dynamo ONNX converter lacks | `matching/models/romav1/card.toml` planned.onnx |
| `xfeat` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | torch.export stops at a data-dependent branch (NMS / keypoint selection) | replace the data-dependent branch with a mask / fixed-size path (torch.cond or padding) | `matching/models/xfeat/card.toml` planned.onnx |
| `xfeat-lighterglue` export_onnx / export_tensorrt | raises 'planned for 0.2.1' (was: fails) | torch.export stops at a data-dependent branch (NMS / keypoint selection) | replace the data-dependent branch with a mask / fixed-size path (torch.cond or padding) | `matching/models/xfeat/card.toml` planned.onnx |
| `dedode-v2-b` export_tensorrt | raises 'planned for 0.2.1' (was: fails) | TensorRT TopK supports K <= 3840; the keypoint top-k is larger | two-stage top-k (per tile, then merge) under K <= 3840 | `matching/models/dedode/card.toml` planned.tensorrt |
| `dedode-v2-g` export_tensorrt | raises 'planned for 0.2.1' (was: fails) | TensorRT TopK supports K <= 3840; the keypoint top-k is larger | two-stage top-k (per tile, then merge) under K <= 3840 | `matching/models/dedode/card.toml` planned.tensorrt |
| `anycalib-pinhole` export_tensorrt | raises 'planned for 0.2.1' | the engine builds but dataset-level camera accuracy acceptance is not yet established | establish and check an acceptance threshold before enabling | `calibration/models/anycalib/card.toml` planned.tensorrt |
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
| `bridgedepth-sf` export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the TensorRT 10 engine build fails (Myelin tactic check) | retry on a newer TensorRT 10 release; else split the attention block out of Myelin (fp32 island or plugin) | `stereo/models/bridgedepth/card.toml` planned.tensorrt |
| `bridgedepth-rvc-pretrain` export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the TensorRT 10 engine build fails (Myelin tactic check) | retry on a newer TensorRT 10 release; else split the attention block out of Myelin (fp32 island or plugin) | `stereo/models/bridgedepth/card.toml` planned.tensorrt |
| `defom-stereo-vitl-sceneflow` export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the TensorRT 10 engine build fails (Myelin tactic check) | retry on a newer TensorRT 10 release; else split the attention block out of Myelin (fp32 island or plugin) | `stereo/models/defom_stereo/card.toml` planned.tensorrt |
| `las2-h` export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the TensorRT 10 engine build fails (Myelin autotuner skips all tactics) | retry on a newer TensorRT 10 release; else split the attention block out of Myelin (fp32 island or plugin) | `stereo/models/las2/card.toml` planned.tensorrt |
| `gmflow-scale2-mixdata` export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the TensorRT 10 engine build fails (Myelin error on the attention graph) | retry on a newer TensorRT 10 release; else split the attention block out of Myelin (fp32 island or plugin) | `flow/models/gmflow/card.toml` planned.tensorrt |
| `gmflow-scale2-regrefine6-kitti` export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the TensorRT 10 engine build fails (Myelin error on the attention graph) | retry on a newer TensorRT 10 release; else split the attention block out of Myelin (fp32 island or plugin) | `flow/models/gmflow/card.toml` planned.tensorrt |
| `gmflow-scale2-regrefine6-mixdata` export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the TensorRT 10 engine build fails (Myelin error on the attention graph) | retry on a newer TensorRT 10 release; else split the attention block out of Myelin (fp32 island or plugin) | `flow/models/gmflow/card.toml` planned.tensorrt |
| `gmflow-scale2-regrefine6-sintelft` export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the TensorRT 10 engine build fails (Myelin error on the attention graph) | retry on a newer TensorRT 10 release; else split the attention block out of Myelin (fp32 island or plugin) | `flow/models/gmflow/card.toml` planned.tensorrt |
| `gmflow-scale2-regrefine6-things` export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the TensorRT 10 engine build fails (Myelin error on the attention graph) | retry on a newer TensorRT 10 release; else split the attention block out of Myelin (fp32 island or plugin) | `flow/models/gmflow/card.toml` planned.tensorrt |
| `gmflow-scale2-sintel` export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the TensorRT 10 engine build fails (Myelin error on the attention graph) | retry on a newer TensorRT 10 release; else split the attention block out of Myelin (fp32 island or plugin) | `flow/models/gmflow/card.toml` planned.tensorrt |
| `gmflow-scale2-things` export_tensorrt | raises 'planned for 0.2.1' (was: fails) | the TensorRT 10 engine build fails (Myelin error on the attention graph) | retry on a newer TensorRT 10 release; else split the attention block out of Myelin (fp32 island or plugin) | `flow/models/gmflow/card.toml` planned.tensorrt |
| `mlsd` export_tensorrt | raises 'planned for 0.2.1' | TensorRT 11 builds the graph; the isolated CNN yields line-map IoU of 0.45–0.66 against portable FP32 on five COCO images | locate the first differing CNN layer; isolated decoding and rasterization agree for the same inputs | `edges/models/mlsd/card.toml` planned.tensorrt |
| `waft-stereo-b` export_tensorrt | raises 'planned for 0.2.1' | the fp16 engine outputs all NaN. At 994x1482 the fp32 builder skips a tactic requesting 6.6 GiB under a 6 GiB cap | validate a smaller strongly typed fp32 engine | `stereo/models/waft_stereo/card.toml` planned.tensorrt |

## Not gated: fails with a clear error, or works with limits

| item | symptom | cause | fix idea | where |
|---|---|---|---|---|
| TRT/compile builds marked `deferred` | the benchmark cell says the build is over 5 min | the 5 min GPU lock rule; the build itself works | measure in a long-lock window (one-off), or cache the engines | `docs/data/speed_compile.jsonl` (`deferred`) |
| `memflow*`, `raft-large*`, `memfof-kitti`, `memfof-sintel`, `raft-small` (1080x1920), `hat-s-x4` export_tensorrt | the TRT builder runs out of GPU memory on 16 GB (`memfof-sintel` compile too) | the builder workspace plus large tactics | a smaller `workspace_gb`, or build on a 24 GB card; gate if it also fails there | speed table `fails:` cells |
| `seedvr2-7b` compile() | CUDA OOM while loading for compile | 7B weights plus the compile buffers > 16 GB | offload during compile, or compile per block | speed table |
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
| resplat `refine` vs `iters` | one option name differs from the other refinement models | card aliases are silent and permanent; a warned rename needs a deprecated-alias kind | add `deprecated_aliases` (FutureWarning, removal date) to the card option parser, then rename | `multiview/models/resplat/card.toml` |
| `@overload` return types | `load()` returns a generic `Model` to type checkers | no per-task overloads yet | `typing.overload` per task literal in `fastforks/__init__.py` | `__init__.py` |
| `StereoResult` properties | properties are typed as non-Optional, but some are `None` for some models; the repr shows `None` fields | the shared dataclass across models | type them `Optional`, and drop `None` fields from the repr | `stereo/_api.py` |
| `save()` keywords | `save(path)` is shared, but the extra keywords still differ per result type | grew per family | one keyword set for all results, with old names warning | result classes |
| verb table | the task verbs (`predict` / `segment` / `match` ...) are documented per page | no single table | one generated table in `docs/api.md` | `tools/gen_docs.py` |
| `labels=` (0.1.1 name of `class_names=`) | still accepted with a FutureWarning | migration rule | remove after 2026-12-15 (with the other renamed names) | `_core/arguments.py` `legacy_labels` |
| weak tests | several error paths are covered only by the usage probe | no unit tests yet | turn the probe cases (the 2026-11-15 API final review) into parametrized tests | `tests/` |
| `VolumeSession.add_point(zyx)` vs `VolumeInteractive(point=)` | the same point has two keyword names | added in separate steps | one name (`point`), the other a warned alias | `volume/_api.py` |
