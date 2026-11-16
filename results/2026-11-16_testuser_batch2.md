# Test users, batch 2 (2026-11-16, Claude Sonnet 5.5)

Setup: fresh venv `~/scratch_tests/tu-venv-b2` (uv, Python 3.12, torch cu130), fastforks installed from the worktree
`wt-tu-batch2` (base work/ports d15b384a), scripts in `~/scratch_tests/tu-b2/`, every run in a capped unit
(`cpuslot --gpu systemd-run -p MemoryMax=20G gpulock-low`). RTX 5060 Ti 16 GB, 2026-11-16.
Format: tried / expected / got / severity.

## Part A: DICOM (HIGH, fixed d15b384a)
- tried: TotalSegmentator on a pydicom volume (unflipped). expected: right organ volumes. got: liver 331 mL instead of 1096, silently.
- fix: `volume.read_dicom(folder_or_file)` returns volume + VolumeMeta in the `read_nifti` convention (sorted by ImagePositionPatient,
  HU via slope/intercept, spacing, LPS->RAS affine); `[dicom]` extra; docs/usage.md; test `test_read_dicom_matches_nifti_convention`.

## Persona 1: SAM
- Points as 0..1 fractions or x/y swapped (outside the image): accepted silently, wrong mask. medium. Fixed 67620035 (raise / warn), test in test_tasks.py.
- `result.numpy()` returns a dict, not a result object; batch with a shared prompt: error text "a tuple is shared" unclear. low, open.
- OK: wrong model name (suggestions), 0..255 float tensors (clear error), HWC tensors, bad labels, unknown kwargs, SAM 3 text, tiny / huge images, bool mask save.

## Persona 2: depth / video
- A video model (video-depth-anything) called with a list / batch gave independent frames (flicker), `.video()` undocumented. medium. Fixed c54b6d0b (warning + docs).
- metric3d-vit-large without intrinsics silently returns depth although docs say intrinsics are needed. medium, open (no card flag for it).
- No colorise helper (numpy recipe in docs); `result.save("d.png")` writes 16-bit PNG; float png via `fastforks.save` errors clearly; 1x1 / 5x5 give a clear "at least 8 px" error.

## Persona 3: detection
- No COCO-style output; `label` is a 0-based index into `det.class_names`, not a COCO category id. medium. Fixed c756ec00 (docs recipe: PIL drawing + records).
- Default score threshold 0.05 returns up to 100 spurious boxes. low, documented, docstring example uses 0.5.
- `classes=` / `iou_threshold=` rejected clearly. 0..255 float tensors rejected clearly. Open-vocabulary detectors need `[transformers]` (clear MissingDependencyError; not exercised); RF-DETR prints a long kernels note on load without transformers. low.

## Persona 4: SeedVR2 / super-resolution (16 GB)
- 1080p one pass with seedvr2-3b: raw CUDA OOM text without a way out. medium. Fixed cb4dcdea: the error names the input size and suggests `tile=512` (test in test_restoration_realesrgan.py). 2000x1500 with tile=512 works (peak 13.2 GB, 33 s).
- OK: 7b with stream (peak 2.7 GB, 90 s), 7b stream=False OOM (expected on 16 GB), 3b stream=False 15.1 GB, quant nf4 2.0 GB peak, typo / bad option messages, tile 16 / 128.5 rejected clearly, float 0..255 rejected clearly, gray / RGBA input fine, 4 px image clear error, odd sizes fine, clip list gives a list of results (one frame at a time, no `video`, so no temporal consistency: documented).

## Persona 5: tracking
- cotracker3 (default `auto` = kernels) crashed on the first call: "weight of shape [0]" (HIGH: the default path of two models). Cause: commit 080c8886 freed all update-former params incl. the LayerNorms read at call time. Fixed e701d00f (free only >1-D), GPU test `tests/test_tracking_cotracker.py` (slow, gpu, model). tapnext, tapir, bootstapir, locotrack, alltracker, track-on2, cotracker online, 200 frames, 1080p: all OK.
- Query positions outside the frame / in 0..1 fractions accepted silently. medium. Fixed (last commit "tracking: query positions ..."), test in test_objects_tracking.py.
- `TrackingResult` has no `save`; a single HWC image as video gives a confusing "at least 16 px, got (854, 3)". low, open.
- OK: THWC uint8 / (x,y,t) order / [Q,2] / t out of range: clear messages.

## Persona 6: ComfyUI pack (branch work/comfyui a236bd40, diverged from work/ports by 333 / 48 commits)
- Read docs/comfyui.md and nodes.py: node names / inputs / outputs match the docs; the pack uses `fastforks.load(id, mode=, device=, dtype=, **options)`, `restoration.load(path)`, `model_card(...).outputs/output_info/options`: all exist on work/ports. The pack's own tests (13) pass against work/ports. Weights: cache read in place, else `models/fastforks`, as documented. No GUI run.
- low, open: the Depth node lists video models (per-frame, flicker); the Segment node passes pixel points so the new point check is consistent with it.

## Extra personas (flow, stereo with kitti_000005 pair)
- flow: OK; `save` to .png errors clearly (.flo / .npy only), no colour-wheel helper outside ComfyUI. low.
- stereo: left/right swapped gives a plausible but wrong disparity silently (median 76 vs 1..100). low, inherent. `baseline=` kwarg rejected clearly (valid: rig, views, size, focal).
- face / keypoints: no `face` task in the library; pose lists dwpose / hrnet (not exercised).

## Tests
test_docs.py + test_policy.py + touched test files pass (1879 passed). Pre-existing, unrelated: `test_bad_spacing_nan_and_dicom_dir_give_clear_errors` fails only with a long tmp path (repr length).
