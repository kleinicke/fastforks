# Fast CPU tier green (2026-11-15)

Commit: d7d66a3d (branch work/ports, on top of dc4c4c1d). Machine: Ryzen 5800X, CPU only (no GPU used).
Command: `review/cpu_tests.sh --light <worktree> fasttier5 tests -rfE -p no:cacheprovider` (6 GB cgroup, no `--slow`).

| run | passed | failed | skipped | time |
|---|---|---|---|---|
| before (dc4c4c1d) | 5444 | 24 | 6336 | 215 s |
| after (d7d66a3d) | 5468 | 0 | 6336 | 149 s |

## Fixes (root causes)
| test | cause | fix |
|---|---|---|
| test_patch_safety::test_convnext_portable_weight_grads | 9e927861 folded the layer scale into fc2 in place: fc2 / gamma grads no longer matched the original (B17-5) | product: parameters untouched; portable applies gamma in the residual `addcmul` (same passes, no copy); kernels path folds gamma into its own fc2 copy (`_w2_4`, `_b2_g`), so the 340 -> 260 MB saving stays |
| test_patch x2 | drct `type(m) is Conv2d`, cotracker bound-method `forward` | product: `is_base`, `override()` |
| test_compile[ocr] | ppocrv5 `Portable.__call__` neither compiled nor host | product: `@host` |
| test_meta_build | `torch.linspace` for drop path followed the meta default device (swin2sr, drct, ema_vfi) | product: `device="cpu"` |
| test_auto_mode::test_list_is_current | auto_portable.json stale | regenerated (tools/gen_auto_mode.py) |
| test_hf_cards x2 | nninteractive quickstart was not a `model(...)` call; the test's class lookup only knew promptable_segmentation | product: snippet `model(volume, point=...)`; test: uses each task's `_model_class` (volume has one too) |
| test_repo_hygiene | results/ citations and machine paths in cards, goldens, docs/plans, tools | product: dates instead of results/ paths, tools read `_data.DATA` / `FF_BENCH_HOME`, docs regenerated; test: results/ skipped (work-branch measurement log, records commands and paths by design; main has no results/) |
| test_pipeline x11 (full run only) | not a leak: the process-wide `fastforks-decode` pool (`_core.image._decoded`, by design like `fastforks-tiff`) is started by any earlier test that decodes two files | test: both persistent pools excluded |
| test_depth_models seeded noise x2 | test assumed the SD2 `Noise(shape, device, dtype)` call; marigold_v2 / seedvr2 have their own | test: per-family first draw checked exactly |
| test_result_to[StereoResult] | StereoResult keeps maps in `_by_view`; the test compared StereoViews with tensors (meta vs cpu) and saw no tensors | test: collects the views' tensors |
| test_pose_rfdetr adapter | fake network lacked upstream's `_export` flag (export once) | test: fake has the flag |

## Notes
- test_licences (datasets terms / pinned links) already passed on dc4c4c1d: no datasets.toml change needed.
- tests/test_contract.py passes. tests/test_devices.py does not exist (never in git history); pytest given a missing path collects nothing. Device coverage is in test_mps.py / test_mps_conv_cpu.py.
- Left: none in the fast tier. Slow and GPU tests not run here (release suite). ConvNeXt kernels/portable numerics changed by a rounding-level amount (gamma no longer folded in the portable GEMM): GPU goldens of wd-convnext-tagger to be re-checked in the next GPU suite.
- Pre-existing ruff debt (15 findings in other files) untouched.
