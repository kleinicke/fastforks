# Family smoke: default model of every family x every mode (2026-11-16)

Test: `tests/test_family_smoke.py` (gpu, model, slow; one param per model and mode; `test_every_family_is_covered` is fast). Script: `tools/family_smoke.py [--task T] [--part I/N] [--only ID] [--mode M] --out f.jsonl`.
Per model: load through the installed API (fresh venv, wheel copy of the worktree), one forward at a small size with the documented minimal input (`bench_inputs`, side >= 256 or the card minimum, 4 views for multi-view / tracking / interpolation), result must be a fastforks result with finite tensors. Modes: auto, kernels, portable.
Run: capped chunks (`cpuslot --gpu systemd-run -p MemoryMax=20G gpulock-low`), RTX 5060 Ti 16 GB, torch 2.14.1+cu130, base 31a83e5c + fixes, fixing commit bb1217d9. Raw: `results/raw/2026-11-16_family_smoke.jsonl` (first pass; the three re-runs below are in the table).

| result | cases |
|---|---|
| ok | 586 of 589 (20 families, 196 default models x 3 modes) |
| fail | 3: nafnet-gopro x 3 modes |

First pass: 10 failures, all explained:
- restoration gfpgan-v1.4, codeformer (6 cases): harness used a 512 x 640 input, the face restorers take exactly one square crop (clear InputError); harness now uses a square. Fixed, ok.
- classification tiny_vit_11m_224.dist_in22k_ft_in1k portable (1): real bug, `RuntimeError: Input type (cuda.HalfTensor) and weight type (HalfTensor)`. timm's re-parameterization (TinyViT `ConvNorm.fuse`) builds its fused convs on the CPU after `optimize` had moved the model. Fixed at the root in `_optimize/__init__.py` (`net.to(dev)` after `fold_bn`, for every conv-like family), ok in all modes.
- restoration nafnet-gopro (3, open): NaN in fp16 on some inputs (synthetic test image, uniform noise at 10 % contrast around grey); real photo fine; `dtype="float32"` fine. First non-finite value: encoders.3.13.conv4 with a finite input, in portable and kernels (the fp16-accumulate convolutions). Existing fp16 scaling (stream / 16, attention gate / 64) does not cover it. Not fixed: needs a per-layer range analysis in the deep-optimization scope (speed floor policy); the smoke test reports it as a failure on purpose until then.

Also found by the multi-view persona (see batch3): ReSplat with identity intrinsics (focal 1 px, principal point at the corner) gave NaN points in portable and a device-side assert in kernels mode (kills the CUDA context). Now refused with an InputError for any multi-view model (principal point must lie in the central 80 % of the image).
GPU pytest check: `pytest --slow -k "cotracker3 and kernels" tests/test_family_smoke.py` passes (needs FASTFORKS_WEIGHTS_PATH for the weight cache).
