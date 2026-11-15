# grounding_dino: deformable attention without copies (2026-11-15)
RTX 5060 Ti, torch 2.14.1+cu130, base f1198b34 + this change. Scripts: profile of cat/copy sources (CatArray + stack) found the
per-level samples being concatenated/stacked in transformers' MultiScaleDeformableAttention.
Change (general, shared): `detection/_attention.sample_maps` accumulates per level (no cat of the samples); grounding_dino
portable routes transformers' MSDA through `deformable_sample`.
Result (portable, 800x1344, tiny): -20 % time, -26 % peak memory vs before the change (earlier same-day rows).
Parity on COCO val2017 000000000139 at 480x640 (tiny, fp16, threshold 0.25): same detection counts (32 vs 32; 15 vs 15 above 0.4),
sorted scores within 0.015 (fp16 order noise), no COCO AP run, no local golden.
Kernels mode stays `none`: reason with numbers in card.toml (a fused MSDA cubin: Triton prototype 12.85 -> 0.84 ms per call, not built).
Existing unit test tests/test_optional_deps.py::test_requires_matches_family_code[detection-grounding_dino] fails ("code calls transformers", card requires ()); this change adds no import of it.
