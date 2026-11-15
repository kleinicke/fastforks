# HF card generator fixes (2026-11-15; follows results/2026-11-15_hf_plan_review.md)

Branch work/ports, tools/stage_hf.py, no upload, no token. Tests: tests/test_hf_cards.py, test_hf_names.py, test_card_sync.py,
test_policy.py, test_docs.py -> 1879 passed, 62 skipped (review/cpu_tests.sh --light).

1. Empty `Files: .` (was in every card with no vendored licence file): the line is now omitted ("Licence files: ..." only when there are some).
2. Licence headers: single known weights licence -> its HF id (added cc-by-sa-4.0, cc-by-sa-3.0, cc-by-nc-sa-3.0, cc-by-nc-nd-4.0, cc0-1.0, isc, zlib,
   unlicense, wtfpl, openrail++, creativeml-openrail-m). Otherwise `other` + plain `license_name`: `see-model-card` (assumed-from-code or compound
   terms; 54 cards), `mixed-see-variant-table` (variants differ; 9), or the single custom licence name (e.g. sam-license, noncommercial); `licenseref`
   and `-code-weights-unstated` slugs are gone. nnFoundation -> `cc-by-sa-4.0`.
3. lotus, stablenormal, marigold-normals: header `openrail++` + a body note on the SD2 OpenRAIL++-M use restrictions (SD2_DERIVED).
4. `base_model:` on 39 cards, only for families in AUTHOR_HF (weights fetched from the authors' own HF repo; no timm/community re-uploads, no backbones).
5. Tags `fastforks-card` (cards), `fastforks-weights` (weights repos, mirrors), `fastforks-kernels` (kernels).
6. tests/test_hf_names.py: weights repos, family repo names, kernels/mirrors/reference-outputs pairwise disjoint.
7. coex-sceneflow@1 left out of the goldens staging (EXCLUDED_GOLDENS); stale mc-stereo goldens are not in any golden.toml (nothing to drop here).
8. `stage_hf.py create CARDS_DIR [--dry-run] [--batch 20] [--pause 60]`: skips existing repos (resumable), 20 per round, pause between rounds,
   doubling wait on HTTP 429; needs HF_TOKEN except with --dry-run; never uploads files. Unit-tested with fakes (no network).

Near-collisions (not renamed): `seedvr2-3b` -> `seedvr2-3b-bf16` (DiT is bf16 + fp32 vectors; family card stays `seedvr2`),
`marigold-v2-depth` -> `marigold-v2-depth-<dtype of the converted file>` (e.g. `-fp16`; not in tools/weights_repos.toml, so check the live file).
Also `seedvr2-7b` -> `seedvr2-7b-bf16`, `seedvr2-vae` -> `seedvr2-vae-bf16` for consistency.
