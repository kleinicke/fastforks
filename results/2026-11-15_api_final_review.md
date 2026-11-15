# API final review + gate for known-broken features (2026-11-15)

- Branch: `attempt/api-final`, on `work/ports` 79daa547. Commits:
  - 7d91021f: the gate;
  - d39d2f48: the card gates and docs;
  - fd8df2e5: export gate order, volume path error;
  - be41c667: removed the stray `.pt`;
  - f61eab94: the notes, changelog and snapshot.
- Machine: CPU only, in capped units (`systemd-run -p MemoryMax=6G`, `review/cpu_tests.sh --light`). torch 2.14.1, Ubuntu 26.04, 2026-11-15.
- Probe: `tools/examples.py` RENDER per task on the smallest model by weight size, then each misuse below. Direct calls were added for matting / restoration, whose example code does not forward bad inputs.
  - Raw logs: 220 + 26 JSON lines, kept in the session scratchpad (not tracked).
  - Each check lists the error text in the table below.

## Usage probe (CPU)

| task | checks | example runs | unclear errors |
|---|---|---|---|
| stereo | 14 | ok | 0 |
| depth | 11 | ok | 0 |
| classification | 11 | ok | 0 |
| segmentation | 11 | ok | 0 |
| matting | 18 | ok | 0 |
| edges | 11 | ok | 0 |
| promptable_segmentation | 11 | ok | 0 |
| detection | 11 | ok | 0 |
| ocr | 11 | ok | 0 |
| pose | 11 | ok | 0 |
| features | 11 | ok | 0 |
| retrieval | 11 | ok | 0 |
| calibration | 11 | ok | 0 |
| multiview | 13 | ok | 0 |
| flow | 12 | ok | 0 |
| matching | 13 | ok | 0 |
| tracking | 11 | ok | 0 |
| interpolation | 11 | ok | 0 |
| restoration | 18 | ok | 0 |
| volume | 15 | needs nibabel (clear `MissingDependencyError`, names `fastforks-core[nifti]`) | 1, fixed |

**Checks per task.**
- Inputs:
  - wrong channels, 1-D array, complex / int64 dtype;
  - `None`, missing file.
- Options and modes:
  - unknown option: lists the valid ones;
  - bad `mode`, bad `dtype`;
  - model typo: did-you-mean.
- Backends and weights:
  - `mode="kernels"` on CPU: names `mode='auto'`;
  - no-kernels reason;
  - missing weights.
- Task-specific:
  - stereo / flow size mismatch;
  - list length mismatch;
  - multiview missing cameras / views.

All errors were clear and named the fix, except two, both fixed in fd8df2e5:
1. `export_onnx` / `export_tensorrt` on a gated model first asked for an export size. The gate now runs first.
2. A path given to a volume model raised `TypeError: new(): invalid data type 'str'`. It now raises `InputError` naming `fastforks.volume.read_nifti(path)` and `meta=`.

## Gate (one mechanism)

- The card field is `planned = { load | kernels | compile | onnx | tensorrt = "reason" }`, at family level (`[planned]`) or per variant.
- The check is `_core.cards.require_available`. It runs in:
  - `load_variant`;
  - `mode="kernels"`;
  - `Model.compile`;
  - `export_onnx`;
  - `export_tensorrt`;
  - `check_exportable`.
  `mode="auto"` skips gated kernels.
- Message: `<ref>: export_onnx is not available in 0.2.0, planned for 0.2.1: <reason>; the portable and kernels backends run as usual`.
- Gated: 52 variants in 34 cards. Each gate is backed by a failure recorded in `docs/data/speed_compile.jsonl` or by the probe:
  - onnx: 31 (this also blocks tensorrt);
  - tensorrt only: 19;
  - compile: 4 (54 entries).
- Docs:
  - benchmark cells say `planned for 0.2.1`;
  - model pages get a "Not in this release" line.
- `gen_docs --check` passes.
- No working path was removed.
- Item list, causes and fix ideas: `docs/plans/KNOWN_BROKEN_0.2.1.md`.

## Decisions (proposal v2, section 8; recorded in 017 DECISIONS.md §16)

- Migration rule unchanged: renamed names warn until 2026-12-15 (DECISIONS §15; `mode=` kept).
- Deferred to 0.2.1, because each needs a new mechanism or touches every family:
  - resplat `refine` -> `iters` (card aliases are silent, so a warned alias kind is needed first);
  - per-task `@overload`s;
  - Optional `StereoResult` properties;
  - one `save()` keyword set;
  - a generated verb table;
  - `VolumeSession.add_point(zyx)` vs `point=`.
- API snapshot updated (+30 lines, all additions from merged volume / depth / ocr work; no removals).

## Tests

- `tests/test_planned.py` (13), `tests/test_volume.py` (+2), `tests/test_export.py`: 45 passed, CPU light slot.
- `tools/select_tests.py origin/work/ports HEAD`:
  - fast tier `tests`, run on CPU in the light slot: this branch before the stand-in fix had 27 failed / 5441 passed; base 79daa547 has 25 failed / 5429 passed. The 2 extra (`test_sizes`) pass after the fix (rerun of all failed files: 13 failed both on base and branch).
    - The failures are the same set on both, all pre-existing: `test_pipeline` x11 (order-dependent; they pass alone), `test_hf_cards` x2, `test_patch` x2, `test_repo_hygiene`, `test_auto_mode` and others.
    - The rule tests (`test_policy`, `test_contract`, `test_docs`, `test_testuser_fixes*`, `test_review*`) pass, apart from the pre-existing failures.
    - The 2 `test_sizes` fakes without `.planned` are fixed: `planned_reason` treats a stand-in variant as planning nothing.
  - GPU goldens not run, because no kernel or backend compute code changed.
- `main HEAD` selects all 205 families, i.e. the release suite, which the rules forbid per merge.
