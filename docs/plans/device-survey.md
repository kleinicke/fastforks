# Brief: speed survey on other devices and per-device benchmark pages

Status (2026-11-06): implemented on branch `docs/device-benchmarks`; no survey run yet (the M1 Pro rows are the
earlier `fastforks bench --device mps` table, converted, at 480x640 and without golden checks). This folds the first
plan (a `survey` command inside `tools/speed_check.py`) into what was built; the parts that changed say why.

## Goal

The RTX 5060 Ti stays the main benchmark (`docs/benchmarks.md`, `docs/data/speed.jsonl`, the regression floor).
Other devices are surveys: the M1 Pro, NVIDIA GPUs rented by the minute (server GPUs first), later AMD. One command
works on every device and the pages are generated from the rows.

## Done

- **Registry and data** (`docs/data/devices.toml`, `tools/_devices.py`): one reference device, survey rows in
  `docs/data/speed/<slug>.jsonl` with `device`, `dtype`, `status` (ok / unsupported / failed / oom), `protocol_id`,
  `memory_kind` (torch-peak / process-footprint, never compared) and the optional `golden` (pass / fail / skip),
  checked by `gen_docs.py`. `speed.jsonl` and everything reading it are unchanged.
- **Survey tool** `tools/speed_survey.py` (`detect`, `list`, `prepare`, `run`). Not inside `speed_check.py`: that
  file is the floor's checker, and a device shim through it would touch the floor's protocol. Instead:
  - cuda / rocm: each job runs `tools/merge_check.py worker`, i.e. speed_check's own worker plus the golden step,
    with one input. Same protocol as the floor (first call, 3 warm-up, 3 x 5), so a rented GPU's time is directly
    comparable with the reference row of the same input. `merge_check.golden_step` now honours the spec's device
    and dtype.
  - mps: its own worker (4 warm-up, median of 5, process footprint via `proc_pid_rusage`, as `torch.mps` has no
    peak API), then the same `golden_step` on the loaded model. `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0` (the
    allocator's cap = the recommended working set, 11.8 GB on a 16 GB M1 Pro; the default 1.7 let FoundationStereo
    allocate into swap) and a 15 GB free-disk guard that waits for swap to shrink keep the Mac out of swap.
  - one input per job (the task's `STANDARD`, the floor's scene), modes portable and kernels; per card the dtypes
    it validates on the backend; everything else an `unsupported` row.
  - job order: cross-device subset first (one variant per family), then the rest; `--subset` stops after it.
  - `prepare` = `fastforks download <refs> [--kernels]` then `--check`; `run` is offline by default.
  - rows record gpu, torch, CUDA / HIP, driver, power limit, clocks and throttle reasons (nvidia-smi).
  - resume: rows already in the file are skipped; `--force` re-measures.
- **Pages** (`tools/gen_docs.py`): `docs/devices/<slug>.md` per survey device, the "Other devices" / "Across
  devices" block in `docs/benchmarks.md` (cell = fastest ok row not failing its golden check, same input and
  protocol id only), `(golden fail)` marks in device tables, nav check in `mkdocs.yml`.
- **Docs**: `docs/data/README.md` ("Devices", with the runbook for a rented NVIDIA GPU).

## Open

1. First run on the M1 Pro (`speed_survey.py run --device apple-m1-pro-16gb --subset`): tests the MPS path for free
   and adds rows at the standard inputs, which the across-devices table needs.
2. First rented NVIDIA run (runbook in `docs/data/README.md`); confirm the subset takes ~10 min and the whole
   floor ~1 h, and that `throttle_reasons` reads idle / none on the host.
3. Kernels on other CUDA archs: `prepare` fetches the binaries for the GPU it runs on; a family without a binary
   for that arch should become a failed row with the loader's message, check that on the first non-sm_120 GPU.
4. AMD: `torch-peak` through `torch.cuda` on ROCm is assumed; confirm on the first run.
5. Originals are not part of the survey (they need the authors' environments). A static audit of which originals
   would run on MPS unchanged is in `docs/plans/originals-on-mps.md`.
