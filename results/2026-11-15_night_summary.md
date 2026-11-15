# Night summary 2026-11-15 (coordinator)

State: origin/work/ports d3762df7. Not "done": the gaps below remain.
Checks (gapmatrix, 573 refs; `results/raw/2026-11-15_gaps/gapmatrix.py`, run with PYTHONPATH=<worktree>/src):
portable ok 570 / OOM 2 / missing 1; kernels ok 477 / missing 95 / OOM 1 (missing = no kernels mode or error rows);
compile ok 544 / error 23 / missing 4 / OOM 2; tensorrt ok 477 / error 88 / missing 5 / OOM 3; upstream ok 559 / missing 12 / error 2.
Tests (test_docs, test_policy, test_speed_floor, test_contract, targeted, capped): 1884 passed, 62 skipped (slow). No full CPU/GPU suite run (release only).

## Closed today
- Portable bar: no portable row slower or bigger than the original beyond noise, except those with a written measured reason (below). Fixes at the root: PixelShuffle strides (restormer), ConvNeXt layer scale, fused SDPA (gmflow), per-level MSDA (grounding-dino), CUDA graph (parseq), LightGlue ALIKED fp32 head, memory temporaries (patchmatchnet, c-foundationstereo-s, locotrack-b, dat), flowformer portable 338 ms / 1410 MiB, matanyone/flexglue/loma portable vs original measured.
- Kernels: WAFT, CodeFormer, LightGlue, LoFTR, DeDoDe, grounding-dino MSDA; shared InstanceNorm fix (ben2 kernels 1942 -> 1430 MB, rife/locotrack-s/hat/dat/drct/swinir smaller). Sweep of 174 kernels jobs: no regressions.
- Docs regenerated (PYTHONPATH fix); numbers audit rerun; ~100 compile/TRT rows plus 7 new CLIP/PDA/pi3/s2m2/bridgedepth compile rows.
- Disk: 86 GB reclaimed by hard-link dedup; user freed more; `/` 31 GB free (HDD also full).

## Remaining, with reason
Portable:
- neuflow-v2 ~14 % slower at 375x1242 (fp32 lookup deliberate; fp16 no faster, channels_last changes flow); allowlisted, reason on card.
- efficientvit_b2, fastvit_t8: launch-bound, allowlisted.
- memfof-sintel 1080x1920 portable and seedvr2-7b OOM on 16 GB; flowformer 1080x1920 original OOM.
- resplat-base-re10k: gated weights. seedvr2-3b weights not in local cache (unmeasured here).
Kernels (not releasing where below the bar; no new kernels per user):
- clip / c-radio-v3-b: duplicate fc2 copy above portable (original still higher). timm families (coatnet, davit, focalnet, hgnetv2, efficientformer(v2), shvit): kernels not implemented.
- flexglue, loma, matanyone: no kernels code. 54 kernels error rows: missing raft_flow/romav2 binaries (local only, HF at release) or nc weights.
TRT/compile:
- cold builds > 5 min: great-igev-middlebury, mocha-v1-sceneflow, tapnext(-pp), foundationstereo-vit-l (unsupported rows). fp16 NaN: waft-stereo-b, mvsanywhere. Myelin: defom-stereo, bridgedepth-sf/-rvc. VDA metric-s, totalsegmentator: no exportable module. nnInteractive compile TypeError; PointDiT TRT.
Weights not cached (disk floor 30 GB): pointdit-l-512 (4.7 GB), pe-core-g14-448, pe-spatial-g14-448 (7 GB each).
Other: Marigold V2 Mac rows must be re-measured on the Mac; 2018 old rows lack commit/command/torch fields (stated limit); GPU goldens exist only locally.
Release tail: HF upload of local-only kernel binaries, full CPU/GPU suites on the final commit, squash to main only on user request.
