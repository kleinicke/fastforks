# Measurement gaps solved (2026-11-15)

Worktree `wt-ports-gaps` off `origin/work/ports`; scripts and raw logs in `results/raw/2026-11-15_gaps/solve/`
(`diag1.log` diagnosis, `master.sh` loop = unit `gapfill-solve`, `master.log`, `rows_compile.jsonl`, `finalize.py`).
GPU: RTX 5060 Ti, every step under `gpulock-verylow timeout 175`. Rows carry `source`/`command`/date as the existing ones.

## Root causes (TRT failures, 36 refs)
| Cause | Refs | Fix / status |
|---|---|---|
| Harness: tracker `_net_input` read the queries tensor as the video (`expected 4, got 2`) | 9 trackers | fixed in `tools/speed_check.py`; bootstapir, tapir, tapnext (compile), alltracker* compile measured |
| ONNX: InstanceNorm got fp16 input with fp32 scale (folded weight cast) | tapir (+ trackers using instance norm) | fixed in `_core/export.py` (`_WEIGHT_CAST_OPS` includes InstanceNormalization): tapir TRT 6.47 ms |
| Outer `--timeout` counted the lock wait -> false "timeout after 190/290 s" | dedode-v2-b, stablenormal-turbo, wd-vit-tagger, segformer, seedvr2-7b ... | fixed in the loop (outer timeout 7000 s); real causes then visible (see rows) |
| torch.export data-dependent guards | xfeat, aliked-lightglue, loftr-outdoor/-indoor, geocalib-pinhole, track-on2, alltracker* | recorded `unsupported:` (no general rewrite is small) |
| ONNX op without decomposition (`_is_all_true`) | grounding-dino-tiny | `unsupported:` |
| TensorRT 10 Myelin / ITopK limits | gmflow-scale2-sintel, las2-h, bridgedepth-sf, dedode-v2-b/-g (K > 3840) | `unsupported:` |
| Engine-build CUDA OOM at 16 GB | raft-small, memfof-kitti (likely raft-large*, memflow*) | `unsupported:` (memory) |
| fp16 engine all NaN | dfine-x-coco | `unsupported:` (open: an fp32 island could fix it) |
| Cold build > 175 s lock slice (`worker rc=124`) | tapnext, mvsanywhere, great-igev, waft-stereo-b, igev-stereo-*, mocha-v1, selective-igev-* | open: needs an unlocked capped cold build (the permission check refused my unlocked stage, so not done) |
| Disk full (`Errno 122`) | stablenormal-turbo, defom-stereo-vitl-sceneflow, dinov2-vitg14 TRT | open: re-run when `/` has space |

## Compile
c-foundationstereo-s: Dynamo "Guard failed on the same frame" (`___from_numpy(divider)` dispatch-key mismatch, torch 2.14.1): open, not
fixed (needs the numpy-tensor constant out of the compiled region). romav1 (first compile > 175 s): `--cache-dir/--cold-from` added
to `speed_check.py` (unlocked cold build, locked timing of the warm start); romav1 weights are not cached, so not run.

## Weights
169 of 213 downloadable refs cached (stopped at the 30 GB free floor; pe-spatial-g14-448, pe-core-g14-448 and the rest of the
largest refs remain: `solve/dl_order.txt` minus `dl_done.txt`). 11 refs skipped (non-commercial gate / HDD-cold). The loop measures TRT,
compile (+warm) and kernels for the newly cached refs; rows measured so far are in `docs/data/speed_compile.jsonl`; the loop keeps
running (`solve/master.log`) and `python3 results/raw/2026-11-15_gaps/solve/finalize.py` appends later rows.

## Originals
memfof/tapnet clones are in `experiments/017-fastforks-lib/upstream/`; the last loop stage measures memfof-kitti/-sintel/-spring and
tapnext-pp (rows in `solve/rows_original.jsonl`, not yet merged). VDA metric-s: no adapter (not a few lines).

## Tests
`test_docs`, `test_policy` pass; `test_speed_floor` fails on `film 1080x1920/480x854` and `restormer-sidd 512x512` portable vs
upstream, rows already on `origin/work/ports` and not touched here (this is also why the earlier gapfill-trt publish failed).
