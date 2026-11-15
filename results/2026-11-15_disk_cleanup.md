# Disk cleanup 2026-11-15 (SSD `/`: 16 GB free at start)

Commit base: origin/work/ports (6e48c99e), machine RTX 5060 Ti box, no GPU used. Raw per-entry log with sizes (KB): `results/raw/2026-11-15_weights_fetch/deletions_raw.log`.

## Deleted
| what | size | note |
|---|---|---|
| ~/scratch_tests: 168 of 193 entries with no file modified since 2026-11-13 (tree-*, bench-*, branch-*, matrix, frames_*, pool, port-* of Nov 10-12 ...) | 18.1 GB | not referenced by tests; kept: entries touched Nov 13+ (port-gaps-tmp, port-p26/p27/p38, sfm2, numbers, ...) and the git repo tapirscan-flutter |
| ~/.cache/uv (`uv cache clean`) + ~/.cache/pip | ~7.1 GB | venvs unaffected |
| ~/hfup_stage (HF upload staging, uploads verified) | 1.4 GB | |
| ~/.cache/ff_newuser, ff_launch, ff_mvs, ff_export_j, ff_dfine_dbg, ff_docs_pass (Oct/early Nov test-user caches, TRT engines, onnx) | 2.5 GB | |
| ~/.cache/ff_r5_tmp: issues_1114, tiffbench, torchinductor, pytest dirs, tmp_k3*, acc/tmp | 1.1 GB | |
| /tmp entries older than 12 h (tmpfs, RAM) | ~3 GB | |
| 63 clean worktrees of 017 (branch contained in origin/work/ports, no changes/untracked, HEAD older than 24 h), `git worktree remove --force`, branches kept | see raw log | |
| duplicate weight blobs hardlinked to ~/.cache/fastforks/blobs (ff_r5_tmp/cache 5.1 GB, huggingface/hub 0.4 GB; same sha256, one inode) | 4.8 GB | |

## Kept / listed, not touched
- Dirty or unmerged worktrees: wt-apireview wt-consistency wt-gmfss wt-p27 wt-p38 wt-port-p1 wt-port-p4 wt-ports-kernels7 (dirty); 256 more with commits not in origin/work/ports; 32 clean merged but touched < 24 h (wt-hf*, wt-fasttier, wt-origgaps, wt-ports-sweep2, ...).
- ~/.cache/ff_r5_tmp/acc (11 GB: venv-tf, venv-mge, feat_cache.npy) and ff_r5_tmp/cache (FASTFORKS_CACHE of review/gpu_t.sh and sfm env.sh): referenced by scripts/results.
- ~/.cache/huggingface/hub (93 GB, read in place by the library), ~/.cache/torch (3 GB checkpoints), ~/.nv, ~/.triton, data/, /mnt/hdd content.

## /mnt/hdd/projects/data/stereo/_archives: NOT deleted (verification failed)
Approved by the user "after verifying that each archive's extracted data exists". Real size 49 GB (the 311 G in INDEX.md included the SceneFlow archives, already gone; `sceneflow/` is empty).
- `booster/booster_nogt.zip` (50.0 GB, 58931 files, top dirs balanced/unbalanced, `unbalanced/2_001/...`): extracted `data/stereo/Booster` has only 2148 files / 21 GB in train/ + test/ layout; none of the zip's 58931 paths exist there. Not the same data, so deleting would lose it.
- `middlebury2014/*-imperfect.zip` (23 x ~70 MB): extracted `Middlebury/2014` holds only `*-perfect` dirs; 204 of 207 sampled members unmatched. Not extracted.
- `models.zip` (200 MB): not checked, kept.
HDD now has 299 GB free (after the user's SceneFlow deletion), so no pressure; kept. Ask the user whether to delete anyway.
