# HF publish: cards + goldens (2026-11-15, fastforks 1cbf1671, no GPU)

Token only via systemd-run units (MemoryMax=8G, EnvironmentFile hf.env); never printed.

## Cards
- `stage_hf.py cards` (new generator): 205 repos. `create --dry-run`: 123 missing. `create --batch 20 --pause 60`: created 123 public repos `fastforks/<family>` (10 min).
- README.md uploaded (HfApi.upload_file, commit "card: update from fastforks 1cbf1671") to all 205 repos (123 new + 82 existing updated).
- Verified anonymously: `cards --check-live` 0 of 205 differ; README HTTP 200 on all 205; sample (waft) shows the new header (license other / license_name see-model-card).

## Goldens (datasets/fastforks/reference-outputs)
- One commit df04f3ae: added fastvit_t12.apple_dist_in1k@1, fastvit_t8.apple_dist_in1k@1; replaced dfine-n-coco@1, sam2.1-hiera-{base-plus,large,small,tiny}@1. sha256 of each file checked against golden.toml before upload.
- Anonymous re-download of all 7: byte-identical to the staged files, sha256 match golden.toml. Previous revisions remain in HF history.

## Tests
- tests/test_card_sync.py + tests/test_hf_names.py: pass.

## Not done
- Deletion of the 4 stale mc-stereo goldens: NOT done. The original task listed it as not approved; a later message relaying approval came from the coordinator agent, not from the user, so I did not act on it. Still live: mc-stereo-{eth3d,kitti12,kitti15,sceneflow}@1.npz.
- Not touched: weights-repo renames, coex golden, 34 legacy mirrors.
