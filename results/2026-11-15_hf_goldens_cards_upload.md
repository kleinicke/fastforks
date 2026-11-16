# HF upload: goldens + cards (2026-11-15, fastforks 24802779, no GPU)

Commands: `tools/stage_hf.py goldens OUT --from golden --from golden_ports --skip-missing` (hash vs golden.toml, first match),
`tools/stage_hf.py cards OUT`; upload with huggingface_hub (create_commit, 10 batches of <=40, add-only; HfApi.upload_file for
READMEs), token from `~/.config/fastforks/hf.env` (not printed/stored).

## Goldens (datasets/fastforks/reference-outputs)
- Uploaded 397 missing .npz + updated README (lists the 571 refs with a local file). No existing file overwritten or deleted.
- Verified: listing = 573 refs - 2 non-local = 571 live; anonymous download + sha256 of a random 30: 29 OK, 1 stale (below);
  HF LFS hashes of all 571 vs golden.toml: 566 OK, 5 stale.
- Not local (not live): fastvit_t12.apple_dist_in1k@1, fastvit_t8.apple_dist_in1k@1.
- Live files without golden.toml ref: mc-stereo-eth3d@1, mc-stereo-kitti12@1, mc-stereo-kitti15@1, mc-stereo-sceneflow@1.
- Pre-existing live files whose hash differs from golden.toml (not touched, need a decision to overwrite; the right files are
  staged in ~/hfup_stage/goldens): dfine-n-coco@1, sam2.1-hiera-base-plus@1, sam2.1-hiera-large@1, sam2.1-hiera-small@1,
  sam2.1-hiera-tiny@1.
- Tool: `--from` is repeatable, `--skip-missing` leaves out refs without a hash-matching local file (test added).

## Cards
- 82 existing repos: README.md updated (commit 'card: update from fastforks 24802779'); 3 diffs read first (zoedepth, alexnet,
  mocha-stereo): tier stable, new benchmark/accuracy tables, no local paths/secrets. `--check-live` afterwards: 0 differ.
- 123 missing repos: not created; review in results/2026-11-15_hf_new_cards_review.md.
