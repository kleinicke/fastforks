# HF follow-ups (2026-11-15, work/ports d74ee788 + this commit, no GPU; user decisions relayed by the coordinator)

Token only via systemd-run units (MemoryMax=8G, EnvironmentFile hf.env), never printed. Ops script: scratchpad `ops.py` (steps mc, rename, readmes, private, cards, collections).

1. Deleted `mc-stereo-{eth3d,kitti12,kitti15,sceneflow}@1.npz` from dataset fastforks/reference-outputs. Anonymous GET: 404.
2. Renamed with `move_repo` (old URLs redirect, anonymous file GET 200 on old and new name): seedvr2-3b -> seedvr2-3b-bf16, seedvr2-7b -> seedvr2-7b-bf16,
   seedvr2-vae -> seedvr2-vae-bf16, marigold-v2-depth -> marigold-v2-depth-nf4 (live file is NF4; the repo is NOT in released 0.1.1 a0659158, so renaming is allowed).
   File names inside the repos unchanged. card.toml urls, tools/weights_repos.toml, docs/plans/marigold-v2-port.md updated.
3. Weights READMEs: source, precision, conversion script (new `precision` / `script` keys in weights_repos.toml, rendered by stage_hf.py; the script is copied into the repo:
   convert_seedvr2.py in the three seedvr2 repos, convert_marigold_v2.py in marigold-v2-depth-nf4, whose README also got a Precision / Conversion script paragraph and the fastforks-weights tag).
4. `mirror = true` removed from all 34 cards entries (incl. the 3 dead WAFT links); `mirror = "<repo>"` (per-weights repos) and the mirror code stay. fastforks/mirrors is private
   (anonymous API/file: 401). Not deleted. Notes in CONTRIBUTING.md, files.py, stage_hf.py.
5. Cards re-uploaded: fastforks/seedvr2, fastforks/marigold-v2 (only these differed; `cards --check-live` after).
6. Collections (namespace fastforks, 22): Weights (8), Kernels and test data (kernels + reference-outputs dataset), Model cards: <task> x20 (largest classification 36, all <= 100).
7. Tests (light): test_card_sync test_hf_names test_policy test_docs test_hf_cards test_files test_offline test_restoration_seedvr2: 1954 passed, 66 skipped.
