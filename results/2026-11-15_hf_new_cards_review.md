# Review of the 123 staged cards for repos that do not exist yet (2026-11-15, commit 24802779)

Nothing created or uploaded. Staged with `tools/stage_hf.py cards OUT`; checked by script for all 123: only `README.md` in each
staged dir (no weight files); `license` front matter matches `licences` / `card.toml` weights-licence strings (SPDX id when
HF knows it, else `other` + name + link; link starts with http); the `commercial use` column of every variant row equals
`card.status(v).commercial` (`**no**` for non-commercial, `unclear (treat as no)`); all variants have a row; no local paths
(`/home`, `/mnt`), tokens or `TODO`. 11 cards carry non-commercial weights, all show `**no**` in the table.

Problems found: none that needs a generator change (no generator edit was made).
Notes for the user:
- 24 cards have cells `deferred (build >5 min)` (compile / TensorRT build over 5 min, e.g. any2full, dat, hat, gimm-vfi, waft,
  seedvr2, wd-tagger); 42 cells in total across the cards, the 82 live updated cards already show them too. Factual, but
  if the 'no unmeasured cells' rule counts them as gaps, measure them or the cells stay.
- Wording `Original = the original code named under Details` replaces the old 'the authors' code' in all cards (intended).
