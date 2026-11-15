# HF: only changed weights stay public (2026-11-15)

User (2026-11-15): "the rule was wrong. The idea was to mirror the weights only if we want to change the weights ... only keep the changed weights public. The others we will discuss in the future."

Base: origin/work/ports 6a3493a6, branch work/hfonly. GPU: none.
- Cards: removed `mirror = "<repo>"` of dinov2-vits14/vitb14/vitl14-pretrain and aliked-n16 (dinov2, dinov2_linear, dedode, romav1, lightglue, flexglue). Mirror code kept.
- HF (HfApi.update_repo_settings private=True, reversible, local staging dirs kept): fastforks/dinov2-vits14-pretrain, -vitb14-pretrain, -vitl14-pretrain, aliked-n16 are private; removed from the Weights collection (now: seedvr2-3b-bf16, seedvr2-7b-bf16, seedvr2-vae-bf16, marigold-v2-depth-nf4).
- `tools/weights_repos.toml`: live list = converted repos only; the 4 data blocks sit under `[discuss_later.*]` (ignored by stage_hf) to restore later.
- Rule text (toml header, CONTRIBUTING.md, plan note, test_policy): weights repos only for weights we changed; licence checked; name = upstream stem + dtype tag; README states source, precision, conversion script. test_policy now requires `converted`, `precision`, `script` per repo.
- Card READMEs: none mention mirrors; `stage_hf.py cards --check-live` = 0 of 205 differ (before and after), so no card upload needed.
- Anonymous check: the 4 repos give HTTP 401; seedvr2-{3b,7b,vae}-bf16 and marigold-v2-depth-nf4 give 200.
