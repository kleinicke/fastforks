# Review of the Hugging Face upload plan (2026-11-15, read-only, no upload)

Basis: origin/work/ports + attempt/weights-repos c94a55a8, live anonymous API (92 model repos incl. kernels, mirrors, 7 weights repos,
dataset reference-outputs), the 123 staged cards (`tools/stage_hf.py cards`, 205 staged). 15+ spot checks against upstream (below).
Verdicts: OK = go; CHANGE = fix before the 123 repos go public.

## 1. Naming
- OK: weights repos named after the file (fastforks/dinov2-vitl14-pretrain), one per SHA-256, cards named after the family. No literal collision exists today
  (set of the 8 weights names vs 205 family names is disjoint).
- CHANGE (cheap, now): add a `tests/test_hf_names.py`: weights names, family names, `kernels`, `mirrors`, `reference-outputs` pairwise disjoint; every weights name
  must not equal a family name or a family name + `-<word>` that could be a variant. Reason: the plan creates `fastforks/seedvr2` (card) next to
  `seedvr2-3b`, `seedvr2-7b`, `seedvr2-vae` (weights) and `marigold-v2` (card) next to `marigold-v2-depth` (weights); `dinov2-linear` is a family card but
  reads like weights. A reader cannot tell card from weights from the name alone. Today only 8 weights repos exist, so a rename costs nothing (HF redirects
  `resolve/` URLs; urllib follows them). Recommended rule: weights name = `<upstream file stem lowercase>`, plus a conversion tag only if bytes changed
  (`-bf16`, `-fp16`, `-safetensors`, `-int8`), never a bare family or family+size name. So `seedvr2-3b` -> `seedvr2-3b-bf16`, `marigold-v2-depth` ->
  `marigold-v2-depth-fp16` (whatever precision it is). Different bytes = different repo; never overwrite a file (a new revision gets `-r2`); identity is the SHA-256.
- CHANGE: tag every repo by kind, so the org page can be filtered: `fastforks-card`, `fastforks-weights`, `fastforks-kernels`; weights repos also `base_model:` (converted
  ones: upstream repo id + `base_model_relation: quantized`/`adapter` only where true) and `library_name: fastforks`. Today weights READMEs only have `license` + `tags`.
- CHANGE: create HF collections (API `create_collection`): `Weights`, `Model cards - <task>` (one per task, ~15), `Kernels and reference outputs`. HF limits a collection to
  ~100 items (verify when creating), so per task, not one "all cards". 205 + ~25 weights repos in one org page is workable (sort by name; search box) only with collections.
  A `datasets/` type for weights is wrong (HF datasets are not loaded by from_pretrained and mix poorly with models); keep models. A `-weights` suffix on every repo is
  not needed if the disjointness test + tags exist; it would only lengthen names.
- Limits: HF throttles repo creation (429s; limits are not documented precisely, per-hour). Create the 123 in batches of ~20 with backoff and a resumable script (skip repos
  that exist); do not run it in one burst. Large files: 16.5 GB seedvr2-7b is fine (limit 50 GB/file, 300 GB recommended per repo; public storage is free but asks for good use).

## 2. Licences
Spot checks (upstream sentence vs our label), all fine unless marked:
- dinov2 (3 sizes) Apache-2.0 incl. weights (README quote) OK; aliked-n16 BSD-3 (file in repo) OK; seedvr2-3b/7b/vae Apache-2.0 (HF API license apache-2.0) OK; converted -> NOTICE present OK.
- oneformer weights MIT (HF API: mit; code Apache via transformers) OK; mask2former weights CC-BY-NC (MODEL_ZOO), code MIT upstream but our card vendors transformers (Apache): OK.
- nnFoundation CNN/ViT CC-BY-SA-4.0 (HF API) OK, header `license: other` + `license_name: cc-by-sa-4.0` is wrong: HF knows `cc-by-sa-4.0`; use it directly. nnInteractive CC-BY-NC-SA 4.0 OK.
- Falconsai nsfw, SmilingWolf wd-tagger Apache-2.0 (HF API) OK; RMBG-2.0 gated CC-BY-NC: card says gated, not mirrored OK; DFN CLIP Apple AMLR (HF license apple-amlr) OK.
- retinaface (facexlib README, MIT) OK, but trained on WIDER FACE (research terms): the training-data note is there; fine per policy.
- lotus, stablenormal, marigold-normals (live): fine-tuned from Stable Diffusion 2 -> OpenRAIL++ use restrictions pass down; header says apache-2.0 / commercial yes. CHANGE: add one
  line "derived from SD2 (CreativeML OpenRAIL++-M, use restrictions apply)" and `license_name` like the marigold card. 
- CHANGE (defect, 50 of 123 cards): `Files: .` / `Files: ,` printed when `licence.files` is empty (nsfw-vit, mask2former, oneformer, clipseg, ...). Print "no licence file vendored" instead.
- CHANGE: `license_name` slugs like `mit-code-weights-unstated` (17+14+6+... cards, ~45) are invented. HF displays `license_name` verbatim and tags it `other`. Use
  `license: other`, `license_name: see-card` (or the SPDX id of the weights) and put "code MIT, weights not stated (assumed)" in the README body only (already there).
- Header rule is implicit (single-variant cards: weights licence; mixed: other). State it in `stage_hf.py` docs and `CONTRIBUTING.md`: header = weights licence, never the code licence;
  `other` + `license_name: mixed-see-variant-table` when variants differ. The cards follow this today, the rule must be written so it stays.
- Card-only repos host no weights, so nc/CC-BY-SA/unlicensed cards are defensible (we only link). nc labels are right on all 11 checked (`**no**` column, header cc-by-nc / noncommercial).
- Hosting (weights repos): the 7 live repos are defensible: weights licence stated by the authors, never gated, README + LICENSE/NOTICE, converted ones say so.
  Future mirrors: CC-BY-SA needs the share-alike note and attribution in README (not planned yet); nc mirror only labelled `license: other` + nc license_name, no `mirror=` shortcut.
- CHANGE (the 34 legacy files in public fastforks/mirrors): README says "We mirror only weights whose licence allows redistribution", but 34 are "assumed from code" (aanet,
  great_igev, gwcnet, hitnet (PINTO conversion), igev*, raft_stereo, selective_igev, crestereo, waft). Policy says unlicensed weights are not mirrored. Several were fine-tuned on KITTI/Middlebury
  (KITTI = CC-BY-NC-SA data). Options: (a) keep, but rewrite the README sentence to the truth ("licence assumed from the code licence, recorded in the card; the authors state no
  weights licence") and mark the files per row "assumed"; (b) delete. I recommend (a) now (the user already approved the add-only rule; deleting breaks `mirror = true` URLs for existing users) and
  ask the user once. Current wording is the actual risk, not the files.
- CHANGE: dataset `reference-outputs` licence table lists coex (GPL-3.0-only, assumed) outputs and NC model outputs (bridgedepth, defom, fast-foundationstereo). They are disparity maps of our
  synthetic pair (little copyright), but policy says skip GPL: drop `coex-sceneflow@1` from the public dataset or add a line stating the outputs carry no code. Also the dataset README promises
  Apache-2.0 for the input pair: fine.

## 3. Robustness
- Order today: local dirs -> cache -> other caches -> `url` -> `mirror` (`files.py::_download_or_mirror`); SHA-256 checked on every path including mirror and caches; a hash mismatch from `url`
  falls through to the mirror (WeightsHashError is caught when a mirror exists), a mirror mismatch is final. Good.
- OK: pinned bytes by SHA-256 make `resolve/main` safe for integrity (a replaced file fails the check, never loads). It is not safe for availability: someone (us) overwriting a file breaks all cards
  until fixed. CHANGE: rule "never overwrite a published weights/kernel file; new bytes = new filename (kernels: build hash is in the name; weights: new repo name)" in CONTRIBUTING.md + `stage_hf.py`
  refusing `--overwrite`. Pinning URLs to a commit is not worth the churn then.
- Single point of failure: (1) converted weights (seedvr2-3b/7b/vae, marigold-v2-depth) have `url` = our repo only, no fallback; a deleted/renamed repo = permanent failure. CHANGE: record a second
  source for converted files (e.g. GitHub release asset of kleinicke/fastforks, <2 GB limit per asset, so 7b needs a split or a note) or document the one-line reproduction `tools/convert_seedvr2.py`
  as the fallback in the error message. (2) kernels: one repo, no mirror; a missing binary is `KernelUnavailableError` by design, fine, but note that the portable backend still works.
  (3) the 3-way Google Drive group: 80 weights URLs are drive.usercontent.google.com (quota errors); only 34 of those have a mirror. Mirror priority should follow host flakiness (Drive, Dropbox, Wasabi,
  Aliyun, Roboflow: ~100 files), not what is easy; but most of them are "assumed from code" -> blocked by policy. State that in the weights plan so the gap is a decision, not an oversight.
- Org/repo deletion risk: add a second HF owner or org admin; HF keeps `redirect` on rename, not on delete.
- HF outage: authors' URL first (works for non-HF hosts), cache second; offline mode documented (`FASTFORKS_OFFLINE`, `FASTFORKS_WEIGHTS_PATH`). OK.
- Rate/auth: anonymous downloads need no token; `HF_TOKEN` only sent to our repos. OK.

## 4. Before making 123 repos public
- CHANGE: fix the `Files: .` defect and the `license_name` slugs before creating (the generator is read once, repos are cheap to edit later but the HF license badge is cached and indexed).
- CHANGE: `deferred (build >5 min)` cells: 42 cells in 24 cards. If the "no unmeasured cells" rule counts them as gaps, measure; else the sentence must say "not measured here: build > 5 min" with the numbers
  of the nearest measured row; the user already accepted them on the live 82 cards, so low risk.
- CHANGE: dedupe with original authors: for families whose authors host the weights on HF (dinov2 facebook/dinov2-*, depth-anything, sam, owlv2, clipseg, siglip2, oneformer...) add `base_model:`
  front matter pointing at the author repo and a "Original weights" link; HF then shows the card in the original model's "derived/related" tab. Not blocking.
- CHANGE: only publish the card of a family when its status is final; the 82 live cards differ from the new generator (zoedepth tier change) and were updated already, so keep `--check-live` in CI.
- OK: no local paths, tokens, TODO in the 123 (checked by the earlier review and by my regex). GitHub links point to kleinicke/fastforks (user-owned). pipeline_tag values (10 distinct) are all valid HF tasks.
- Goldens: 5 pre-existing live files (dfine-n-coco, sam2.1-hiera-{tiny,small,base-plus,large}) have a different hash from golden.toml: any user running the parity test gets a hash error. CHANGE: overwrite
  them (HF history keeps the old version) in the same step. Remove the 4 stale mc-stereo-* goldens (mc_stereo dropped by policy).
- Not local: fastvit_t12/t8 goldens missing live; tests for those families skip or fail on a clean machine; regenerate or note.

## Verdict list
1. Names OK as is; add disjointness test + `fastforks-*` tags + collections (before upload). Rename the 2 near-collisions now (optional, free today).
2. Fix generator: empty `Files:`, invented `license_name`, nnFoundation `cc-by-sa-4.0`, SD-derived note (before upload).
3. Mirrors README wording for the 34 assumed-licence files (before the repo is announced); ask user keep vs delete.
4. Never-overwrite rule + second source for converted weights (before release).
5. Dataset: overwrite 5 stale goldens, drop mc-stereo and coex outputs.
6. Create repos in resumable batches of ~20 with backoff; then collections.
