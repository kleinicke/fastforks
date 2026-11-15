# Weights repos named after the weights: plan and licence checks (2026-11-15)

User decision 2026-11-15: one Hugging Face repo `fastforks/<weights-name>` per distinct weight file; its identity is
the SHA-256, and every card that pins those bytes points at it. Each repo has a README (source, revision, licence +
evidence, SHA-256 + size, users, how fastforks downloads it) and the upstream LICENSE/NOTICE.

Base: `work/ports` f465d984, branch `attempt/weights-repos`. Inventory: 660 weight entries in all cards, 634 distinct
SHA-256 (script: scratch `inv.py`, read via `fastforks._core.registry`). Repo table: `tools/weights_repos.toml`;
staging: `tools/stage_hf.py weights OUT [--standin]`; card syntax `mirror = "<repo>"` (fallback URL
`https://huggingface.co/fastforks/<repo>/resolve/main/<original file name>`), old `mirror = true` URLs unchanged.

Policy applied: the weights' OWN licence must allow redistribution (code licence alone does not count; weights that
only inherit the code licence by assumption are not mirrored); never gated; no GPL/AGPL/LGPL; no face recognition;
nc only labelled (none here).

## Repos (mirror / upload)

| repo | file | size (bytes) | SHA-256 | cards | licence (verified) | decision |
|---|---|---|---|---|---|---|
| dinov2-vits14-pretrain | dinov2_vits14_pretrain.pth | 88283115 | b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9 | dinov2, dinov2_linear | Apache-2.0: README @7764ea0 "DINOv2 code and model weights are released under the Apache License 2.0"; LICENSE identical to upstream | mirror (unchanged copy) |
| dinov2-vitb14-pretrain | dinov2_vitb14_pretrain.pth | 346378731 | 0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73 | dinov2, dinov2_linear | same | mirror |
| dinov2-vitl14-pretrain | dinov2_vitl14_pretrain.pth | 1217586395 | d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428 | dinov2, dinov2_linear, dedode, romav1 | same | mirror (4 families share it) |
| aliked-n16 | aliked-n16.pth | 2738091 | 5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2 | lightglue, flexglue | BSD-3-Clause: the file is part of the ALIKED repo @683d7c6 (README: "pretrained ALIKED models are provided in models/"), LICENSE identical | mirror |
| seedvr2-3b | seedvr2-3b.safetensors | 6784909160 | 61df4f7979f4a4005029dd93ee186e97397d326b81b585d4fdf445573b85f0a7 | seedvr2 | Apache-2.0: HF card ByteDance-Seed/SeedVR2-3B @37255ff8 `license: apache-2.0`, ungated; code LICENSE @e4de8c2 Apache-2.0 | upload (fastforks conversion; card url already points here, repo did not exist: url broken until upload) |
| seedvr2-7b | seedvr2-7b.safetensors | 16482579008 | bf73826841cfd31895370e92472fae58810ef456310dcc7a167e22dd673eeb7b | seedvr2 | Apache-2.0: SeedVR2-7B @eb0c4281 `license: apache-2.0`, ungated | upload (same) |
| seedvr2-vae | seedvr2-vae.safetensors | 501325110 | bcf7f47d490efcfdce05659510baec78e6bb8301ae6b2dee584a09bdfa54cd4a | seedvr2 (3b, 7b) | Apache-2.0 (both cards) | upload (same) |

Existing `fastforks/marigold-v2-depth` (converted, public, README/LICENSE/NOTICE): already per-weights; untouched.

## Not given a repo

| weights | cards | reason |
|---|---|---|
| 34 `mirror = true` files | aanet, crestereo, great_igev, gwcnet, hitnet, igev_plusplus, igev_stereo, raft_stereo, selective_igev, waft | weights licence only "assumed from code" (no upstream statement on the weights at the pinned commits; hitnet's file is PINTO's model-zoo conversion). Policy says: not mirrored. They stay in the legacy public `fastforks/mirrors` (uploaded 2026-11-01 with the user's OK; add-only rule: not deleted, old URLs keep working). **Conflict for the user to decide.** |
| DINOv3 ViT-B/16, ViT-L/16 (shared) | dinov3, pointdit, insid3 | DINOv3 License, gated on HF: never mirrored |
| CLIP BPE vocab / merges (shared) | owlv2, clipseg, sam3 | hosted on HF at pinned revisions (google/owlv2, CIDAS/clipseg): no availability gain |
| SD VAE / text encoder (shared) | lotus, stablenormal | HF-hosted at a pinned revision: no gain |
| Marigold VAE (shared) | marigold, marigold_normals | HF-hosted at a pinned revision (OpenRAIL++): no gain |
| all other 600+ files | one card each | authors' hosts; no shared SHA, no mirror need |
