# Weights repos upload + anonymous verification (2026-11-15)

Commit: 7f109e4f (branch attempt/weights-repos, staged by `tools/stage_hf.py weights`). GPU: none.
Command: `systemd-run --user -p MemoryMax=8G -p EnvironmentFile=~/.config/fastforks/hf.env bash upload.sh` (stage,
upload with huggingface_hub create_repo(private=False) + upload_file one file at a time, add-only; then an
anonymous `urllib` download of every file, without HF_TOKEN, SHA-256 against the staged file). 22 min wall.
Plan and licence checks: results/2026-11-15_weights_repo_plan.md.

All 7 repos were new (created public). 24 / 24 files match.

| repo | file | bytes | SHA-256 (anonymous download) | match |
|---|---|---|---|---|
| aliked-n16 | LICENSE | 1521 | c48ae933e262f77087a1956ae95f4675bfe20895b52fb328eed5b4e11d477bfb | ok |
| aliked-n16 | README.md | 2065 | ed9ea1dfbb4d9e5d43713b22447ceb5a183cb27dcbf81e6e96cb84068f267531 | ok |
| aliked-n16 | aliked-n16.pth | 2738091 | 5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2 | ok |
| dinov2-vits14-pretrain | LICENSE | 11359 | 600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7 | ok |
| dinov2-vits14-pretrain | README.md | 1959 | 0c23f6a42c8cbe4a7298860c4cdeca29398453574f1ba593b6ed2303dd36ca8a | ok |
| dinov2-vits14-pretrain | dinov2_vits14_pretrain.pth | 88283115 | b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9 | ok |
| dinov2-vitb14-pretrain | LICENSE | 11359 | 600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7 | ok |
| dinov2-vitb14-pretrain | README.md | 1961 | ee0e447d2a9549bcf7ed581520fd459d9f71c3a6559175329d033ad1de2c6fac | ok |
| dinov2-vitb14-pretrain | dinov2_vitb14_pretrain.pth | 346378731 | 0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73 | ok |
| seedvr2-vae | LICENSE | 11334 | 5967a0c153e731109b76c5d83b5e0fae2382b6aaf7b898d08ff03702eea59430 | ok |
| seedvr2-vae | NOTICE | 356 | 2245ab7988bed75ffd1aae7c4db80664bb53659ba1c7cf860354565f25aabf29 | ok |
| seedvr2-vae | README.md | 2260 | 92f03ea8b72b6cd2847f2f1a3d62deda9158448c738d2c3c4b3ad830fef586a9 | ok |
| seedvr2-vae | seedvr2-vae.safetensors | 501325110 | bcf7f47d490efcfdce05659510baec78e6bb8301ae6b2dee584a09bdfa54cd4a | ok |
| dinov2-vitl14-pretrain | LICENSE | 11359 | 600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7 | ok |
| dinov2-vitl14-pretrain | README.md | 2014 | a80a435bdf7b569c079f1ed197bad87849c1848e1110c636e771ab2a18547bdd | ok |
| dinov2-vitl14-pretrain | dinov2_vitl14_pretrain.pth | 1217586395 | d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428 | ok |
| seedvr2-3b | LICENSE | 11334 | 5967a0c153e731109b76c5d83b5e0fae2382b6aaf7b898d08ff03702eea59430 | ok |
| seedvr2-3b | NOTICE | 377 | 1a78b103eb361501883621928ecd5e0d2da157c90bb6f6f83b6ba16c4c505707 | ok |
| seedvr2-3b | README.md | 2462 | 86099b05df2007e02b4e4cec597a38e7f0943b37808bc5d57a2792a0f9afb61d | ok |
| seedvr2-3b | seedvr2-3b.safetensors | 6784909160 | 61df4f7979f4a4005029dd93ee186e97397d326b81b585d4fdf445573b85f0a7 | ok |
| seedvr2-7b | LICENSE | 11334 | 5967a0c153e731109b76c5d83b5e0fae2382b6aaf7b898d08ff03702eea59430 | ok |
| seedvr2-7b | NOTICE | 406 | 56592e038f1cdd8dd0c3b5bf78557fe46ba1cf07fdc595673a352c8c48da4dbf | ok |
| seedvr2-7b | README.md | 2534 | 5859cd7df3a75c7c4fe1fd477af95b59947a9624c765c2fab15aa558f57b7fda | ok |
| seedvr2-7b | seedvr2-7b.safetensors | 16482579008 | bf73826841cfd31895370e92472fae58810ef456310dcc7a167e22dd673eeb7b | ok |
