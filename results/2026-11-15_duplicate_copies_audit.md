# Duplicate weight copies audit (2026-11-15)

Commit: see `git log` (fix "Kernels ViT/PE blocks: folded layer-scale/QuickGELU weights replace the parameters"), GPU RTX 5060 Ti,
torch 2.14.1+cu130. Tools: `tests/test_no_duplicate_weights.py` (general), per-family `tools/speed_check.py run --modes kernels --only REF`
(protocol of speed.jsonl), unique-storage audit of synthetic timm models (original fp16 = 0.50 of the fp32 bytes).

## Static audit (builds a transformed copy; is the original alive afterwards?)

| place | copy | original alive? | verdict |
|---|---|---|---|
| `transformer._vit_prep` (kernels): gamma into proj / fc2, QuickGELU into fc1 / fc2 / norm2 | `_w_proj`, `_w_fc1`, `_w_fc2`, ln2 | yes (parameters kept) | DUPLICATE, fixed (`_adopt`) |
| `transformer._pe_prep` (kernels, Perception Encoder): gamma into proj / fc2 | `_w_proj`, `_w_fc2` | yes | DUPLICATE, fixed |
| `transformer._eva_prep` (portable): gamma folds, fused qkv / padded fc1 | `_wp`, `_w2` | qkv / fc1 aliased via `_alias`; gamma copies kept on purpose (portable keeps autograd through the parameters, B17-5) | not above original in the test (eva02 portable 0.50) |
| `transformer._cnx_kprep` (ConvNeXt) | `_w2_4` | folded in place (956ea9ea) | ok |
| `_swin_prep`, `_siglip_prep`, `_w4` without fold | view of the parameter (channels_last of `[O,I,1,1]` is contiguous) | aliased | ok |
| `_pe_prep` / `_vit_prep` portable | none | - | ok |
| `cnn.py` / `fold.py` BN / scale folds | `.data =` replaces the weight | original freed | ok |
| dinov3 ViT `pack()` | merged qkv / gate_up | q/k/v deleted | ok |
| mapanything, vggt_omega | own block classes | originals replaced by empty tensors | ok |
| vggt, pi3, lingbot_map, anysplat, dinov3_convnext, tipsv2, dedode, loma, romav1/2 | no folded copies | - | ok (measured below) |

## Measured: weight memory after load (`memory_allocated`, kernels = portable unless noted)

| family | alloc MB portable | kernels | params x 2 B (fp16 floor) | verdict |
|---|---|---|---|---|
| vggt-1b | 2825 | 2831 | 2381 (+216 M fp32 params, same in both) | no copy |
| pi3 | 1924 | 1929 | 1917 | ok |
| anysplat / lingbot-map | 2824 / 2754 | 2832 / - | as vggt | ok |
| mapanything / vggt-omega | 2370 / 2705 | same | own block classes, originals dropped | ok |
| dinov3-vitl16 | 606 | 607 | 455 + merged qkv 151 (q/k/v deleted) | ok |
| dinov3-convnext-large | 392 | 392 | 392 | ok |
| tipsv2-l14 | 611 | 613 | 608 | ok |
| romav1 / romav2 | 841 / 724 | same | 831 / 700 | ok |
| dedode-v2-g | 743 | 679 | 675 | ok |
| loma-b (portable only) | 761 | - | 683 | ok |

## Fixed families: peak allocated MB at the tested size, floor row -> after (kernels, speed unchanged within noise)

| family (ref) | before | after | note |
|---|---|---|---|
| clip-l14-openai | 1054 | 622 | -41 % |
| clip-h14-dfn5b | 2150 | 1250 | -42 % |
| clip-b16-openai | 324 | 203 | |
| pe-core-l14-336 | 937 | 649 | |
| pe-spatial-b16-512 | 331 | 215 | |
| c-radio-v3-l / -g | 930 / 3211 | 631 / 2283 | |
| siglip2-l16-256 / so400m14-224 / g16-256 | 919 / 1182 / 3200 | 641 / 858 / 2259 | |
| dinov2_vitl14_lc / vit_base augreg2 | 890 / 300 | 623 / 200 | |

73 rows dropped by > 5 % (list: appended rows with `source` = this file in `docs/data/speed.jsonl`; a few, e.g. dinov3-convnext, fast-foundationstereo,
vitpose, also gained from earlier commits). One row measured +3 % (dinov2-s-linear-ade20k 512x512, 3.07 -> 3.16-3.20 ms, noise < 1.3 %, inside the 5 % floor, forward unchanged by this fix).
