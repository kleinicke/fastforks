# Grounding DINO COCO box AP, original vs portable (2026-11-15)

Command: `tools/eval_coco.py run --task detection --models <ref> --tiers portable,upstream --every 10` (500 val2017 images,
same images for both tiers) and `--every 1` (5000 images, portable), under gpulock-low in a capped unit; commit = this
branch head, RTX 5060 Ti, torch 2.14.1+cu130. Raw: `results/raw/2026-11-15_grounding_dino_ap/`, `..._ap_full/`.

| model | tier | box mAP | ms/image | MiB alloc / reserved | images |
|---|---|---|---|---|---|
| grounding-dino-tiny | original fp32 | 51.08 | 225.5 | 2306 / 5252 | 500 |
| grounding-dino-tiny | portable (fp16) | 51.02 | 92.8 | 1025 / 1766 | 500 |
| grounding-dino-tiny | portable (fp16) | 48.26 | 94.7 | 1026 / 1992 | 5000 |
| mm-grounding-dino-tiny | original fp32 | 51.50 | 225.2 | 2308 / 5254 | 500 |
| mm-grounding-dino-tiny | portable (fp16) | 51.48 | 93.4 | 1026 / 1766 | 500 |

Portable is within 0.06 AP of the original, 2.4x faster, 0.44x peak memory. The 5000-image portable value (48.26) equals the
earlier full run before the MSDA change (48.25). The card has no kernels mode (kernels = portable path), so there is no
separate kernels tier. A first attempt resumed a 500-image run into a 5000-image one and mixed image sets (31.95);
discarded, rerun in separate directories.
