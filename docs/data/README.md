# Measured data behind the model pages

`tools/gen_docs.py` renders the "Measured" tables of the model pages, [overview](../overview.md) and
[benchmarks](../benchmarks.md) from the JSON-lines files here. The numbers are copied as measured, never recomputed. The
measuring scripts are not part of this repository; the protocols are described below.

## Accuracy: `<task>_eval.jsonl`

One file per task (`stereo`, `depth`, `classification`, `segmentation`, `detection`, `promptable_segmentation`,
`features`, `multiview`), one JSON object per model, tier and dataset. Measured in the 2026-11-05 campaign
(`tools/acc_collect.py` builds the files from the raw output of the `tools/eval_*.py` scripts; the older 2026-10-27
stereo / depth rows were replaced: they predate this repository's history and the current protocol).

| key | meaning |
|---|---|
| `ref`, `task`, `dataset`, `n` | model id, task, dataset key (below), images / pairs / instances evaluated |
| `tier` | how the model ran: `upstream` (the authors' code, fp32), `upstream16` (the authors' own precision option, else `torch.autocast` float16), `upstreamac` (autocast float16 also where the authors have an option), `upstreamhalf` (`.half()` on the authors' modules and inputs), `portable32` (fastforks portable, float32), `portable`, `kernels` (card default dtype) |
| `column` | the docs column the row fills: `original fp32`, `original fp16`, `portable`, `kernels`; `null` = a tier the docs do not show (`portable32`, `upstreamhalf`, `upstreamac` of earlier runs) |
| `mode` | `upstream` (original fp32), `upstream-fp16` (the chosen fp16 original), `portable-fp32`, `portable`, `kernels`; unchosen fp16 variants keep their tier name. One row per `mode` and `dataset` |
| `dtype`, `how`, `note` | precision, how it was obtained (`fp32`, `authors flag`, `autocast`, `half`), details |
| `selection` | on the `original fp16` rows: why this variant (below) |
| `metrics` | the task's metrics (below); stereo / depth rows also carry them flat (`bad`, `epe`, `rel_absrel`, `rel_d1`) |
| `ms_per_img`, `peak_alloc_mb`, `peak_reserved_mb` | mean time per image (pair) after the first, peak torch memory, of the eval run |
| `error` | the row failed; the only accepted failure is fp32 running out of memory (`OOM, <size>`) |
| `commit`, `gpu`, `torch`, `date`, `command` | fastforks commit (`git log -1 -- src`), where, when, the eval command |

"original fp16" is `upstream16`: the authors' own precision option where the code has one, else `torch.autocast`
float16, shown as measured (also where it loses accuracy against fp32).

Datasets and metrics (all pixels, card default options, uint8 CUDA inputs unless stated):

- Stereo: `bad` (% of pixels with an error above `bad_thr`), `epe`. ETH3D two-view training (27 pairs, bad > 1 px),
  Middlebury 2014 trainingH (15 pairs, bad > 2 px), masks and thresholds of the RAFT-Stereo evaluation code
  (`_BAD_THR` in `tools/gen_docs.py`).
- Depth: `rel_absrel`, `rel_d1` after scale and shift alignment. NYUv2 test (654) and iBims-1 (100) from the MoGe
  evaluation set, left images of ETH3D two-view training (27) and Middlebury 2014 trainingH (15), depth from the
  ground-truth disparity. Metric3D gets the ground-truth intrinsics (they only rescale its depth). Marigold's initial
  noise comes from a seeded CPU generator in fastforks and a CUDA generator upstream (different noise draws).
- Classification: `top1`, `top5` (%), ImageNet-1k val, a seeded random subset of 5000 images (seed 0), each model's
  own preprocessing.
- Segmentation: `miou`, `aacc` (%), ADE20K val, every 4th image (500), single scale at the card's size. Sapiens
  (Goliath body parts, no ADE20K classes): COCO val2017 persons instead (the images of the detection subset with a
  non-crowd person), `person_iou` = IoU of "any body part" against the union of the person masks (crowd regions
  ignored), the primary metric.
- Detection: `box_map` (and `mask_map` for Mask R-CNN), COCO val2017 every 10th image (500), pycocotools.
- Promptable segmentation: SAM / SAM 2 `miou_best` (the candidate with the best predicted IoU of three) and
  `miou_single`, one interior point per non-crowd instance of the same 500 COCO images. SAM 3 (concept prompts):
  `box_map` / `mask_map`, one text prompt per COCO category present in the image, threshold 0.5.
- Features: kNN `top1` (k = 20, cosine, similarity-weighted) of the CLS feature at 224x224, ImageNet val split into two
  seeded disjoint subsets (10000 memory, 5000 test).
- Multiview: `absrel`, `d125` (% within 1.25) on DTU scans 1, 9, 24, 33, 75 (reference view 24 + 7 sources, 640x512),
  median-scaled for uncalibrated models; calibrated MVS also `mae_mm`, `le2mm`, `le4mm`. `n` = scans.

## Speed: `speed.jsonl`

Speed and peak memory per model id, mode and input size. `mode` is `upstream` (the original code, shown as "original"),
`upstream-default` (the original code at its own default precision where that is not the card's), `portable` or `kernels`.

| key | meaning |
|---|---|
| `ref`, `mode`, `input` | model id, mode, `HxW` of the network input |
| `scene` | the image or `synthetic` |
| `ms` | median time |
| `peak_alloc_mb`, `peak_reserved_mb` | peak torch memory, when recorded |
| `load_s`, `first_call_s` | startup times, when recorded |
| `protocol`, `gpu`, `torch`, `date`, `note` | how, where and when it was measured (`torch` = `null`: not recorded) |
| `commit` | fastforks commit of the measurement, empty where it predates this repository's history; `upstream` rows name the card's upstream pin |
| `source` | measurement log the numbers were copied from; empty in this repository |

Sources of one family may differ in commit, date and torch version; each row of a "Speed and peak memory" table states its
own. Stereo `portable` and `kernels` rows are one variant per family at KITTI 2015 375x1242 and Middlebury trainingH
Motorcycle 994x1482. Depth rows use the Middlebury image 1482x994 as a uint8 CUDA tensor, and Depth Anything V2 on the
fused-DPT kernels at network size 518x770. Peak memory of the older stereo `upstream` runs was not recorded.

Rows of 2026-11-05 (`tools/speed_original.py`, HITNet / CREStereo `tools/eval_foreign.py --speed`) time the original in
the precision the accuracy tables use: `upstream-default` = fp32, `upstream` = the chosen "original fp16" (above), with
`tier`, `dtype` and `how`; `commit` = the upstream pin, `ff_commit` = the fastforks commit; peaks over the first and the
timed calls, `peak_first_*` of the first call alone.

## `plain.toml`

One plain-language sentence per model family, shown on the model pages.
