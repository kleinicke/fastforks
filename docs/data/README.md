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
  `box_map` / `mask_map`, one text prompt per COCO category present in the image, threshold 0.5; dataset key
  `coco-val2017-every10-text` (same images, another benchmark).
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

## Devices: `devices.toml`, `speed/<slug>.jsonl`

`devices.toml` lists the measured devices (slug = vendor + model + memory, e.g. `rx-7900-xtx-24gb`; fields described
in the file). Exactly one is the reference (`reference = true`, the RTX 5060 Ti): its rows are `speed.jsonl` above, its
tables `docs/benchmarks.md`, and only its rows are a regression floor (`tests/test_speed_floor.py`,
`tools/speed_check.py`, `tools/merge_check.py`). Every other device is a survey: rows in `speed/<slug>.jsonl`, a
generated page `docs/devices/<slug>.md`, and **no floor**: a slower survey row never fails a check.

A survey row has the keys of `speed.jsonl` plus:

| key | meaning |
|---|---|
| `device` | the slug; must match the file name |
| `task` | the model's task (`stereo`, `depth`, ...) |
| `dtype` | `float32`, `bfloat16` or `float16`: every row names its compute dtype |
| `status` | `ok`, `unsupported` (the card does not offer this mode / dtype on the backend), `failed` (it ran and raised; `error`), `oom` |
| `protocol_id` | the timing protocol class; cross-device tables compare only equal ids. `warm-median`: median of warm, synchronised calls after warm-up |
| `memory_kind` | `torch-peak` (`peak_alloc_mb`, `peak_reserved_mb`; CUDA, ROCm) or `process-footprint` (`peak_footprint_mb`, the process's peak physical footprint; MPS, unified memory). Kinds are never compared |
| `golden` | optional: `pass`, `fail` or `skip`, the golden check (`tests/test_golden.py::test_gpu`) of the same loaded model on this device; `golden_mean` / `golden_p99` (error), `golden_msg` (why skipped), `golden_error`. A `fail` row keeps its time but is never a table's fastest |

An `ok` row has `ms` and its memory fields; any other status has no `ms` and a `note` or `error`. `ref`, `mode`,
`dtype`, `input` are unique per file. `tools/gen_docs.py` checks every row and fails on a bad one.

Adding a device:

1. On the device: `uv run python tools/speed_survey.py detect` prints a `devices.toml` entry for its GPU; paste it
   (`backend` = `cuda`, `rocm` or `mps`; `match` = the GPU name the tools record; edit the slug if you like).
2. Add `docs/devices/<slug>.md` to the Benchmarks section of `mkdocs.yml` (`gen_docs.py` refuses a page missing there).
3. `... speed_survey.py list --device <slug>` shows the jobs (`*` = cross-device subset); `... prepare --device
   <slug>` downloads their weights (and on CUDA the kernel binaries for this GPU) and checks them; `... run --device
   <slug>` measures every ref of the floor at its task's standard input, in each mode and dtype the card validates for
   the backend (the rest become `unsupported` rows), runs the golden check on the loaded model, and merges the rows
   into `speed/<slug>.jsonl`. `--subset` = only one variant per family; `--only REF ...` a part; rows already there
   are skipped unless `--force`; `run` is offline (`FASTFORKS_OFFLINE=1`) unless `--online`.
4. `uv run python tools/gen_docs.py` writes the page (hand-written text outside its markers is kept) and the
   "Other devices" / "Across devices" block of `docs/benchmarks.md`.

On CUDA / ROCm the worker is `tools/merge_check.py`'s, i.e. the floor's own (`tools/speed_check.py`: first call, 3
warm-up, peak reset, median of 3 rounds x 5), so a survey time and the reference time of the same input are measured
the same way. MPS has its own worker (4 warm-up, median of 5; process footprint). Rows record the environment
(`gpu`, `torch`; on NVIDIA `arch`, `cuda`, `driver`, `power_limit`, `clocks`, `throttle_reasons`; on a Mac `gpu_cores`
and `os`, as the chip name does not say how many GPU cores it has).

On a Mac (Apple silicon only; an Intel Mac is refused): the MPS allocator is capped at the recommended working set
(about 75 % of RAM), so a model that needs more is an `oom` row instead of swapping, and before each job the run waits
until 15 GiB (`FASTFORKS_MIN_FREE_GB`) are free on the system disk (macOS swaps there). `match` must equal the chip name exactly ("Apple M1" is
not an M1 Pro). The Metal kernels compile on the device at first use (`torch.mps.compile_shader`): nothing to
download, `prepare` fetches only weights.

### Runbook: a rented NVIDIA GPU

Rented time is paid by the minute, so everything that is not a measurement happens before or after it.

1. Before: on the Mac, nothing to build. Have the goldens ready to copy (`FASTFORKS_GOLDEN_PATH`, about 200 MB) and
   a Hugging Face token if the gated families (DINOv3, SAM 3) are wanted.
2. On the box (any CUDA image with a driver >= 570; a persistent volume for the cache saves the downloads next time):

       git clone <repo> fastforks && cd fastforks && git checkout <branch>
       uv sync                                      # the same lock as the reference
       rsync -a mac:/Volumes/Daten/fastforks-golden/ ~/golden/ && export FASTFORKS_GOLDEN_PATH=~/golden
       uv run python tools/speed_survey.py detect   # paste into docs/data/devices.toml, add the page to mkdocs.yml
       uv run python tools/speed_survey.py prepare --device <slug>   # downloads; exit 1 = a file is missing
       nvidia-smi -q -d CLOCK,POWER | head -40      # what the rows will record; lock clocks only where the host allows

3. Measure, subset first (about 10 min), then the rest (about 1 h for every ref) in the same file:

       uv run python tools/speed_survey.py run --device <slug> --subset
       uv run python tools/speed_survey.py run --device <slug>

   A kill or a lost connection loses at most the job in flight; the same command continues. Use `tmux` or `nohup`.
4. After: copy `docs/data/speed/<slug>.jsonl` (and the `devices.toml` entry) back, then `uv run python
   tools/gen_docs.py` and `mkdocs build --strict` on the Mac. Check the rows: `golden` `fail` cells say a mode is
   wrong on this GPU, and `throttle_reasons` other than none / idle say a time is not the GPU's best.

## `plain.toml`

One plain-language sentence per model family, shown on the model pages.
