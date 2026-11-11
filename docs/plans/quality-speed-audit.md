# Quality and speed audit (planned, start when the port queue is short)

Requested by the user on 2026-11-11. Runs over every model: published (v0.1 on `main`) and unpublished (`develop`, `work/ports`).
Do not start it while more than two other GPU workstreams are active.

## 1. Quality versus the original

- The reference is the original's suggested precision.
  - If the authors ship or recommend fp16 or bf16, compare against that.
  - Otherwise compare against original fp32.
  - Always also record original fp16 when it runs.
- Our fp16 (portable and kernels) must not be much worse than that reference.
  - Proposed threshold: no worse than the larger of the original's own fp32↔fp16 gap and 0.5 % relative (0.02 dB PSNR, 0.1 AP points).
  - Any larger gap is a bug to explain or fix, like Marigold ETH3D (0.1075 → 0.1141, being debugged).
- Our fp32 must match original fp32 within noise.
- Known gaps to close first:
  - Models with no original row: raft (4), sea_raft (7), vitpose (6), hrnet (4), parts of rtmpose, moge-3, 4 depth and 2 stereo models on `main`.
  - Models with no eval row at all: dfine, clip, siglip2, perception_encoder, c_radio, rtmpose hand/face/ap10k.
- Output: one table per task in `results/raw/<date>_audit/quality.md` with columns model, reference tier, reference value, ours fp16, ours fp32, delta, verdict.

## 2. Portable speed and memory versus the original

- Same input size and batch size, median of N runs after warm-up, peak allocated and reserved memory.
- Portable must be at least as fast as the original and use no more peak memory.
- Exception: the original uses Triton or another custom kernel that portable may not use. Label the row and compare kernels instead.
- Known cases:
  - NeuFlow at 1080p: portable 103 ms / 8.4 GB vs original fp16 75.6 ms / 4.4 GB. Being tuned. Kernels: 59 ms / 0.8 GB.
  - FastViT T8/T12 vs the reparameterized original. Being tuned.
  - SegNeXt. Being tuned.
- Source data: `docs/data/speed.jsonl` (tier `upstream` and `upstream16` against `portable`). A script lists every model where portable is slower or uses more memory. Only those models get re-measured.

## 3. Kernels versus TensorRT

- Target: kernels within 20 % of TensorRT.
- Status 2026-11-11: 137 of 231 pairs pass, median ratio 1.35.
- Worst pairs: rtmo-t 7.1×, sea-raft spring 5.2×, faster-rcnn 3.6×.
- Work the worst ratios first, one family at a time.

## Order

1. Quality table from existing rows (no GPU).
2. Fill missing original rows.
3. Portable speed check from existing rows.
4. Re-measure only the flagged models.
5. TensorRT gaps by family.

## Status 2026-11-12: steps 1 and 3 from existing rows

Script and tables: `results/raw/2026-11-12_audit/` in the experiment folder (`audit.py`, `quality.md`, `speed.md`; no GPU).

- Quality: 512 model × dataset rows, 14 flagged (11 models, all stereo except `depth-pro` ETH3D and `romav2`); worst `defom-stereo-vitl-sceneflow`, `igev-plusplus-middlebury`.
- No original row: `c-foundationstereo-s`, `great-igev-middlebury` (original OOM), the 6 ViTPose variants. No eval row: 39 ids (CLIP, SigLIP 2, PE, C-RADIO, RTMPose ap10k / face / hand).
- Data fix: `moge-2-vitb-normal` upstream16 rows carry `mode: upstream` (must be `upstream-fp16`), `c-foundationstereo-s` ETH3D has `mode: upstream16`.
- Speed: 44 model × input pairs slower than the best original (21 by > 5 %: NeuFlow v2, LaS2-S/M, FlowFormer-small, EfficientViT-B2, MoGe-3, GMFlow scale1); 67 pairs with more peak memory (30 of them < 64 MiB). No flagged original uses custom CUDA / Triton code.
- Kernels / TensorRT > 1.2: 150 of 283 pairs; 67 of them compare a network-only engine with a whole call (pose / flow with `net_input`), so only 83 are like-for-like.

## Docs proposal: paper, original default, portable, kernels per model (public website)

Not implemented. One compact table per family on the model page, the other precisions folded away.

- Data:
  - Paper values: a new `docs/data/paper.jsonl` (ref, dataset, metric, value, source URL with page / table, protocol note).
  - The protocol note says how the paper's protocol differs from ours, e.g. flip test, detector boxes, full val set.
  - Not in the cards: they stay code facts, and paper numbers change with errata.
  - Original default precision: a per-family key in the card's `[upstream]` table, e.g. `default_dtype = "float16"` and `default_how = "mixed_precision flag in the README demo"`.
  - gen_docs picks the matching eval / speed row (`upstream-fp16` or `upstream`) by that key instead of the current `how` heuristic.
  - Missing key = fp32.
- Main table, one row per variant and the task's main dataset:
  - `variant | paper | original (default dtype) | portable | kernels`.
  - Each cell holds: accuracy, then ms, then peak MiB (reserved) on separate lines (`<br>`, as in today's speed cells).
  - The paper cell holds accuracy only, linked to the source, with † when the protocol differs (footnote from `protocol note`).
  - The original cell names its dtype in the header (e.g. "original (fp16, authors' flag)").
  - The portable / kernels cells mark deltas beyond the audit tolerance (`audit.py` rule) with a warning sign and a footnote.
  - Accuracy and speed rows come from different runs (eval vs `speed.jsonl`): the cell takes speed at the family's standard input, and the header says so.
- Collapsed detail (`<details><summary>All precisions and inputs</summary>`, already used on the detection page):
  - Today's per-dataset table: original fp32, original fp16, portable, kernels, plus `portable32` where measured.
  - The full speed table: every input, original fp32, original default, compile, TensorRT.
  - The parity line.
- gen_docs changes:
  - A `_paper_rows()` loader with the same row check as the other files (fail on a missing source).
  - `_main_table(card)` replaces the top of `_accuracy` / `_speed` on the model page; the existing table functions render the detail block.
  - `tests/test_docs.py`: every v0.1 model has either a paper row or an explicit `paper: none` entry, so gaps stay visible.
- Overview page unchanged, except one optional column, "vs paper", for the main metric.
