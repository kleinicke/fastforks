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
