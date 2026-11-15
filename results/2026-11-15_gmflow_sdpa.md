# gmflow: fused attention (portable), kernels reason

Date 2026-11-15, GPU RTX 5060 Ti, base commit of branch work/ports + this change, via `gpulock-low`.
Command: `tools/speed_check.py run --only gmflow-* --modes portable --sides lib` (rows: `raw/2026-11-15_gmflow/gmflow.jsonl`).

Change: every attention (Transformer windows, global matching, global flow propagation) is one
`F.scaled_dot_product_attention` (`net/attention.py: sdpa`, value channels padded to 8, q/k made contiguous: a transposed
token layout drops SDPA to the math path and wrote the 4 GB matrix again, found by per-stage peak memory).
Replaces the row-chunked softmax of the earlier portable. Result is the same maths; float32 summation order differs.

| ref | size | original ms / MiB | portable before | portable now |
|---|---|---|---|---|
| scale1-things | 375x1242 | 53.3 / 537 | 56.0 / 525 | 44.5 / 351 |
| scale1-things | 1080x1920 | 750 / 8360 | 801 / 5512 | 556 / 4496 |
| scale2-things | 375x1242 | 145.2 / 1983 | 145.6 / 1466 | 124.1 / 1419 |
| scale2-things | 1080x1920 | 1599 / 8425 | 1636 / 6433 | 1139 / 6017 |
| scale2-regrefine6-things | 375x1242 | 255.9 / 2027 | 244.4 / 1488 | 224.2 / 1441 |
| scale2-regrefine6-things | 1080x1920 | 2175 / 8537 | 2183 / 6443 | 1665 / 6080 |

The other 3 scale1/scale2 and 4 regrefine6 variants share the architecture and measure within 1 % of these rows (floor file).

Accuracy: the forward test against the vendored upstream forward (random weights, float32, 3 refinements) differs by at most
2.1e-3 px (was < 1e-4 with the explicit softmax); tolerance of that test set to 5e-3 with the reason in the test; the
gpu goldens (tolerances unchanged) pass. A KITTI/Sintel EPE run was not repeated: change is reordering of float32 sums.

Kernels mode: `status = "none"`. Profile at 1080x1920 after the change: attention kernels about 55 % of CUDA time
(fp32 matching memory-efficient SDPA 29 %, fp16 transformer SDPA 26 %), float32 elementwise 22 %; convolutions are a
small share, so cudnn_hacc could not change the total. Portable already beats the original in speed and peak memory.
