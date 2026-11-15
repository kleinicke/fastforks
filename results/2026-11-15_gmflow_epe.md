# gmflow flow EPE after the SDPA change (portable vs original fp32)
Date 2026-11-15, GPU RTX 5060 Ti 16 GB, base commit a3c3a18a, `tools/eval_flow.py`, KITTI 2015 train, 200 pairs.
Raw: results/raw/2026-11-15_gmflow_epe.jsonl

| ref | portable EPE | original fp32 EPE (audit 2026-11-12) | portable ms | peak MiB |
|---|---|---|---|---|
| gmflow-scale1-things | 10.3461 | 10.35 | 44.85 | 356 |
| gmflow-scale2-things | 6.7468 | 6.747 | 124.33 | 1423 |

Difference within 0.001 EPE: no accuracy change from the SDPA change.
