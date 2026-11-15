# flowformer-small-chairs: profile at 375x1242 (portable, float32), 2026-11-15

GPU RTX 5060 Ti, `gpulock-low`, torch.profiler over 3 calls after 4 warm-ups. Portable 351 ms / 3165 MiB vs original 303 ms / 2382 MiB (floor rows).
CUDA time is spread over many kernels: TF32 implicit-GEMM convolutions (14.6 + 10.8 + 5.1 + 3.7 %), magma sgemm (14.5 %, 159 calls),
NCHW<->NHWC layout kernels (9 %, 3084 calls), float elementwise ops (6.7 + 4.1 %). No single attention kernel dominates, so a fused
SDPA alone does not close the gap; a fix needs channels_last / fp16 for the CNN and the cost-memory path (accuracy to be re-validated).
Status: bar NOT met in portable (slower, more memory) and kernels = none; open, not fixed in this round.
