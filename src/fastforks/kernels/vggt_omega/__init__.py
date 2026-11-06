"""VGGT-Omega kernels: the LayerNorm / residual / q-k-norm + RoPE fusions of kernels mode (the Triton kernels).

`vggt_omega_kernels.cu`, one block of 256 threads per row for the LayerNorms (C % 4 == 0, C <= 4096, 16-byte aligned
rows), one warp per (token, q / k, head) row of the packed `[B, N, 3, heads, 64]` fp16 qkv for the q / k pass.
"""

from __future__ import annotations

from pathlib import Path

from fastforks._core.kernels import cubin_library

LN_THREADS = 256
LN_MAX_C = LN_THREADS * 4 * 4  # float4 per thread, 4 per thread at most
QK_ROWS = 8  # warps (rows) per block of omega_qk_norm_rope

# p: pointer, i: int32, l: int64, f: float32 (must match the extern "C" kernels in vggt_omega_kernels.cu)
SIGNATURES = {
    "omega_ln_f2h": "ppppillif",
    "omega_add_ln": "pppppppif",
    "omega_qk_norm_rope": "ppppppp" + "liii" + "ff" + "ii",
}

load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
