"""Instance norm (affine) + LeakyReLU of channels-last 3D fp16 activations, in place (`vol_norm.cu`; fp32 math).

`norm_act_`: nnU-Net's conv block epilogue (`InstanceNorm3d` + `LeakyReLU`) in two passes over the activation
(per-channel shifted sums, then scale / shift / activation), instead of PyTorch's strided reductions and broadcast
elementwise kernels on `channels_last_3d`. Used by `fastforks.volume.models.totalsegmentator` (kernels mode).
"""

from __future__ import annotations

from pathlib import Path

import torch

from fastforks._core import kernels
from fastforks._core.kernels import cubin_library

# p: pointer, i: int32, l: int64, f: float32 (must match the extern "C" kernels in vol_norm.cu)
SIGNATURES = {"in_stats_h": "ppii", "in_final_h": "ppppiiif", "in_apply_h": "ppliif"}
_ROWS = 256  # threads per stats block
_V8 = 8
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def supported(x: torch.Tensor) -> bool:
    """Whether `norm_act_` takes `x`: CUDA fp16 `[N, C, D, H, W]`, dense channels-last, C % 8 == 0, aligned."""
    return (
        x.is_cuda
        and x.dtype == torch.float16
        and x.ndim == 5  # noqa: PLR2004
        and x.shape[1] % _V8 == 0
        and x.is_contiguous(memory_format=torch.channels_last_3d)
        and x.data_ptr() % 16 == 0
    )


def norm_act_(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float, slope: float) -> torch.Tensor:
    """`leaky_relu(instance_norm(x, weight, bias, eps), slope)` in place; `x` as `supported`, fp16 weight / bias."""
    n, c = x.shape[:2]
    s = x[0, 0].numel()
    lib = load(x.device)
    stats = torch.zeros((n, c, 2), dtype=torch.float32, device=x.device)
    c8 = c // _V8
    rows = max(_ROWS // c8, 1)
    blocks = min(max(s // (rows * 64), 1), 1024)
    lib.launch("in_stats_h", (blocks, n, 1), (c8, rows, 1), x, stats, s, c, smem=rows * c8 * 16 * 4)
    lib.launch("in_final_h", *kernels.linear(n * c), x, stats, weight, bias, n, s, c, float(eps))
    total = n * s * c8
    lib.launch("in_apply_h", *kernels.linear(total), x, stats, total, s, c, float(slope))
    return x
