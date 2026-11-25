"""CUDA normalization and window accumulation for channels-last FP16 volume inference.

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
SIGNATURES = {
    "in_stats_h": "ppiip",
    "in_final_h": "pppppiiifp",
    "in_apply_h": "ppliifp",
    "vol_accum_h": "pppiiiillli",
    "vol_resize_labels_h": "pppiiiiiiiiiilllfff",
}
_ROWS = 256  # threads per stats block
_VECTOR_BYTES = 16
_V8 = 8
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def accumulate_(out: torch.Tensor, logits: torch.Tensor, gaussian: torch.Tensor) -> None:
    """Add Gaussian-weighted logits in place, rounding the product to FP16 before the addition."""
    if (
        out.ndim != 4  # noqa: PLR2004 - [C, D, H, W]
        or not out.numel()
        or out.stride(0) != 1
        or logits.shape != out.shape
        or not logits.permute(1, 2, 3, 0).is_contiguous()
        or gaussian.shape != out.shape[1:]
        or not gaussian.is_contiguous()
        or any(t.dtype != torch.float16 or t.device != out.device for t in (out, logits, gaussian))
        or not out.is_cuda
    ):
        msg = "Volume accumulation needs CUDA FP16 logits and output in channels-last layout and contiguous weights"
        raise ValueError(msg)
    channels, d, h, w = out.shape
    oz, oy, ox = out.stride()[1:]
    packed = not ((channels | oz | oy | ox) & 1 or (out.data_ptr() | logits.data_ptr()) & 3)
    count = out.numel() // (2 if packed else 1)
    load(out.device).launch(
        "vol_accum_h", *kernels.linear(count), out, logits, gaussian, channels, d, h, w, oz, oy, ox, int(packed)
    )


def resize_labels_(  # noqa: PLR0913 - source slab and destination position in their respective grids
    out: torch.Tensor,
    current: torch.Tensor,
    previous: torch.Tensor | None,
    *,
    first_src_z: int,
    first_out_z: int,
    src_depth: int,
    dst_depth: int,
) -> None:
    """Trilinearly resample FP16 scores in FP32 and write uint8 argmax labels on the destination grid."""
    if (
        current.ndim != 4  # noqa: PLR2004 - [C, D, H, W]
        or current.stride(0) != 1
        or current.dtype != torch.float16
        or current.device != out.device
        or out.ndim != 3  # noqa: PLR2004 - [D, H, W]
        or not out.numel()
        or not current.numel()
        or out.dtype != torch.uint8
        or not out.is_cuda
        or not out.is_contiguous()
        or not 1 <= current.shape[0] <= 256  # noqa: PLR2004 - uint8 output
    ):
        msg = "Volume label resampling needs CUDA FP16 channels-last scores and a contiguous uint8 output"
        raise ValueError(msg)
    channels, available, src_h, src_w = current.shape
    out_depth, out_h, out_w = out.shape
    if not (
        0 <= first_src_z < first_src_z + available <= src_depth
        and 0 <= first_out_z < first_out_z + out_depth <= dst_depth
    ):
        msg = "Source and destination slabs must be within the resampling grids"
        raise ValueError(msg)
    if previous is not None and (
        previous.shape != (src_h, src_w, channels)
        or previous.dtype != current.dtype
        or previous.device != current.device
        or not previous.is_contiguous()
    ):
        msg = "Previous score plane must match the current slab's CUDA device, dtype, height, width and channels"
        raise ValueError(msg)
    load(out.device).launch(
        "vol_resize_labels_h",
        *kernels.linear(out.numel() * 32),
        out,
        current,
        previous,
        channels,
        src_depth,
        src_h,
        src_w,
        first_src_z,
        available,
        out_h,
        out_w,
        out_depth,
        first_out_z,
        *current.stride()[1:],
        src_depth / dst_depth,
        src_h / out_h,
        src_w / out_w,
    )


def supported(x: torch.Tensor) -> bool:
    """Whether `norm_act_` takes `x`: nonempty CUDA fp16 `[N, C, D, H, W]`, channels-last, C % 8 == 0, aligned."""
    return (
        x.is_cuda
        and x.dtype == torch.float16
        and x.ndim == 5  # noqa: PLR2004
        and x.numel() > 0
        and x.shape[1] % _V8 == 0
        and x.is_contiguous(memory_format=torch.channels_last_3d)
        and x.data_ptr() % 16 == 0
    )


def norm_act_(  # noqa: PLR0913 - two distinct bias vectors in the convolution and normalization
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    slope: float,
    *,
    conv_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize and activate in place, optionally adding the convolution bias with fp16 rounding first."""
    if not supported(x):
        msg = "Volume norm needs a nonempty, aligned CUDA fp16 channels-last volume with channels divisible by 8"
        raise ValueError(msg)
    n, c = x.shape[:2]
    if any(
        t.dtype != x.dtype or t.device != x.device or t.shape != (c,) or not t.is_contiguous() for t in (weight, bias)
    ):
        msg = "Volume norm needs contiguous fp16 scale and bias vectors with one element per channel"
        raise ValueError(msg)
    if conv_bias is not None:
        if conv_bias.dtype != x.dtype or conv_bias.device != x.device or conv_bias.shape != (c,):
            msg = "Expected one convolution bias per channel on the activation device and dtype"
            raise ValueError(msg)
        conv_bias = conv_bias.contiguous()
        if conv_bias.data_ptr() % _VECTOR_BYTES:
            conv_bias = conv_bias.clone()
    s = x[0, 0].numel()
    lib = load(x.device)
    stats = torch.empty((n, c, 2), dtype=torch.float32, device=x.device)
    c8 = c // _V8
    rows = max(_ROWS // c8, 1)
    blocks = min(max(s // (rows * 64), 1), 1024)
    partial = torch.empty((n, blocks, c, 2), dtype=torch.float32, device=x.device)
    lib.launch("in_stats_h", (blocks, n, 1), (c8, rows, 1), x, partial, s, c, conv_bias, smem=rows * c8 * 16 * 4)
    lib.launch(
        "in_final_h",
        ((c + 31) // 32, n, 1),
        (32, 8, 1),
        x,
        partial,
        stats,
        weight,
        bias,
        blocks,
        s,
        c,
        float(eps),
        conv_bias,
    )
    total = n * s * c8
    lib.launch("in_apply_h", *kernels.linear(total), x, stats, total, s, c, float(slope), conv_bias)
    return x


class ImageNorm(torch.nn.Module):
    """Evaluate an unaffined image instance norm and optional ReLU on a dense channels-last image."""

    ones: torch.Tensor
    zeros: torch.Tensor

    def __init__(self, norm: torch.nn.InstanceNorm2d, device: torch.device, *, relu: bool) -> None:
        """Create unit scale and zero bias for the upstream normalization."""
        super().__init__()
        self.eps = norm.eps
        self.slope = 0.0 if relu else 1.0
        self.register_buffer("ones", torch.ones(norm.num_features, device=device, dtype=torch.float16))
        self.register_buffer("zeros", torch.zeros_like(self.ones))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """View H x W as a depth-one volume; normalize the convolution output in place."""
        volume = x.contiguous(memory_format=torch.channels_last).unsqueeze(2)
        return norm_act_(volume, self.ones, self.zeros, self.eps, self.slope).squeeze(2)


def prepare_image_norms(net: torch.nn.Module, device: torch.device) -> None:
    """Fuse instance norm and an adjacent ReLU in sequential convolution blocks."""
    load(device)
    net.to(memory_format=torch.channels_last)
    for seq in list(net.modules()):
        if isinstance(seq, torch.nn.Sequential):
            for i, child in enumerate(seq):
                if isinstance(child, torch.nn.InstanceNorm2d):
                    if child.affine or child.track_running_stats or child.num_features % _V8:
                        msg = "ImageNorm requires unaffined instance norm with a multiple of eight channels"
                        raise ValueError(msg)
                    relu = i + 1 < len(seq) and isinstance(seq[i + 1], torch.nn.ReLU)
                    seq[i] = ImageNorm(child, device, relu=relu)
                    if relu:
                        seq[i + 1] = torch.nn.Identity()
