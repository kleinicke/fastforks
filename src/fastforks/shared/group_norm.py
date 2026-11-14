"""GroupNorm on large channels-last CUDA maps from per-channel statistics (fastforks).

PyTorch's GroupNorm on a channels-last tensor is slow on big maps (RealPLKSR at 512 x 512: 4.2 ms per block, 40 % of
the network). `group_norm` computes the same result from per-channel `var_mean` (float32 group statistics by the law
of total variance) and one multiply-add. Measured on an RTX 5060 Ti (2026-11-14, 86 shapes, fp16 and fp32): 1.1-10x
faster from 8 M elements on and from 4 M with at least 512 K elements per (sample, group); below, its ~0.2 ms floor
loses in isolation, so small maps keep `F.group_norm`.
`group_norm_stats` skips the size rule for nets where the whole network says otherwise (RealPLKSR: 1.1-1.3x faster
at 128-192 px with it). The output dtype follows GroupNorm's (float32 under autocast), the values agree within fp16
rounding.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.patch import override

MIN_NUMEL = 1 << 23  # from here on the statistics path
MIN_NUMEL_LARGE_GROUPS = 1 << 22  # ... and from here on when a (sample, group) holds at least MIN_PER_GROUP elements
MIN_PER_GROUP = 1 << 19
_NCHW = 4


def group_norm(norm: nn.GroupNorm, x: torch.Tensor) -> torch.Tensor:
    """`norm(x)`; large 4-D channels-last inputs from per-channel statistics."""
    n = x.numel()
    if n >= MIN_NUMEL or (n >= MIN_NUMEL_LARGE_GROUPS and n // (x.shape[0] * norm.num_groups) >= MIN_PER_GROUP):
        return group_norm_stats(norm, x)
    return F.group_norm(x, norm.num_groups, norm.weight, norm.bias, norm.eps)


def group_norm_stats(norm: nn.GroupNorm, x: torch.Tensor) -> torch.Tensor:
    """`norm(x)`; every 4-D channels-last input from per-channel statistics (RealPLKSR: wins below the thresholds)."""
    b, c = x.shape[:2]
    g = norm.num_groups
    if x.dim() != _NCHW or x.is_contiguous() or not x.is_contiguous(memory_format=torch.channels_last):
        return F.group_norm(x, g, norm.weight, norm.bias, norm.eps)
    out = torch.float32 if torch.is_autocast_enabled(x.device.type) else x.dtype  # autocast runs GroupNorm in fp32
    var, mean = torch.var_mean(x, dim=(2, 3), correction=0)
    var, mean = var.float().view(b, g, -1), mean.float().view(b, g, -1)
    mg = mean.mean(2, keepdim=True)
    a = torch.rsqrt((var + (mean - mg) ** 2).mean(2, keepdim=True) + norm.eps).expand(b, g, c // g)
    if norm.weight is not None:
        a = a * norm.weight.float().view(1, g, -1)
    shift = -mg * a
    if norm.bias is not None:
        shift = shift + norm.bias.float().view(1, g, -1)
    return torch.addcmul(shift.reshape(b, c, 1, 1).to(out), x.to(out), a.reshape(b, c, 1, 1).to(out))


def fast_group_norms(net: nn.Module, *, always: bool = False) -> int:
    """Route every `nn.GroupNorm` of `net` through `group_norm` (`group_norm_stats` with `always`); the count."""
    n = 0
    for m in net.modules():
        if isinstance(m, nn.GroupNorm):
            override(m, group_norm_stats if always else group_norm)
            n += 1
    return n
