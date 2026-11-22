"""Portable-mode memory fixes (same math).

TAO's disparity transformer runs in chunks of its (B*H*W) disparity sequences; the cost-volume hourglass frees its
upsampled volume right after merging it and applies LeakyReLU in place.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.half import pure_half
from fastforks._core.patch import is_base, override
from fastforks._core.tracing import EXPORTING_FOR_MLX

from .net.fast_foundation_stereo.fast_foundation_stereo import _FFSPostForwardHelper
from .net.foundation_stereo.convolution_helper import Conv
from .net.foundation_stereo.submodule import (
    CostVolumeDisparityAttention,
    FeatureAtt,
    FlashMultiheadAttention,
    HourGlass,
)

# sequences per chunk: memory bound (Q, K, V, out of one chunk are ~200 MB at 375x1242); flash SDPA takes < 65536; the
# result does not depend on it
CHUNK = 1 << 13
LARGE_VOLUME_BYTES = 256 * 1024**2
PATCH_SCALE = 4


def _chunked(
    self: CostVolumeDisparityAttention, cv: torch.Tensor, window_size: tuple[int, int] = (-1, -1)
) -> torch.Tensor:
    """`CostVolumeDisparityAttention.forward`, with each sequence chunk run through all layers at once."""
    b, c, d, h, w = cv.shape
    x = cv.permute(0, 3, 4, 2, 1).reshape(b * h * w, d, c)
    # on MLX the chunks are joined by `cat`: an exported `empty_like` buffer would be float32 on MLX
    out = None if EXPORTING_FOR_MLX.get() else torch.empty_like(x)
    chunks = []
    for i in range(0, x.shape[0], CHUNK):
        y = self.pos_embed0(x[i : i + CHUNK], resize_embed=self.resize_embed)
        for layer in self.sa:
            y = layer(y, window_size=window_size)
        if out is None:
            chunks.append(y)
        else:
            out[i : i + CHUNK] = y
    out = torch.cat(chunks) if out is None else out
    return out.reshape(b, h, w, d, c).permute(0, 4, 3, 1, 2)


def _padded_attention(  # noqa: PLR0913, PLR0917 - upstream `FlashMultiheadAttention.forward`'s signature
    self: FlashMultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """`FlashMultiheadAttention.forward` with a half-precision CUDA head dim zero-padded to a multiple of 8.

    Same output: the zero columns add nothing to `q . k` nor to the kept `v` columns, and the scale stays
    `head_dim ** -0.5`. PyTorch's flash pads head dim 7 to 8 itself; the memory-efficient and cuDNN backends refuse it,
    so without flash (Windows) the call fell to the math backend (C-FS-S 375x1242: 1706 ms vs 1170 ms on Linux).
    """
    d = self.head_dim
    if not -d % 8 or not query.is_cuda:
        return FlashMultiheadAttention.forward(self, query, key, value, attn_mask, window_size)
    b, n, _ = query.shape
    shape = (b, self.num_heads, n, d) if self.qkv_layout == "standard" else (b, n, self.num_heads, d)
    q, k, v = (proj(x).view(shape) for proj, x in ((self.q_proj, query), (self.k_proj, key), (self.v_proj, value)))
    if q.dtype in (torch.float16, torch.bfloat16):
        q, k, v = (F.pad(x, (0, -d % 8)) for x in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, scale=d**-0.5)[..., :d]
    else:  # float32: upstream's call as is
        o = F.scaled_dot_product_attention(q, k, v)
    return self.out_proj(o.reshape(b, n, -1))


def _post(
    self: _FFSPostForwardHelper, conv_high: torch.Tensor, conv_low: torch.Tensor, left_feat: torch.Tensor | None = None
) -> torch.Tensor:
    """`_FFSPostForwardHelper.forward` without keeping the upsampled volume alive next to the merged one."""
    if self.op == "sum":
        x = self.upsample(conv_low) + conv_high
    else:
        x = torch.cat((self.upsample(conv_low), conv_high), dim=1)
    for layer in self.out:
        x = layer(x, left_feat) if isinstance(layer, FeatureAtt) else layer(x)
    return x


def _use_volume_strips(volume: torch.Tensor) -> bool:
    return (
        volume.is_cuda
        and not (torch.compiler.is_compiling() or torch.jit.is_tracing() or torch.is_grad_enabled())
        and volume.numel() * volume.element_size() >= LARGE_VOLUME_BYTES
    )


def _merge_attention(self: HourGlass, conv: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
    """Interpolate and attend in row strips, then add each strip into the owned decoder output."""
    if not _use_volume_strips(conv):
        return HourGlass.merge_attention(self, conv, patches)

    height = patches.shape[-2]
    rows = max(1, CHUNK // (PATCH_SCALE * conv.shape[-1]))
    attention = self.atts["cost_vol_disp"]
    for start in range(0, height, rows):
        stop = min(height, start + rows)
        first, last = max(0, start - 1), min(height, stop + 1)
        # Align-corners=False needs at most one coarse row of context on each side.
        expanded = F.interpolate(
            patches[..., first:last, :], scale_factor=PATCH_SCALE, mode="trilinear", align_corners=False
        )
        inner = expanded[..., (start - first) * PATCH_SCALE : (stop - first) * PATCH_SCALE, :]
        dst = conv[..., start * PATCH_SCALE : stop * PATCH_SCALE, :]
        dst.add_(attention(inner))
    return conv


def _hourglass_tail(self: nn.Sequential, volume: torch.Tensor) -> torch.Tensor:
    """Filter large inference volumes one height strip at a time."""
    if self.training or not _use_volume_strips(volume):
        return nn.Sequential.forward(self, volume)

    height, width = volume.shape[-2:]
    rows = max(8, CHUNK // width)
    if rows >= height:
        return nn.Sequential.forward(self, volume)
    halo = 2  # two 3 x 3 spatial convolutions
    layout = torch.channels_last_3d
    out = torch.empty_like(volume, memory_format=layout)
    for start in range(0, height, rows):
        stop = min(height, start + rows)
        first, last = max(0, start - halo), min(height, stop + halo)
        strip = volume[..., first:last, :].contiguous(memory_format=layout)
        filtered = nn.Sequential.forward(self, strip)
        out[..., start:stop, :] = filtered[..., start - first : stop - first, :]
        del strip, filtered
    return out


def _conv(self: Conv, x: torch.Tensor) -> torch.Tensor:
    """`Conv.forward` with the LeakyReLU in place (on the fresh conv / norm output)."""
    x = self.conv(x)
    if self.norm is not None:
        x = self.norm(x)
    return F.leaky_relu(x, inplace=True) if self.relu else x


def _channels_last(net: nn.Module) -> None:
    """Conv weights in channels-last layout: cuDNN then runs its NHWC kernels without transposing every activation."""
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
        elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last_3d)


def prepare_portable(net: nn.Module, *, channels_last: bool = False) -> None:
    """Chunk every disparity transformer of `net` and apply the hourglass memory fixes (the output is unchanged).

    With `channels_last` (the portable runner on CUDA) the conv weights go channels-last as well. The in-place LeakyReLU
    is applied to the Fast-FoundationStereo hourglass only: in FoundationStereo it leaves peak allocated unchanged and
    raised peak reserved (C-FS-S Midd-H 6590 -> 6850 MiB).
    """
    if channels_last:
        _channels_last(net)
    fast = any(isinstance(m, _FFSPostForwardHelper) for m in net.modules())
    for m in net.modules():
        if isinstance(m, CostVolumeDisparityAttention):
            override(m, _chunked)
        elif isinstance(m, FlashMultiheadAttention):
            override(m, _padded_attention)
        elif isinstance(m, _FFSPostForwardHelper):
            override(m, _post)
        elif channels_last and isinstance(m, HourGlass):
            override(m, _merge_attention, "merge_attention")
            if is_base(m, HourGlass):
                override(m.conv_out, _hourglass_tail)
        elif fast and isinstance(m, Conv):
            override(m, _conv)


def mps_half(net: nn.Module) -> None:
    """The whole network in fp16 on MPS (TAO's AMP blocks only autocast on CUDA).

    The GRU lookups keep TAO's fp32 (`features.float()`, `disp.float()`); the update block and the upsampling heads
    cast their inputs back to fp16 (`pure_half` on each).
    """
    pure_half(net, channels_last=False)
    for m in (net.update_block, net.spx_2_gru, net.spx_gru):
        pure_half(m, channels_last=False)
