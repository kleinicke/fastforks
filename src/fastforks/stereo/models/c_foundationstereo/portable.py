"""Portable-mode memory fixes (same math).

TAO's disparity transformer runs in chunks of its (B*H*W) disparity sequences; the cost-volume hourglass frees its
upsampled volume right after merging it and applies LeakyReLU in place.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.patch import override
from fastforks._core.tracing import EXPORTING_FOR_MLX

from .net.fast_foundation_stereo.fast_foundation_stereo import _FFSPostForwardHelper
from .net.foundation_stereo.convolution_helper import Conv
from .net.foundation_stereo.submodule import CostVolumeDisparityAttention, FeatureAtt, HourGlass

# sequences per chunk: memory bound (Q, K, V, out of one chunk are ~200 MB at 375x1242); flash SDPA takes < 65536; the
# result does not depend on it
CHUNK = 1 << 13
ATTENTION_VOLUME_LIMIT = 256 * 1024**2
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


def _merge_attention(self: HourGlass, conv: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
    """Interpolate and attend in row strips, then add each strip into the owned decoder output."""
    if (
        not conv.is_cuda
        or torch.compiler.is_compiling()
        or torch.jit.is_tracing()
        or torch.is_grad_enabled()
        or conv.numel() * conv.element_size() < ATTENTION_VOLUME_LIMIT
    ):
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
        elif isinstance(m, _FFSPostForwardHelper):
            override(m, _post)
        elif channels_last and isinstance(m, HourGlass):
            override(m, _merge_attention, "merge_attention")
        elif fast and isinstance(m, Conv):
            override(m, _conv)
