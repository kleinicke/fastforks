"""Kernels inference path for CodeFormer (general: any batch, the aligned 512 x 512 crop; no per-size compilation).

`prepare_kernels(net, device)` (fp32 network on `device`), then `forward_kernels(net, x, w)` (`x` in [-1, 1], fp16):
  - pure fp16 weights and NHWC activations; every convolution of the VQGAN encoder / generator and the fusion blocks
    through `cnn_kernels.conv_ep` (fp16-accumulate cuDNN graphs, or CUTLASS for >= 128 output channels)
  - the pre-activation `ResBlock`s as GroupNorm + SiLU, conv + bias, conv + bias + residual (the skip added in the
    epilogue); the fusion blocks' conv + LeakyReLU runs as one epilogue each
  - the 16 x 16 transformer as in portable (one SDPA call per attention); logits head, code lookup and the adaptive
    instance normalisation of the quantised features in float32
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.patch import override
from fastforks.kernels.group_silu import group_silu
from fastforks.shared import cnn_kernels
from fastforks.shared.cnn_kernels import CL, conv_ep

from ..net.codeformer_arch import calc_mean_std
from ..net.vqgan_arch import Downsample, ResBlock

if TYPE_CHECKING:
    from ..net import CodeFormer


def _resblock_forward(self: ResBlock, x_in: torch.Tensor) -> torch.Tensor:
    """GroupNorm -> SiLU -> conv -> GroupNorm -> SiLU -> conv, + skip (1 x 1 conv or identity), the sum in the epilogue."""
    skip = x_in if self.in_channels == self.out_channels else conv_ep(self.conv_out, x_in)
    x = conv_ep(self.conv1, group_silu(self.norm1, x_in, inplace=False))
    return conv_ep(self.conv2, group_silu(self.norm2, x, inplace=True), r=skip.contiguous(memory_format=CL))


def _downsample_forward(self: Downsample, x: torch.Tensor) -> torch.Tensor:
    return conv_ep(self.conv, F.pad(x, (0, 1, 0, 1)).contiguous(memory_format=CL))


def prepare_kernels(net: CodeFormer, device: torch.device) -> None:
    """Switch `net` (fp32, eval, on `device`) to the kernels path in place.

    Raises:
        KernelUnavailableError: A kernel library is unusable on `device`.
    """
    for m in net.modules():
        if isinstance(m, ResBlock):
            override(m, _resblock_forward)
        elif isinstance(m, Downsample):
            override(m, _downsample_forward)
    cnn_kernels.prepare_generic(net, device)


def _adain(content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
    """`adaptive_instance_normalization` (float32 statistics) of two small channels-last maps."""
    content, style = content.float().contiguous(), style.float().contiguous()
    s_mean, s_std = calc_mean_std(style)
    c_mean, c_std = calc_mean_std(content)
    return ((content - c_mean) / c_std * s_std + s_mean).contiguous(memory_format=CL)


@torch.no_grad()
def forward_kernels(net: CodeFormer, x: torch.Tensor, w: float) -> torch.Tensor:
    """`CodeFormer.forward(x, w=w, adain=True)[0]` at inference; `x` fp16 [-1, 1]."""
    m: nn.Module = net
    x = x.contiguous(memory_format=CL)
    enc = {}
    taps = {m.fuse_encoder_block[s] for s in m.connect_list}
    for i, block in enumerate(m.encoder.blocks):
        x = block(x)
        if w > 0 and i in taps:
            enc[str(x.shape[-1])] = x
    lq = x
    b = x.shape[0]
    pos = m.position_emb.unsqueeze(1).expand(-1, b, -1)
    q = m.feat_emb(lq.flatten(2).permute(2, 0, 1))
    for layer in m.ft_layers:
        q = layer(q, query_pos=pos)
    ln, fc = m.idx_pred_layer
    logits = F.linear(
        F.layer_norm(q.float(), ln.normalized_shape, ln.weight.float(), ln.bias.float(), ln.eps), fc.weight.float()
    )
    idx = logits.argmax(dim=2).transpose(0, 1)
    quant = F.embedding(idx, m.quantize.embedding.weight).view(b, 16, 16, -1).permute(0, 3, 1, 2)
    x = _adain(quant, lq).to(lq.dtype)
    fuse = {m.fuse_generator_block[s] for s in m.connect_list}
    for i, block in enumerate(m.generator.blocks):
        x = block(x)
        if i in fuse and w > 0:
            x = m.fuse_convs_dict[str(x.shape[-1])](enc.pop(str(x.shape[-1])), x, w)
    return x
