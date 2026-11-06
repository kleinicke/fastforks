"""EdgeNeXt-small backbone (stem + 4 stages, no head) without timm.

The part of timm's `timm/models/edgenext.py` (Apache-2.0, https://github.com/huggingface/pytorch-image-models; model by
Maaz et al., https://github.com/mmaaz60/EdgeNeXt) that FoundationStereo's `Feature` uses via
`timm.create_model('edgenext_small')`: the same modules, parameter names and maths, so the checkpoint keys
(`feature.stem.*`, `feature.stages.*`) are unchanged. Reduced to the edgenext_small configuration; `timm.layers`
helpers (`Mlp`, `LayerNorm2d`, `create_conv2d`, DropPath) are inlined, training-only parts (drop path, dropout,
grad checkpointing) and the classifier head are removed.
"""

import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channels of an NCHW tensor (timm's `LayerNorm2d`).

    `contiguous`: return an NCHW-contiguous tensor instead of the channels-last view (the values are the same; the
    following convolution picks a different CPU algorithm, so the last float bits differ).
    """

    contiguous = False

    def forward(self, x):
        y = F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        return y.contiguous() if self.contiguous else y


class Mlp(nn.Module):
    """timm's `Mlp`: fc1, activation, fc2."""

    def __init__(self, in_features, hidden_features, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def dw_conv(channels, kernel_size):
    """timm's `create_conv2d(..., depthwise=True)` at stride 1: 'same' zero padding k // 2."""
    return nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)


class PositionalEncodingFourier(nn.Module):
    def __init__(self, hidden_dim=32, dim=768, temperature=10000.0):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, shape):
        device = self.token_projection.weight.device
        dtype = self.token_projection.weight.dtype
        inv_mask = ~torch.zeros(shape).to(device=device, dtype=torch.bool)
        y_embed = inv_mask.cumsum(1, dtype=torch.float32)
        x_embed = inv_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.int64, device=device).to(torch.float32)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return self.token_projection(pos.to(dtype))


class ConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, expand_ratio=4, ls_init_value=1e-6, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.conv_dw = dw_conv(dim, kernel_size)
        self.norm = norm_layer(dim)
        self.mlp = Mlp(dim, int(expand_ratio * dim))
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim))

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.mlp(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return shortcut + x


class CrossCovarianceAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 4, 1)
        q, k, v = qkv.unbind(0)
        # not a spatial attention: q, k, v are B, num_heads, C, L  ->  C x C attention map
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.permute(0, 3, 1, 2).reshape(B, N, C)
        return self.proj(x)


class SplitTransposeBlock(nn.Module):
    def __init__(
        self, dim, num_scales=1, num_heads=8, expand_ratio=4, use_pos_emb=True, qkv_bias=True, ls_init_value=1e-6,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        width = max(int(math.ceil(dim / num_scales)), int(math.floor(dim // num_scales)))
        self.width = width
        self.num_scales = max(1, num_scales - 1)
        self.convs = nn.ModuleList([dw_conv(width, 3) for _ in range(self.num_scales)])
        self.pos_embd = PositionalEncodingFourier(dim=dim) if use_pos_emb else None
        self.norm_xca = norm_layer(dim)
        self.gamma_xca = nn.Parameter(ls_init_value * torch.ones(dim))
        self.xca = CrossCovarianceAttn(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm = norm_layer(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(expand_ratio * dim))
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim))

    def forward(self, x):
        shortcut = x
        spx = x.chunk(len(self.convs) + 1, dim=1)
        spo = []
        sp = spx[0]
        for i, conv in enumerate(self.convs):
            if i > 0:
                sp = sp + spx[i]
            sp = conv(sp)
            spo.append(sp)
        spo.append(spx[-1])
        x = torch.cat(spo, 1)

        # XCA
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        if self.pos_embd is not None:
            pos_encoding = self.pos_embd((B, H, W)).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding
        x = x + self.gamma_xca * self.xca(self.norm_xca(x))
        x = x.reshape(B, H, W, C)

        # inverted bottleneck
        x = self.norm(x)
        x = self.mlp(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return shortcut + x


class EdgeNeXtStage(nn.Module):
    def __init__(
        self, in_chs, out_chs, stride=2, depth=2, num_global_blocks=1, num_heads=4, scales=2, kernel_size=7,
        expand_ratio=4, use_pos_emb=False, ls_init_value=1.0,
    ):
        super().__init__()
        if stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(LayerNorm2d(in_chs, eps=1e-6), nn.Conv2d(in_chs, out_chs, kernel_size=2, stride=2))
            in_chs = out_chs
        blocks = []
        for i in range(depth):
            if i < depth - num_global_blocks:
                blocks.append(ConvBlock(in_chs, kernel_size=kernel_size, expand_ratio=expand_ratio, ls_init_value=ls_init_value))
            else:
                blocks.append(
                    SplitTransposeBlock(
                        in_chs, num_scales=scales, num_heads=num_heads, expand_ratio=expand_ratio,
                        use_pos_emb=use_pos_emb, ls_init_value=ls_init_value,
                    )
                )
            in_chs = out_chs
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(self.downsample(x))


class EdgeNeXtSmall(nn.Module):
    """timm `edgenext_small` without head: `stem` (4x4 patch conv + LayerNorm2d) and four `stages`."""

    dims = (48, 96, 160, 304)
    depths = (3, 3, 9, 3)
    global_block_counts = (0, 1, 1, 1)
    kernel_sizes = (3, 5, 7, 9)
    heads = (8, 8, 8, 8)
    d2_scales = (2, 2, 3, 4)
    use_pos_emb = (False, True, False, False)

    def __init__(self, contiguous_norm=False):
        super().__init__()
        dims = self.dims
        self.stem = nn.Sequential(nn.Conv2d(3, dims[0], kernel_size=4, stride=4), LayerNorm2d(dims[0], eps=1e-6))
        stages = []
        in_chs = dims[0]
        for i in range(4):
            stages.append(
                EdgeNeXtStage(
                    in_chs, dims[i], stride=2 if i > 0 else 1, depth=self.depths[i],
                    num_global_blocks=self.global_block_counts[i], num_heads=self.heads[i], scales=self.d2_scales[i],
                    expand_ratio=4, kernel_size=self.kernel_sizes[i], use_pos_emb=self.use_pos_emb[i], ls_init_value=1e-6,
                )
            )
            in_chs = dims[i]
        self.stages = nn.Sequential(*stages)
        for m in self.modules():
            if isinstance(m, LayerNorm2d):
                m.contiguous = contiguous_norm
