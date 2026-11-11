"""NeuFlow v2 network (Zhang, Gupta, Singh & Jiang, 2024) as neufieldrobotics/NeuFlow_v2 builds it (`NeuFlow/`).

Inference only.

Ported from neufieldrobotics/NeuFlow_v2 (Apache-2.0, see `LICENSE`) with the same module tree, so the released
checkpoints load as they are. Changes: no huggingface_hub import (the hub mixin); no `init_bhwd` (the grids, the
position channels and the zero iteration context are made in the forward pass, so any batch and size run); the input
is already scaled to [0, 1] and padded (the core does both, as the card's `[io]` says); the flow is upsampled once,
after the last 1/8 iteration (upstream's last list entry); with `fp16=True` the convolutions and the cross-attention
run under fp16 autocast while the global matching, the correlation, its lookup, the flow and the convex upsampling
stay float32 (upstream's own half mode keeps coordinates and correlation in fp16 too).

- Backbone: conv-bn-LeakyReLU blocks on the image averaged down by 2, 4 and 8 give 1/8 and 1/16 features of both
  images (192 channels: 64 context + 128 matching; the 1/16 ones end with two position channels).
- 1/16: two cross-attention layers between the images, global matching (softmax over all positions of image 2) for
  the initial flow, then `ITERS_S16` refinements: a 9 x 9 lookup in the all-pairs correlation around the current
  match, eight 3 x 3 convs on [correlation | context | iteration context | flow | radius], a delta flow.
- 1/8: flow and 1/16 features upsampled (nearest) and merged with the 1/8 features, `iters` refinements as above.
- Upsampling: convex combination of 3 x 3 neighbours with weights predicted from an 8 x 8 patch embedding of image 1.
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn.functional as F
from torch import nn

FACTOR = 16  # fixed: the coarse level works at 1/16 resolution
FEATURE, CONTEXT, ITER_CONTEXT = 128, 64, 64  # fixed: upstream NeuFlow/config.py (the same at 1/16 and 1/8)
RADIUS = 4  # fixed: correlation lookup radius (one level), upstream neuflow.py
ITERS_S16 = 1  # fixed: refinements at 1/16, upstream's `iters_s16` default (all its evaluations)


class ConvBlock(nn.Module):
    """conv-bn-LeakyReLU(0.1), then a 3 x 3 conv-bn-LeakyReLU(0.1) (upstream backbone_v7.ConvBlock)."""

    def __init__(self, cin: int, cout: int, kernel_size: int, stride: int, padding: int) -> None:
        """Build the block."""
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, kernel_size, stride, padding, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.relu = nn.LeakyReLU(0.1)
        self.norm1 = nn.BatchNorm2d(cout)
        self.norm2 = nn.BatchNorm2d(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Two conv-bn-activation steps."""
        x = self.relu(self.norm1(self.conv1(x)))
        return self.relu(self.norm2(self.conv2(x)))


class CNNEncoder(nn.Module):
    """1/16 and 1/8 features from the image averaged down by 2, 4 and 8 (upstream backbone_v7.CNNEncoder)."""

    def __init__(self) -> None:
        """Build the blocks."""
        super().__init__()
        dim = FEATURE + CONTEXT
        self.block_8_1 = ConvBlock(3, FEATURE * 2, 8, 4, 2)
        self.block_8_2 = ConvBlock(3, FEATURE, 6, 2, 2)
        self.block_cat_8 = ConvBlock(FEATURE * 3, dim, 3, 1, 1)
        self.block_16_1 = ConvBlock(3, FEATURE, 6, 2, 2)
        self.block_8_16 = ConvBlock(dim, FEATURE, 6, 2, 2)
        self.block_cat_16 = ConvBlock(FEATURE * 2, dim - 2, 3, 1, 1)

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`[N, 3, H, W]` -> 1/16 features (the last two channels: centred row and column index) and 1/8 features."""
        img = F.avg_pool2d(img, 2, 2)
        x_8 = self.block_8_1(img)
        img = F.avg_pool2d(img, 2, 2)
        x_8 = self.block_cat_8(torch.cat([x_8, self.block_8_2(img)], dim=1))
        img = F.avg_pool2d(img, 2, 2)
        x_16 = self.block_cat_16(torch.cat([self.block_16_1(img), self.block_8_16(x_8)], dim=1))
        n, _, h, w = x_16.shape
        ys = torch.arange(h, device=x_16.device, dtype=x_16.dtype) - h / 2
        xs = torch.arange(w, device=x_16.device, dtype=x_16.dtype) - w / 2
        pos = torch.stack([ys[:, None].expand(h, w), xs[None].expand(h, w)])
        return torch.cat([x_16, pos.expand(n, 2, h, w)], dim=1), x_8


class TransformerLayer(nn.Module):
    """Single-head cross-attention and an MLP on [source | message], post-norm (upstream transformer.py)."""

    def __init__(self, dim: int) -> None:
        """Build the layer."""
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.merge = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim * 2, bias=False), nn.GELU(), nn.Linear(dim * 2, dim, bias=False)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """`[N, L, C]` tokens attend to `target`."""
        # Explicit singleton heads preserve the upstream single-head operation and permit ONNX conversion.
        message = F.scaled_dot_product_attention(
            self.q_proj(source).unsqueeze(1), self.k_proj(target).unsqueeze(1), self.v_proj(target).unsqueeze(1)
        ).squeeze(1)
        message = self.norm1(self.merge(message))
        return source + self.norm2(self.mlp(torch.cat([source, message], dim=-1)))


class FeatureAttention(nn.Module):
    """Cross-attention between the two images of each pair, then a batch norm (upstream transformer.py)."""

    def __init__(self, dim: int, num_layers: int) -> None:
        """Build the layers."""
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(dim) for _ in range(num_layers)])
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[2B, C, H, W]` (images 1, then images 2) -> the same shape."""
        n, c, h, w = x.shape
        x = x.flatten(-2).permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x, torch.cat(x.chunk(2, dim=0)[::-1], dim=0))
        return self.norm(x.reshape(n, h, w, c).permute(0, 3, 1, 2))


class _Conv(nn.Module):
    """3 x 3 conv (no bias) and LeakyReLU(0.1) (upstream refine.ConvBlock)."""

    def __init__(self, cin: int, cout: int) -> None:
        """Build the conv."""
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, 1, 1, bias=False)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Conv and activation."""
        return self.relu(self.conv(x))


class Refine(nn.Module):
    """One flow update: convs on [correlation | context | iteration context | flow | radius] (upstream refine.py)."""

    def __init__(self, inter_dim: int, num_layers: int = 5) -> None:
        """Build the convs."""
        super().__init__()
        dim = CONTEXT + ITER_CONTEXT
        self.conv1 = _Conv((RADIUS * 2 + 1) ** 2 + dim + 2 + 1, dim)
        self.conv2 = _Conv(dim, inter_dim)
        self.conv_layers = nn.ModuleList([_Conv(inter_dim, inter_dim) for _ in range(num_layers)])
        self.conv3 = nn.Conv2d(inter_dim, ITER_CONTEXT + 2, 3, 1, 1)

    def forward(
        self, corrs: torch.Tensor, context: torch.Tensor, iter_context: torch.Tensor, flow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The next iteration context (clamped to [-4, 4]) and the delta flow."""
        radius = torch.full_like(flow[:, :1], RADIUS)
        x = self.conv2(self.conv1(torch.cat([corrs, context, iter_context, flow, radius], dim=1)))
        for layer in self.conv_layers:
            x = layer(x)
        x = self.conv3(x)
        return F.hardtanh(x[:, 2:], -4.0, 4.0), x[:, :2]


class UpSample(nn.Module):
    """Convex upsampling by 8 with weights from the image-1 features and the flow (upstream upsample.py)."""

    def __init__(self, dim: int, factor: int = 8) -> None:
        """Build the weight head."""
        super().__init__()
        self.factor = factor
        self.conv1 = nn.Conv2d(2 + dim, 256, 3, 1, 1)
        self.conv2 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv3 = nn.Conv2d(512, factor**2 * 9, 1, 1, 0)

    def forward(self, feature: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """`[B, 2, h, w]` -> `[B, 2, 8h, 8w]` (not scaled: the caller multiplies by 8)."""
        mask = self.conv3(F.relu(self.conv2(F.relu(self.conv1(torch.cat([flow, feature], dim=1))))))
        b, _, h, w = flow.shape
        k = self.factor
        mask = torch.softmax(mask.float().view(b, 1, 9, k, k, h, w), dim=2)
        up = F.unfold(flow, (3, 3), padding=1).view(b, 2, 9, 1, 1, h, w)
        up = torch.sum(mask * up, dim=2).permute(0, 1, 4, 2, 5, 3)
        return up.reshape(b, 2, k * h, k * w)


def _merge(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.GELU(), nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
        nn.BatchNorm2d(cout),
    )  # fmt: skip


def _grid(flow: torch.Tensor) -> torch.Tensor:
    """Pixel coordinates `[1, 2, H, W]` (x, y) on the flow's grid."""
    h, w = flow.shape[-2:]
    ys = torch.arange(h, device=flow.device, dtype=flow.dtype)
    xs = torch.arange(w, device=flow.device, dtype=flow.dtype)
    return torch.stack([xs[None].expand(h, w), ys[:, None].expand(h, w)])[None]


def _split(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`[2B, 192, H, W]` -> the matching features of both images and the context (ReLU) of image 1."""
    context, features = torch.split(features, [CONTEXT, FEATURE], dim=1)
    return features, torch.relu(context.chunk(2, dim=0)[0])


def _match(feature0: torch.Tensor, feature1: torch.Tensor) -> torch.Tensor:
    """Global matching: the softmax-weighted position in image 2 minus the own position (upstream matching.py)."""
    b, _, h, w = feature0.shape
    grid = _grid(feature0)
    f0, f1 = (f.flatten(-2).permute(0, 2, 1) for f in (feature0, feature1))
    # ONNX's SDPA translator requires [batch, heads, tokens, channels], including for one head.
    values = grid.flatten(-2).permute(0, 2, 1).expand(b, h * w, 2)
    match = F.scaled_dot_product_attention(f0.unsqueeze(1), f1.unsqueeze(1), values.unsqueeze(1)).squeeze(1)
    return match.view(b, h, w, 2).permute(0, 3, 1, 2) - grid


def _correlation(feature0: torch.Tensor, feature1: torch.Tensor) -> torch.Tensor:
    """All-pairs correlation `[B * H * W, 1, H, W]`: dot product / sqrt(C) (upstream corr.py, one level)."""
    b, c, h, w = feature0.shape
    corr = torch.matmul(feature0.view(b, c, h * w).transpose(1, 2), feature1.view(b, c, h * w))
    return corr.view(b * h * w, 1, h, w) / math.sqrt(c)


def _lookup(corr: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """The (2r + 1)^2 correlation window around each pixel's current match, bilinear, zeros outside: `[B, 81, H, W]`.

    As upstream (and RAFT), the window's first axis steps in x: channel `i * 9 + j` is the offset (x, y) = (i - r,
    j - r).
    """
    b, _, h, w = flow.shape
    r = torch.linspace(-RADIUS, RADIUS, 2 * RADIUS + 1, device=flow.device, dtype=flow.dtype)
    delta = torch.stack(torch.meshgrid(r, r, indexing="ij"), dim=-1)
    coords = (_grid(flow) + flow).permute(0, 2, 3, 1).reshape(b * h * w, 1, 1, 2) + delta
    x, y = coords.unbind(-1)
    grid = torch.stack([2 * x / (w - 1) - 1, 2 * y / (h - 1) - 1], dim=-1)
    out = F.grid_sample(corr, grid, align_corners=True)
    return out.view(b, h, w, -1).permute(0, 3, 1, 2)


class NeuFlow(nn.Module):
    """NeuFlow v2: `forward(image1, image2, iters)` -> the flow `[B, 2, H, W]` in pixels."""

    def __init__(self) -> None:
        """Build the network."""
        super().__init__()
        dim = FEATURE + CONTEXT
        self.backbone = CNNEncoder()
        self.cross_attn_s16 = FeatureAttention(dim, num_layers=2)
        self.merge_s8 = _merge(FEATURE * 2, FEATURE)
        self.context_merge_s8 = _merge(CONTEXT * 2, CONTEXT)
        self.refine_s16 = Refine(inter_dim=128)
        self.refine_s8 = Refine(inter_dim=96)
        self.conv_s8 = ConvBlock(3, FEATURE, 8, 8, 0)
        self.upsample_s8 = UpSample(FEATURE)
        self.fp16 = False  # fp16 autocast for the convolutions and the cross-attention (CUDA)

    def _amp(self, device: torch.device) -> contextlib.AbstractContextManager:
        return torch.autocast(device.type, torch.float16) if self.fp16 else contextlib.nullcontext()

    def _refine(
        self, refine: Refine, features: torch.Tensor, context: torch.Tensor, flow: torch.Tensor, iters: int
    ) -> torch.Tensor:
        corr = _correlation(*features.float().chunk(2, dim=0))
        iter_context = context.new_zeros(context.shape[0], ITER_CONTEXT, *context.shape[-2:])
        for _ in range(iters):
            corrs = _lookup(corr, flow)
            with self._amp(flow.device):
                iter_context, delta = refine(corrs, context, iter_context, flow)
            flow = flow + delta.float()
        return flow

    def forward(self, image1: torch.Tensor, image2: torch.Tensor, iters: int) -> torch.Tensor:
        """Flow from `image1` to `image2` (`[B, 3, H, W]` in [0, 1], H and W multiples of 16)."""
        device = image1.device
        with self._amp(device):
            features_s16, features_s8 = self.backbone(torch.cat([image1, image2], dim=0))
            features_s16, context_s16 = _split(self.cross_attn_s16(features_s16))
            features_s8, context_s8 = _split(features_s8)
        flow = _match(*features_s16.float().chunk(2, dim=0))
        flow = self._refine(self.refine_s16, features_s16, context_s16, flow, ITERS_S16)
        flow = F.interpolate(flow, scale_factor=2, mode="nearest") * 2
        with self._amp(device):
            features_s16 = F.interpolate(features_s16, scale_factor=2, mode="nearest")
            features_s8 = self.merge_s8(torch.cat([features_s8, features_s16], dim=1))
            context_s16 = F.interpolate(context_s16, scale_factor=2, mode="nearest")
            context_s8 = self.context_merge_s8(torch.cat([context_s8, context_s16], dim=1))
        flow = self._refine(self.refine_s8, features_s8, context_s8, flow, iters)
        with self._amp(device):
            feature = self.conv_s8(image1)
            return self.upsample_s8(feature, flow) * 8
