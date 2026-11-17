# Copyright 2024 DeepMind Technologies Limited
# Copyright 2024 KU-CVLAB (LocoTrack)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# fastforks: upstream `locotrack_pytorch/models/{locotrack_model,nets,utils,cmdtop}.py` for one video at the training
# resolution (256 x 256: one feature grid for the initialisation and the refinement), offline, einops removed. Exact
# simplifications: the mixer's feature outputs (`res[..., 4:]`, carried between iterations but never read by the
# mixer or the outputs) are not computed, and CMDTop runs its two directions as one batch (same convs, per-sample
# GroupNorm). Parameter names as upstream (its state dicts load unchanged after stripping `model.`).
"""LocoTrack (offline point tracker, 4D-correlation refinement) for one 256 x 256 video in `[-1, 1]`."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

SIZE = 256  # training resolution; the network runs at it
PIPS_ITERS = 4
SOFTMAX_TEMPERATURE = 20.0
SUPPORT = 7  # 7 x 7 local neighbourhoods (query support and target)
POSENC_DEGREES = 10
HIRES, LOWRES = 128, 256
# model size -> (mixer width, heads, CMDTop channels, kernels, strides)
SIZES = {
    "small": (256, 4, (64, 128), (5, 2), (4, 2)),
    "base": (384, 6, (64, 128, 128), (3, 3, 2), (2, 2, 2)),
}


class Conv2dSamePadding(nn.Conv2d):
    """Conv2d with TensorFlow "SAME" padding (the extra pixel right / bottom), computed per input size."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        pads = []
        for i, k, s in zip(reversed(x.shape[-2:]), reversed(self.kernel_size), reversed(self.stride), strict=True):
            p = max((math.ceil(i / s) - 1) * s + k - i, 0)
            pads += [p // 2, p - p // 2]
        if any(pads):
            x = F.pad(x, pads)
        return F.conv2d(x, self.weight, self.bias, self.stride)


def same_conv(cin: int, cout: int, k: int, s: int, *, bias: bool = True) -> Conv2dSamePadding:  # noqa: D103
    return Conv2dSamePadding(cin, cout, k, s, padding=0, bias=bias)


class BlockV2(nn.Module):
    """Pre-activation ResNet block with instance norms."""

    def __init__(self, cin: int, cout: int, stride: int, use_projection: bool) -> None:  # noqa: D107, FBT001
        super().__init__()
        self.proj_conv = same_conv(cin, cout, 1, stride, bias=False) if use_projection else None
        self.bn_0 = nn.InstanceNorm2d(cin, affine=True)
        self.conv_0 = same_conv(cin, cout, 3, stride, bias=False)
        self.conv_1 = same_conv(cout, cout, 3, 1, bias=False)
        self.bn_1 = nn.InstanceNorm2d(cout, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        shortcut = x
        x = torch.relu(self.bn_0(x))
        if self.proj_conv is not None:
            shortcut = self.proj_conv(x)
        x = torch.relu(self.bn_1(self.conv_0(x)))
        return self.conv_1(x) + shortcut


class BlockGroup(nn.Module):
    """Blocks of one resolution."""

    def __init__(self, cin: int, cout: int, num_blocks: int, stride: int) -> None:  # noqa: D107
        super().__init__()
        self.blocks = nn.ModuleList(
            BlockV2(cin if i == 0 else cout, cout, 1 if i else stride, use_projection=i == 0) for i in range(num_blocks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        for block in self.blocks:
            x = block(x)
        return x


class ResNet(nn.Module):
    """ResNet-18-like backbone; returns units 0 (stride 2, 64 ch), 1 (stride 4, 128 ch) and 3 (stride 8, 256 ch)."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.initial_conv = same_conv(3, 64, 7, 2, bias=False)
        channels, strides = (64, HIRES, 256, LOWRES), (1, 2, 2, 1)
        self.block_groups = nn.ModuleList(
            BlockGroup(channels[i - 1] if i else 64, channels[i], 2, strides[i]) for i in range(4)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`x` `[N, 3, H, W]` -> (unit 0, unit 1, unit 3), NCHW."""
        x = self.initial_conv(x)
        outs = []
        for group in self.block_groups:
            x = group(x)
            outs.append(x)
        return outs[0], outs[1], outs[3]


class CMDTop(nn.Module):
    """Convs over the 4D correlation `[B, 7, 7, 7, 7]` in both directions, spatially averaged: `[B, 2 * C]`."""

    def __init__(self, channels: tuple[int, ...], kernels: tuple[int, ...], strides: tuple[int, ...]) -> None:  # noqa: D107
        super().__init__()
        cins = (SUPPORT * SUPPORT, *channels[:-1])
        self.conv = nn.ModuleList(
            nn.Sequential(same_conv(ci, co, k, s), nn.GroupNorm(co // 16, co), nn.ReLU())
            for ci, co, k, s in zip(cins, channels, kernels, strides, strict=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x` `[B, h, w, i, j]` (target neighbourhood x query support)."""
        b, h, w, i, j = x.shape
        y = torch.cat([x.permute(0, 3, 4, 1, 2).reshape(b, i * j, h, w), x.reshape(b, h * w, i, j)])
        for conv in self.conv:
            y = conv(y)
        y = y.mean((2, 3))
        return torch.cat([y[:b], y[b:]], -1)


def alibi_bias(heads: int, t: int, device: torch.device) -> torch.Tensor:
    """`[heads, t, t]`: half the heads attend to past frames, half to future ones, with ALiBi distance slopes."""
    half = heads // 2
    base = 24 ** (1 / half)
    slope = torch.tensor([1 / base ** (i + 1) for i in range(half)], device=device)[:, None, None]
    ar = torch.arange(t, device=device)
    rel = (ar[None, :] - ar[:, None]).float()  # [query, key] key - query
    neg = torch.full((t, t), float("-inf"), device=device)
    fwd = slope * torch.tril(rel) + torch.triu(neg, 1)
    bwd = slope * torch.triu(-rel) + torch.tril(neg, -1)
    return torch.cat([fwd, bwd])


class MultiHeadAttention(nn.Module):
    """Attention over the frames of each track with the ALiBi bias."""

    def __init__(self, heads: int, dim: int) -> None:  # noqa: D107
        super().__init__()
        self.heads = heads
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.final_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:  # noqa: D102
        b, t, c = x.shape

        def split(y: torch.Tensor) -> torch.Tensor:
            return y.view(b, t, self.heads, -1).transpose(1, 2)

        q, k, v = split(self.query_proj(x)), split(self.key_proj(x)), split(self.value_proj(x))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        return self.final_proj(y.transpose(1, 2).reshape(b, t, c))


class Transformer(nn.Module):
    """Pre-LayerNorm transformer stack (exact-GELU MLP x4) with a final LayerNorm."""

    def __init__(self, heads: int, dim: int, num_layers: int = 3) -> None:  # noqa: D107
        super().__init__()
        self.layers = nn.ModuleList(
            nn.ModuleDict({
                "attn": MultiHeadAttention(heads, dim),
                "dense": nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)),
                "layer_norm1": nn.LayerNorm(dim),
                "layer_norm2": nn.LayerNorm(dim),
            })
            for _ in range(num_layers)
        )
        self.ln_out = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:  # noqa: D102
        for layer in self.layers:
            h = h + layer["attn"](layer["layer_norm1"](h), bias)
            h = h + layer["dense"](layer["layer_norm2"](h))
        return self.ln_out(h)


class PIPSTransformer(nn.Module):
    """The refinement mixer over the frames of each track."""

    def __init__(self, cin: int, cout: int, dim: int, heads: int) -> None:  # noqa: D107
        super().__init__()
        self.heads = heads
        self.transformer = Transformer(heads, dim)
        self.input_proj = nn.Linear(cin, dim)
        self.output_proj = nn.Linear(dim, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x` `[N, T, cin]` -> the position, occlusion and expected-distance updates `[N, T, 4]`."""
        h = self.transformer(self.input_proj(x), alibi_bias(self.heads, x.shape[1], x.device))
        return F.linear(h, self.output_proj.weight[:4], self.output_proj.bias[:4])


class Features(NamedTuple):
    """Unit-normalised features of every frame (`[1, T, h, w, C]`) and the query supports (`[1, Q, 7, 7, C]`)."""

    grids: tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # hires (64^2, 128), lowres (32^2, 256), highest (128^2, 64)
    supports: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt(torch.clamp(torch.sum(torch.square(x), -1, keepdim=True), min=1e-12))


def map_coordinates_3d(feats: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Trilinear samples `[B, N, C]` of `feats` `[B, T, H, W, C]` at `coords` `[B, N, 3]` `(t, y, x)` (raster)."""
    x = feats.permute(0, 4, 1, 2, 3)
    y = coords[:, :, None, None, :].float().clone()
    y[..., 0] += 0.5
    y = 2 * (y / torch.tensor(x.shape[2:], device=y.device)) - 1
    out = F.grid_sample(x, y.flip(-1), mode="bilinear", align_corners=False, padding_mode="border")
    return out.squeeze(4).squeeze(3).permute(0, 2, 1)


def map_coordinates_2d(feats: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Bilinear samples `[B, P, T, S, C]` of `feats` `[B, T, H, W, C]` at `coords` `[B, P, T, S, 2]` `(y, x)`."""
    n, t, h, w, c = feats.shape
    x = feats.permute(0, 1, 4, 2, 3).reshape(n * t, c, h, w)
    n, p, t, s, _ = coords.shape
    y = coords.permute(0, 2, 1, 3, 4).reshape(n * t, p, s, 2)
    y = (2 * (y / torch.tensor([h, w], device=y.device)) - 1).flip(-1).to(x.dtype)
    out = F.grid_sample(x, y, mode="bilinear", align_corners=False, padding_mode="zeros")
    return out.permute(0, 2, 3, 1).reshape(n, t, p, s, -1).permute(0, 2, 1, 3, 4)


def soft_argmax(heat: torch.Tensor, threshold: float = 5) -> torch.Tensor:
    """`(x, y)` `[B, N, T, 2]` (grid cells, centres at +0.5): the mean of `heat` within `threshold` of its argmax."""
    b, n, t, h, w = heat.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=heat.device), torch.arange(w, device=heat.device), indexing="ij")
    coords = torch.stack([xx + 0.5, yy + 0.5], -1)  # [h, w, 2]
    pos = coords.reshape(-1, 2)[heat.reshape(b, n, t, -1).argmax(-1)]  # [b, n, t, 2]
    valid = (coords - pos[:, :, :, None, None]).square().sum(-1, keepdim=True) < threshold**2
    weight = valid * heat[..., None]
    total = (coords * weight).sum((3, 4))
    return total / torch.clamp(weight.sum((3, 4)), min=1e-12)


def posenc(x: torch.Tensor, degrees: int = POSENC_DEGREES) -> torch.Tensor:
    """`x` and `sin` / `cos` of `x * 2^0 .. 2^(degrees - 1)` (upstream order)."""
    scales = 2.0 ** torch.arange(degrees, device=x.device, dtype=x.dtype)
    xb = (x[..., None, :] * scales[:, None]).flatten(-2)
    return torch.cat([x, torch.sin(torch.cat([xb, xb + 0.5 * math.pi], -1))], -1)


def _offsets(device: torch.device) -> torch.Tensor:
    """The 49 `(dy, dx)` offsets of a 7 x 7 neighbourhood, row-major."""
    r = torch.arange(-(SUPPORT // 2), SUPPORT // 2 + 1, device=device)
    dy, dx = torch.meshgrid(r, r, indexing="ij")
    return torch.stack([dy, dx], -1).reshape(-1, 2)


class LocoTrack(nn.Module):
    """LocoTrack at 256 x 256: initialisation from two cost volumes, then 4 refinements on local 4D correlations."""

    def __init__(self, model_size: str = "base") -> None:  # noqa: D107
        super().__init__()
        dim, heads, channels, kernels, strides = SIZES[model_size]
        self.resnet_torch = ResNet()
        self.torch_pips_mixer = PIPSTransformer(854, 4 + HIRES + LOWRES, dim, heads)
        self.cmdtop = nn.ModuleList(CMDTop(channels, kernels, strides) for _ in range(3))
        self.cost_conv = same_conv(2, 1, 3, 1)
        self.occ_linear = nn.Linear(6, 2)

    def frame_features(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unit-normalised (hires, lowres, highest) NHWC features of frames `[N, 3, 256, 256]` in `[-1, 1]`."""
        highest, hires, lowres = self.resnet_torch(frames)
        return tuple(_unit(f.permute(0, 2, 3, 1)) for f in (hires, lowres, highest))

    def features(self, grids: tuple[torch.Tensor, ...], queries: torch.Tensor) -> Features:
        """Frame features `[T, h, w, C]` (hires, lowres, highest) and the 7 x 7 query supports at `queries` `[1, Q, 3]`
        `(t, y, x)` (256 raster)."""
        grids = tuple(g[None] for g in grids)
        ctx = F.pad(_offsets(queries.device), (1, 0)).to(queries.dtype)  # (0, dy, dx)
        supports = []
        for grid in grids:
            scale = torch.tensor([1.0, grid.shape[2] / SIZE, grid.shape[3] / SIZE], device=queries.device)
            pos = (queries * scale)[:, :, None] + ctx  # [1, Q, 49, 3]
            supp = map_coordinates_3d(grid, pos.flatten(1, 2))
            supports.append(supp.view(1, -1, SUPPORT, SUPPORT, grid.shape[-1]))
        return Features(grids, tuple(supports))

    def initial(self, f: Features, query: tuple[torch.Tensor, torch.Tensor], queries: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Points `[B, N, T, 2]` `(x, y)` (256 raster), occlusion and expected-distance logits `[B, N, T]`."""
        hires_q, lowres_q = query
        cost = torch.einsum("bnc,bthwc->tbnhw", lowres_q, f.grids[1])
        cost_hires = torch.einsum("bnc,bthwc->tbnhw", hires_q, f.grids[0])
        t, b, n, h, w = cost_hires.shape
        interp = F.interpolate(cost.reshape(t * b * n, 1, *cost.shape[3:]), (h, w), mode="bilinear", align_corners=False)
        stack = torch.cat([interp, cost_hires.reshape(t * b * n, 1, h, w)], 1)  # [tbn, 2, h, w]
        pos = self.cost_conv(stack).reshape(t, b, n, h, w).permute(1, 2, 0, 3, 4)
        heat = F.softmax(pos.reshape(b, n, t, -1) * SOFTMAX_TEMPERATURE, -1).view_as(pos)
        points = soft_argmax(heat) * torch.tensor([SIZE / w, SIZE / h], device=heat.device)
        is_query = torch.round(queries[..., :1]) == torch.arange(t, device=queries.device)  # [B, N, T]
        points = torch.where(is_query[..., None], queries[:, :, None, 1:].flip(-1), points)
        occ = self.occ_linear(torch.cat([stack.mean((2, 3)), stack.amax((2, 3)), stack.amin((2, 3))], -1))
        occ = occ.reshape(t, b, n, 2).permute(1, 2, 0, 3)
        return points, occ[..., 0], occ[..., 1]

    def refine(
        self, supports: list[torch.Tensor], points: torch.Tensor, occ: torch.Tensor, expd: torch.Tensor, f: Features
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One refinement: 4D correlations around `points`, the mixer's update of points and logits."""
        ctx = _offsets(points.device)
        corrs = []
        for supp, grid, cmdtop in zip(supports, f.grids, self.cmdtop, strict=True):
            yx = (points * torch.tensor([grid.shape[3] / SIZE, grid.shape[2] / SIZE], device=points.device)).flip(-1)
            neighborhood = map_coordinates_2d(grid, yx[:, :, :, None] + ctx)  # [b, n, t, 49, c]
            b, n, t = neighborhood.shape[:3]
            corr = torch.einsum("bntsc,bnjc->bntsj", neighborhood, supp.flatten(2, 3).to(neighborhood.dtype))
            corrs.append(cmdtop(corr.reshape(b * n * t, SUPPORT, SUPPORT, SUPPORT, SUPPORT)).view(b, n, t, -1))
        rel_fwd = F.pad(points[..., :-1, :] - points[..., 1:, :], (0, 0, 0, 1))
        rel_bwd = F.pad(points[..., 1:, :] - points[..., :-1, :], (0, 0, 1, 0))
        rel = torch.cat([rel_fwd, rel_bwd], -1) / SIZE
        x = torch.cat([occ[..., None], expd[..., None], *corrs, posenc(rel)], -1)
        b, n, t, c = x.shape
        res = self.torch_pips_mixer(x.reshape(b * n, t, c)).reshape(b, n, t, -1).float()
        return points + res[..., :2], occ + res[..., 2], expd + res[..., 3]

    def track(self, f: Features, queries: torch.Tensor, chunk: slice) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Final points `[1, Q, T, 2]` `(x, y)`, occlusion and expected-distance logits `[1, Q, T]` of a query chunk."""
        supports = [s[:, chunk] for s in f.supports]
        centre = [s[:, :, SUPPORT // 2, SUPPORT // 2] for s in supports]
        points, occ, expd = self.initial(f, (centre[0], centre[1]), queries[:, chunk])
        for _ in range(PIPS_ITERS):
            points, occ, expd = self.refine(supports, points, occ, expd, f)
        return points, occ, expd
