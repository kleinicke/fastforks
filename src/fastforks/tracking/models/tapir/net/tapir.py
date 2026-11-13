# Copyright 2026 Google LLC
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
# fastforks: upstream `tapnet/torch/tapir_model.py`, `nets.py` and `utils.py` for one video at the training resolution
# (256 x 256: one feature grid, used for the initialisation and the refinement), offline (the causal convolution and
# its state removed), einshape / dm-tree removed, queries in order (upstream permutes them randomly into chunks; each
# query is computed on its own, so the order does not change the result). Parameter names as upstream (its state dicts
# load unchanged).
"""TAPIR / BootsTAPIR (offline point tracker) for one 256 x 256 video `[1, T, 256, 256, 3]` in `[-1, 1]`."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

SIZE = 256  # training resolution; the network runs at it
PIPS_ITERS = 4
SOFTMAX_TEMPERATURE = 20.0
HIRES, LOWRES = 128, 256


class ExtraConvBlock(nn.Module):
    """LayerNorm, 3 x 3 conv up (x4), GELU-tanh, 3 x 3 conv down, residual; NHWC in and out."""

    def __init__(self, dim: int, mult: int) -> None:  # noqa: D107
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim * mult, 3, padding=1)
        self.conv_1 = nn.Conv2d(dim * mult, dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = self.layer_norm(x).permute(0, 3, 1, 2)
        x = x + self.conv_1(F.gelu(self.conv(x), approximate="tanh"))
        return x.permute(0, 2, 3, 1)


class ExtraConvs(nn.Module):
    """BootsTAPIR's 5 extra conv blocks on the low-resolution features."""

    def __init__(self, num_layers: int = 5, dim: int = 256, mult: int = 4) -> None:  # noqa: D107
        super().__init__()
        self.blocks = nn.ModuleList(ExtraConvBlock(dim, mult) for _ in range(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        for block in self.blocks:
            x = block(x)
        return x


class ConvChannelsMixer(nn.Module):
    """The mixer block's channel MLP (x4, GELU-tanh)."""

    def __init__(self, dim: int) -> None:  # noqa: D107
        super().__init__()
        self.mlp2_up = nn.Linear(dim, dim * 4)
        self.mlp2_down = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.mlp2_down(F.gelu(self.mlp2_up(x), approximate="tanh"))


class PIPsConvBlock(nn.Module):
    """Mixer block: two depthwise temporal convs (channel multiplier 4, summed back), then the channel MLP."""

    def __init__(self, dim: int, kernel: int = 3) -> None:  # noqa: D107
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, bias=False)
        self.mlp1_up = nn.Conv1d(dim, dim * 4, kernel, padding=1, groups=dim)
        self.mlp1_up_1 = nn.Conv1d(dim * 4, dim * 4, kernel, padding=1, groups=dim * 4)
        self.layer_norm_1 = nn.LayerNorm(dim, bias=False)
        self.conv_channels_mixer = ConvChannelsMixer(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x` `[N, T, C]`."""
        y = self.layer_norm(x).permute(0, 2, 1)
        y = self.mlp1_up_1(F.gelu(self.mlp1_up(y), approximate="tanh")).permute(0, 2, 1)
        x = x + (y[..., 0::4] + y[..., 1::4] + y[..., 2::4] + y[..., 3::4])
        return x + self.conv_channels_mixer(self.layer_norm_1(x))


class PIPSMLPMixer(nn.Module):
    """Depthwise-conv MLP mixer over the frames of each track."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, num_blocks: int = 12) -> None:  # noqa: D107
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden)
        self.layer_norm = nn.LayerNorm(hidden, bias=False)
        self.linear_1 = nn.Linear(hidden, out_dim)
        self.blocks = nn.ModuleList(PIPsConvBlock(hidden) for _ in range(num_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x` `[N, T, in_dim]` -> `[N, T, out_dim]`."""
        x = self.linear(x)
        for block in self.blocks:
            x = block(x)
        return self.linear_1(self.layer_norm(x))


class BlockV2(nn.Module):
    """Pre-activation ResNet block with instance norms (JAX SAME padding for stride 2: right / bottom)."""

    def __init__(self, cin: int, cout: int, stride: int, use_projection: bool) -> None:  # noqa: D107, FBT001
        super().__init__()
        self.padding = (1, 1, 1, 1) if stride == 1 else (0, 2, 0, 2)
        self.proj_conv = nn.Conv2d(cin, cout, 1, stride, bias=False) if use_projection else None
        self.bn_0 = nn.InstanceNorm2d(cin, affine=True)
        self.conv_0 = nn.Conv2d(cin, cout, 3, stride, bias=False)
        self.conv_1 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn_1 = nn.InstanceNorm2d(cout, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        shortcut = x
        x = torch.relu(self.bn_0(x))
        if self.proj_conv is not None:
            shortcut = self.proj_conv(x)
        x = torch.relu(self.bn_1(self.conv_0(F.pad(x, self.padding))))
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
    """ResNet-18-like backbone; returns the outputs of units 1 (stride 4, 128 ch) and 3 (stride 8, 256 ch)."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.initial_conv = nn.Conv2d(3, 64, 7, 2, bias=False)
        channels, strides = (64, HIRES, 256, LOWRES), (1, 2, 2, 1)
        self.block_groups = nn.ModuleList(
            BlockGroup(channels[i - 1] if i else 64, channels[i], 2, strides[i]) for i in range(4)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`x` `[N, 3, H, W]` -> (unit 1, unit 3), NCHW."""
        x = self.initial_conv(F.pad(x, (2, 4, 2, 4)))
        outs = []
        for group in self.block_groups:
            x = group(x)
            outs.append(x)
        return outs[1], outs[3]


class Features(NamedTuple):
    """Unit-normalised features of every frame (NHWC) and of every query."""

    hires: torch.Tensor  # [1, T, 64, 64, 128]
    lowres: torch.Tensor  # [1, T, 32, 32, 256]
    query_hires: torch.Tensor  # [1, Q, 128]
    query_lowres: torch.Tensor  # [1, Q, 256]


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
    """Bilinear samples `[B, P, T, S, C]` of `feats` `[B, T, H, W, C]` at `coords` `[B, P, T, S, 2]` `(y, x)`.

    As upstream: normalised by the height alone (the grids are square), zeros outside.
    """
    n, t, h, w, c = feats.shape
    x = feats.permute(0, 1, 4, 2, 3).reshape(n * t, c, h, w)
    n, p, t, s, _ = coords.shape
    y = coords.permute(0, 2, 1, 3, 4).reshape(n * t, p, s, 2)
    y = (2 * (y / h) - 1).flip(-1).to(x.dtype)
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


class TAPIR(nn.Module):
    """TAPIR / BootsTAPIR at 256 x 256: per-frame initialisation from a cost volume, then 4 PIPs refinements."""

    def __init__(self, pyramid_level: int = 1, extra_convs: bool = True) -> None:  # noqa: D107, FBT001, FBT002
        super().__init__()
        self.pyramid_level = pyramid_level
        self.resnet_torch = ResNet()
        self.torch_cost_volume_track_mods = nn.ModuleDict({
            "hid1": nn.Conv2d(1, 16, 3, 1, 1),
            "hid2": nn.Conv2d(16, 1, 3, 1, 1),
            "hid3": nn.Conv2d(16, 32, 3, 2, 0),
            "hid4": nn.Linear(32, 16),
            "occ_out": nn.Linear(16, 2),
        })
        dim = 4 + HIRES + LOWRES
        self.torch_pips_mixer = PIPSMLPMixer(dim + (pyramid_level + 2) * 49, dim)
        self.extra_convs = ExtraConvs() if extra_convs else None

    def frame_features(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Unit-normalised (hires, lowres) NHWC features of frames `[N, 3, 256, 256]` in `[-1, 1]`."""
        hires, lowres = self.resnet_torch(frames)
        lowres = lowres.permute(0, 2, 3, 1)
        if self.extra_convs is not None:
            lowres = self.extra_convs(lowres)
        return _unit(hires.permute(0, 2, 3, 1)), _unit(lowres)

    def features(self, hires: torch.Tensor, lowres: torch.Tensor, queries: torch.Tensor) -> Features:
        """Frame features `[T, h, w, C]` and the query features at `queries` `[1, Q, 3]` `(t, y, x)` (256 raster)."""
        hires, lowres = hires[None], lowres[None]
        out = []
        for grid in (hires, lowres):
            scale = torch.tensor([1.0, grid.shape[2] / SIZE, grid.shape[3] / SIZE], device=queries.device)
            out.append(map_coordinates_3d(grid, queries * scale))
        return Features(hires, lowres, out[0], out[1])

    def initial(
        self, query: torch.Tensor, grid: torch.Tensor, queries: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Points `[B, N, T, 2]` `(x, y)` (256 raster), occlusion and expected-distance logits `[B, N, T]`."""
        mods = self.torch_cost_volume_track_mods
        cost = torch.einsum("bnc,bthwc->tbnhw", query, grid)
        t, b, n, h, w = cost.shape
        hidden = mods["hid1"](cost.reshape(t * b * n, 1, h, w)).relu_()
        del cost  # fastforks: each `[T * B * N, C, H, W]` map freed once used (upstream keeps all to the end)
        pos = mods["hid2"](hidden).reshape(t, b, n, h, w).permute(1, 2, 0, 3, 4).reshape(b, n, t, -1)
        heat = F.softmax(pos.mul_(SOFTMAX_TEMPERATURE), -1).view(b, n, t, h, w)
        del pos
        points = soft_argmax(heat) * torch.tensor([SIZE / w, SIZE / h], device=queries.device)
        del heat
        is_query = torch.round(queries[..., :1]) == torch.arange(t, device=queries.device)  # [B, N, T]
        points = torch.where(is_query[..., None], queries[:, :, None, 1:].flip(-1), points)
        hidden = F.pad(hidden, (0, 2, 0, 2))
        occ = mods["hid3"](hidden).relu_().mean((-1, -2))
        del hidden
        occ = mods["occ_out"](torch.relu(mods["hid4"](occ)))  # [T * B * N, 2]
        occ = occ.reshape(t, b, n, 2).permute(1, 2, 0, 3)
        return points, occ[..., 0], occ[..., 1]

    def refine(  # noqa: PLR0913, PLR0917 - upstream's refine_pips state
        self,
        queries: list[torch.Tensor],
        pyramid: list[torch.Tensor],
        points: torch.Tensor,
        occ: torch.Tensor,
        expd: torch.Tensor,
        last: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One PIPs step: local correlations around `points` and the mixer's update of points, logits and features."""
        ctx = torch.stack(
            torch.meshgrid(torch.arange(-3, 4), torch.arange(-3, 4), indexing="ij"), -1
        ).reshape(-1, 2).to(points.device)
        corrs = []
        for i, (query, grid) in enumerate(zip(queries, pyramid, strict=True)):
            yx = (points * torch.tensor([grid.shape[3] / SIZE, grid.shape[2] / SIZE], device=points.device)).flip(-1)
            neighborhood = map_coordinates_2d(grid, yx[:, :, :, None] + ctx)
            if last is None:
                corrs.append(torch.einsum("bnfsc,bnc->bnfs", neighborhood, query))
            else:
                q = last[..., :HIRES] if i == 0 else last[..., HIRES:]
                corrs.append(torch.einsum("bnfsc,bnfc->bnfs", neighborhood, q))
        if last is None:
            feats = torch.cat([queries[0], queries[1]], -1)[:, :, None].expand(-1, -1, points.shape[2], -1)
        else:
            feats = last
        x = torch.cat([torch.zeros_like(points), occ[..., None], expd[..., None], feats, *corrs], -1)
        b, n, t, c = x.shape
        res = self.torch_pips_mixer(x.reshape(b * n, t, c).float()).reshape(b, n, t, -1)
        return points + res[..., :2], occ + res[..., 2], expd + res[..., 3], feats + res[..., 4:]

    def track(self, f: Features, queries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Final points `[1, Q, T, 2]` `(x, y)`, occlusion and expected-distance logits `[1, Q, T]` of the queries."""
        points, occ, expd = self.initial(f.query_lowres, f.lowres, queries)
        queries_ = [f.query_hires, f.query_lowres] + [f.query_lowres] * self.pyramid_level
        pyramid = [f.hires, f.lowres]
        for _ in range(self.pyramid_level):
            pyramid.append(F.avg_pool3d(pyramid[-1], kernel_size=(2, 2, 1), stride=(2, 2, 1)))
        last = None
        for _ in range(PIPS_ITERS):
            points, occ, expd, last = self.refine(queries_, pyramid, points, occ, expd, last)
        return points, occ, expd
