"""SEA-RAFT network (Wang, Lipson & Deng, ECCV 2024) as princeton-vl/SEA-RAFT builds it (`core/raft.py`).

Inference only.

Ported from princeton-vl/SEA-RAFT (BSD-3-Clause, see `LICENSE`) with the same module tree, so the released
checkpoints load as they are. Changes: no torchvision / huggingface_hub import (the ImageNet initialisation and the
hub mixin are training-time only); the input is already scaled to [-1, 1] and padded (the core does both, as the
card's `[io]` says); the flow is upsampled (and the upsampling weights predicted) after the last iteration only,
which is upstream's `final`; the uncertainty (`info`) channels of the flow head are not returned; with `fp16=True`
the encoders and the update block run under fp16 autocast while the correlation and its lookup stay float32.

- Context encoder (ResNet18 / ResNet34 stages, batch norm) on both images stacked as 6 channels, then a 3 x 3 conv:
  the initial hidden state and the context features (no activation). The flow head on the hidden state gives the
  initial flow (direct regression, iteration 0).
- Feature encoder (same architecture, 3 channels) on each image: stride 8. All-pairs correlation (dot product /
  sqrt(C)) against fmap2 bilinearly downsampled by 2 per level (4 levels); each iteration samples a (2r + 1)^2 window
  around the current match on every level (bilinear, `grid_sample`).
- Update block: motion encoder (correlation + flow convs), two ConvNeXt blocks on [hidden | context | motion]; the
  flow head adds a delta flow at 1/8 resolution.
- Upsampling: convex combination of 3 x 3 neighbours with predicted weights.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

FACTOR = 8  # fixed: the encoders downsample by 8 (three stride-2 stages)


@dataclass(frozen=True)
class Config:
    """Architecture of one SEA-RAFT size (the fields of upstream's `config/eval/*.json` that shape the network)."""

    blocks: tuple[int, int, int]  # BasicBlocks per stage: resnet18 (2, 2, 2), resnet34 (3, 4, 6)
    block_dims: tuple[int, int, int]
    initial_dim: int
    dim: int  # hidden state and context channels
    radius: int  # correlation lookup radius
    levels: int  # correlation pyramid levels
    num_blocks: int  # ConvNeXt blocks in the update block


SMALL = Config(
    blocks=(2, 2, 2), block_dims=(64, 128, 256), initial_dim=64, dim=128, radius=4, levels=4, num_blocks=2
)  # fixed: upstream config/eval/*-S.json (pretrain resnet18)
MEDIUM = Config(
    blocks=(3, 4, 6), block_dims=(64, 128, 256), initial_dim=64, dim=128, radius=4, levels=4, num_blocks=2
)  # fixed: upstream config/eval/*-M.json and *-L.json (pretrain resnet34; L = M weights with 12 iterations)


def _conv1x1(cin: int, cout: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, kernel_size=1, stride=stride, padding=0)


def _conv3x3(cin: int, cout: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1)


class BasicBlock(nn.Module):
    """Two 3 x 3 conv-bn-relu with a residual (1 x 1 conv-bn projection when the shape changes) and a final relu."""

    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        """Build the block."""
        super().__init__()
        self.conv1 = _conv3x3(cin, cout, stride)
        self.conv2 = _conv3x3(cout, cout)
        self.bn1 = nn.BatchNorm2d(cout)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.downsample: nn.Module | None = None
        if stride != 1 or cin != cout:
            # upstream: downsample = Sequential(conv1x1, bn3) with bn3 shared; the released safetensors keep the shared
            # tensors once (as `bn3.*`), so the projection conv sits alone at `downsample.0` and bn3 follows it
            self.bn3 = nn.BatchNorm2d(cout)
            self.downsample = nn.Sequential(_conv1x1(cin, cout, stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual block."""
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        if self.downsample is not None:
            x = self.bn3(self.downsample(x))
        return self.relu(x + y)


class ResNetFPN(nn.Module):
    """7 x 7 stride-2 stem (conv-bn-relu), three ResNet stages (strides 1, 2, 2), 1 x 1 output conv: stride 8."""

    def __init__(self, cfg: Config, cin: int, cout: int) -> None:
        """Build the encoder."""
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cfg.initial_dim, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(cfg.initial_dim)
        self.relu = nn.ReLU(inplace=True)
        dims = (cfg.initial_dim, *cfg.block_dims)
        self.layer1, self.layer2, self.layer3 = (
            nn.Sequential(*(BasicBlock(dims[i] if j == 0 else dims[i + 1], dims[i + 1], s if j == 0 else 1)
                            for j in range(n)))
            for i, (n, s) in enumerate(zip(cfg.blocks, (1, 2, 2), strict=True))
        )  # fmt: skip
        self.final_conv = _conv1x1(cfg.block_dims[2], cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, cin, H, W]` -> `[B, cout, H / 8, W / 8]`."""
        x = self.relu(self.bn1(self.conv1(x)))
        return self.final_conv(self.layer3(self.layer2(self.layer1(x))))


class LayerNorm(nn.Module):
    """LayerNorm over the last (channel) dimension, with upstream's parameter names."""

    def __init__(self, dim: int, eps: float) -> None:
        """Build the norm."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised `[..., C]`."""
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


class ConvNextBlock(nn.Module):
    """7 x 7 depthwise conv, LayerNorm, Linear-GELU-Linear MLP, layer scale, residual, then a 1 x 1 conv to `cout`."""

    def __init__(self, dim: int, cout: int) -> None:
        """Build the block."""
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * cout)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * cout, dim)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.final = nn.Conv2d(dim, cout, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, dim, h, w]` -> `[B, cout, h, w]`."""
        y = self.dwconv(x).permute(0, 2, 3, 1)
        y = self.gamma * self.pwconv2(self.act(self.pwconv1(self.norm(y))))
        return self.final(x + y.permute(0, 3, 1, 2))


class BasicMotionEncoder(nn.Module):
    """Encodes the correlation features and the current flow; the flow is appended to the output."""

    def __init__(self, corr_channels: int, dim: int) -> None:
        """Build the encoder."""
        super().__init__()
        self.convc1 = nn.Conv2d(corr_channels, dim * 2, 1, padding=0)
        self.convc2 = nn.Conv2d(dim * 2, dim + dim // 2, 3, padding=1)
        self.convf1 = nn.Conv2d(2, dim, 7, padding=3)
        self.convf2 = nn.Conv2d(dim, dim // 2, 3, padding=1)
        self.conv = nn.Conv2d(dim * 2, dim - 2, 3, padding=1)

    def forward(self, flow: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """Motion features `[B, dim, h, w]`."""
        cor = F.relu(self.convc2(F.relu(self.convc1(corr))))
        flo = F.relu(self.convf2(F.relu(self.convf1(flow))))
        return torch.cat([F.relu(self.conv(torch.cat([cor, flo], dim=1))), flow], dim=1)


class BasicUpdateBlock(nn.Module):
    """Motion encoder, then ConvNeXt blocks on [hidden | context | motion] -> new hidden state."""

    def __init__(self, cfg: Config, corr_channels: int) -> None:
        """Build the block."""
        super().__init__()
        self.encoder = BasicMotionEncoder(corr_channels, cfg.dim)
        self.refine = nn.ModuleList(ConvNextBlock(3 * cfg.dim, cfg.dim) for _ in range(cfg.num_blocks))

    def forward(self, net: torch.Tensor, inp: torch.Tensor, corr: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Updated hidden state."""
        inp = torch.cat([inp, self.encoder(flow, corr)], dim=1)
        for blk in self.refine:
            net = blk(torch.cat([net, inp], dim=1))
        return net


def _coords(b: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    """Pixel coordinates `[B, 2, h, w]` (x, y), float32."""
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([xs, ys])[None].expand(b, -1, -1, -1)


def _pyramid(fmap1: torch.Tensor, fmap2: torch.Tensor, levels: int) -> list[torch.Tensor]:
    """Correlation of fmap1 with fmap2 halved `l` times (bilinear), `[B * h * w, 1, h2, w2]` per level."""
    b, c, h, w = fmap1.shape
    f1 = fmap1.reshape(b, c, h * w).transpose(1, 2)
    out = []
    for _ in range(levels):
        h2, w2 = fmap2.shape[-2:]
        corr = torch.matmul(f1, fmap2.reshape(b, c, h2 * w2)) / math.sqrt(c)
        out.append(corr.reshape(b * h * w, 1, h2, w2))
        fmap2 = F.interpolate(fmap2, scale_factor=0.5, mode="bilinear", align_corners=False)
    return out


def _lookup(pyramid: list[torch.Tensor], coords: torch.Tensor, radius: int) -> torch.Tensor:
    """Correlation windows around `coords` `[B, 2, h, w]` on every level -> `[B, levels * (2r + 1)^2, h, w]`."""
    side = 2 * radius + 1
    d = torch.linspace(-radius, radius, side, device=coords.device)
    dy, dx = torch.meshgrid(d, d, indexing="ij")
    delta = torch.stack([dy, dx], dim=-1).view(1, side, side, 2)  # as upstream: (dy, dx) offsets (x, y)
    b, _, h, w = coords.shape
    centroids = coords.permute(0, 2, 3, 1).reshape(b * h * w, 1, 1, 2)
    out = []
    for i, corr in enumerate(pyramid):
        ch, cw = corr.shape[-2:]
        gx, gy = (centroids / 2**i + delta).split([1, 1], dim=-1)
        grid = torch.cat([2 * gx / (cw - 1) - 1, 2 * gy / (ch - 1) - 1], dim=-1)
        out.append(F.grid_sample(corr, grid, align_corners=True).view(b, h, w, -1))
    return torch.cat(out, dim=-1).permute(0, 3, 1, 2).contiguous()


def _upsample(flow: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Flow at 1/8 -> full resolution: convex combination of 3 x 3 neighbours."""
    b, c, h, w = flow.shape
    m = torch.softmax(mask.view(b, 1, 9, FACTOR, FACTOR, h, w), dim=2)
    up = F.unfold(FACTOR * flow, kernel_size=3, padding=1).view(b, c, 9, 1, 1, h, w)
    up = torch.sum(m * up, dim=2)
    return up.permute(0, 1, 4, 2, 5, 3).reshape(b, c, h * FACTOR, w * FACTOR)


class SEARAFT(nn.Module):
    """SEA-RAFT: `(image1, image2)` in `[-1, 1]`, H and W multiples of 8 -> flow `[B, 2, H, W]` (image1 to image2)."""

    def __init__(self, cfg: Config) -> None:
        """Build the network for one size (`SMALL`, `MEDIUM`)."""
        super().__init__()
        self.cfg = cfg
        dim = cfg.dim
        self.cnet = ResNetFPN(cfg, 6, 2 * dim)
        self.init_conv = _conv3x3(2 * dim, 2 * dim)
        self.upsample_weight = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(dim * 2, 64 * 9, 1, padding=0)
        )
        self.flow_head = nn.Sequential(
            nn.Conv2d(dim, 2 * dim, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(2 * dim, 6, 3, padding=1)
        )
        self.fnet = ResNetFPN(cfg, 3, 2 * dim)
        self.update_block = BasicUpdateBlock(cfg, cfg.levels * (2 * cfg.radius + 1) ** 2)
        self.fp16 = False  # fp16 autocast for the encoders and the update block (CUDA)

    def _amp(self, device: torch.device) -> contextlib.AbstractContextManager:
        return torch.autocast(device.type, torch.float16) if self.fp16 else contextlib.nullcontext()

    def forward(self, image1: torch.Tensor, image2: torch.Tensor, iters: int) -> torch.Tensor:
        """Flow after `iters` updates, float32 `[B, 2, H, W]`."""
        cfg = self.cfg
        b, _, h, w = image1.shape
        with self._amp(image1.device):
            net, context = self.init_conv(self.cnet(torch.cat([image1, image2], dim=1))).split(cfg.dim, dim=1)
            flow = self.flow_head(net)[:, :2].float()
            fmaps = self.fnet(torch.cat([image1, image2], dim=0))  # batch norm in eval mode: per-image as upstream
        fmap1, fmap2 = fmaps.float().chunk(2, dim=0)
        pyramid = _pyramid(fmap1, fmap2, cfg.levels)
        grid = _coords(b, h // FACTOR, w // FACTOR, image1.device)
        for _ in range(iters):
            corr = _lookup(pyramid, grid + flow, cfg.radius)
            with self._amp(image1.device):
                net = self.update_block(net, context, corr, flow)
                delta = self.flow_head(net)[:, :2]
            flow = flow + delta.float()
        with self._amp(image1.device):
            mask = 0.25 * self.upsample_weight(net)
        return _upsample(flow, mask.float())
