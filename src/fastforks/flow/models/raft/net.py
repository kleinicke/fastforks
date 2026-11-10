"""RAFT network (Teed & Deng 2020) as torchvision 0.29.1 builds it (`torchvision/models/optical_flow/raft.py`).

Inference only.

Ported from torchvision (BSD-3-Clause, see `LICENSE`) with the same module tree, so torchvision's checkpoints load
as they are. Changes: no torchvision import (`Conv2dNormActivation` is a plain `nn.Sequential` with the same
indices); the flow is upsampled (and the mask predicted) after the last iteration only, which is the last entry of
torchvision's list of predictions; the coordinate grids are built on the input's device; with `fp16=True` the
encoders and the update block run under fp16 autocast while the correlation volume and its lookup stay float32 (as
upstream RAFT's `mixed_precision`).

- Feature encoder (instance norm) on both images, context encoder (batch norm; none for raft-small) on the first:
  stride 8. The context splits into the initial GRU state (tanh) and the context features (relu).
- All-pairs correlation of the two feature maps (dot product / sqrt(C)), average-pooled into a 4-level pyramid; each
  iteration samples a (2r + 1)^2 window around the current match on every level (bilinear, `grid_sample`).
- Update block: motion encoder (correlation + flow convs), ConvGRU(s), flow head -> delta flow at 1/8 resolution.
- Upsampling: convex combination of 3 x 3 neighbours with a predicted mask (raft-large), or bilinear (raft-small).
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
    """Architecture of one RAFT size (the arguments of torchvision's `raft_large` / `raft_small`)."""

    feature_layers: tuple[int, int, int, int, int]
    context_layers: tuple[int, int, int, int, int]
    bottleneck: bool  # BottleneckBlock (small) instead of ResidualBlock (large)
    context_norm: bool  # batch norm in the context encoder (large); none (small)
    corr_levels: int
    corr_radius: int
    motion_corr_layers: tuple[int, ...]
    motion_flow_layers: tuple[int, int]
    motion_out: int
    hidden: int
    gru_kernels: tuple[tuple[int, int], ...]  # one ConvGRU per kernel; padding keeps the size
    flow_head_hidden: int
    mask: bool  # convex upsampling with a mask predictor (large); bilinear (small)


LARGE = Config(
    feature_layers=(64, 64, 96, 128, 256),
    context_layers=(64, 64, 96, 128, 256),
    bottleneck=False,
    context_norm=True,
    corr_levels=4,
    corr_radius=4,
    motion_corr_layers=(256, 192),
    motion_flow_layers=(128, 64),
    motion_out=128,
    hidden=128,
    gru_kernels=((1, 5), (5, 1)),
    flow_head_hidden=256,
    mask=True,
)  # fixed: torchvision raft_large

SMALL = Config(
    feature_layers=(32, 32, 64, 96, 128),
    context_layers=(32, 32, 64, 96, 160),
    bottleneck=True,
    context_norm=False,
    corr_levels=4,
    corr_radius=3,
    motion_corr_layers=(96,),
    motion_flow_layers=(64, 32),
    motion_out=82,
    hidden=96,
    gru_kernels=((3, 3),),
    flow_head_hidden=128,
    mask=False,
)  # fixed: torchvision raft_small


def _cna(  # noqa: PLR0913 - the arguments of torchvision's `Conv2dNormActivation` that RAFT sets
    cin: int, cout: int, k: int, *, stride: int = 1, norm: str | None = None, act: bool = True
) -> nn.Sequential:
    """As torchvision's `Conv2dNormActivation` (bias always on): conv [0], norm [1] if any, then ReLU."""
    layers: list[nn.Module] = [nn.Conv2d(cin, cout, k, stride, (k - 1) // 2, bias=True)]
    if norm == "instance":
        layers.append(nn.InstanceNorm2d(cout))
    elif norm == "batch":
        layers.append(nn.BatchNorm2d(cout))
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    """Two 3 x 3 conv-norm-relu with a residual (1 x 1 projection when strided) and a final relu."""

    def __init__(self, cin: int, cout: int, norm: str | None, stride: int) -> None:
        """Build the block."""
        super().__init__()
        self.convnormrelu1 = _cna(cin, cout, 3, stride=stride, norm=norm)
        self.convnormrelu2 = _cna(cout, cout, 3, norm=norm)
        self.downsample = nn.Identity() if stride == 1 else _cna(cin, cout, 1, stride=stride, norm=norm, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual block."""
        return self.relu(self.downsample(x) + self.convnormrelu2(self.convnormrelu1(x)))


class BottleneckBlock(nn.Module):
    """1 x 1 -> 3 x 3 (strided) -> 1 x 1 conv-norm-relu at a quarter of the channels, residual, final relu."""

    def __init__(self, cin: int, cout: int, norm: str | None, stride: int) -> None:
        """Build the block."""
        super().__init__()
        self.convnormrelu1 = _cna(cin, cout // 4, 1, norm=norm)
        self.convnormrelu2 = _cna(cout // 4, cout // 4, 3, stride=stride, norm=norm)
        self.convnormrelu3 = _cna(cout // 4, cout, 1, norm=norm)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Identity() if stride == 1 else _cna(cin, cout, 1, stride=stride, norm=norm, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Bottleneck block."""
        y = self.convnormrelu3(self.convnormrelu2(self.convnormrelu1(x)))
        return self.relu(self.downsample(x) + y)


class FeatureEncoder(nn.Module):
    """7 x 7 stride-2 stem, three stages of two blocks (strides 1, 2, 2), 1 x 1 output conv: stride 8."""

    def __init__(self, layers: tuple[int, ...], norm: str | None, *, bottleneck: bool) -> None:
        """Build the encoder."""
        super().__init__()
        block = BottleneckBlock if bottleneck else ResidualBlock
        self.convnormrelu = _cna(3, layers[0], 7, stride=2, norm=norm)
        self.layer1 = nn.Sequential(block(layers[0], layers[1], norm, 1), block(layers[1], layers[1], norm, 1))
        self.layer2 = nn.Sequential(block(layers[1], layers[2], norm, 2), block(layers[2], layers[2], norm, 1))
        self.layer3 = nn.Sequential(block(layers[2], layers[3], norm, 2), block(layers[3], layers[3], norm, 1))
        self.conv = nn.Conv2d(layers[3], layers[4], kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, 3, H, W]` -> `[B, C, H / 8, W / 8]`."""
        return self.conv(self.layer3(self.layer2(self.layer1(self.convnormrelu(x)))))


class MotionEncoder(nn.Module):
    """Encodes the correlation features and the current flow; the flow is appended to the output."""

    def __init__(self, cfg: Config, corr_channels: int) -> None:
        """Build the encoder."""
        super().__init__()
        corr, fl = cfg.motion_corr_layers, cfg.motion_flow_layers
        self.convcorr1 = _cna(corr_channels, corr[0], 1)
        self.convcorr2 = _cna(corr[0], corr[1], 3) if len(corr) == 2 else nn.Identity()  # noqa: PLR2004 - large: 2
        self.convflow1 = _cna(2, fl[0], 7)
        self.convflow2 = _cna(fl[0], fl[1], 3)
        self.conv = _cna(corr[-1] + fl[-1], cfg.motion_out - 2, 3)

    def forward(self, flow: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """Motion features `[B, motion_out, h, w]`."""
        c = self.convcorr2(self.convcorr1(corr))
        f = self.convflow2(self.convflow1(flow))
        return torch.cat([self.conv(torch.cat([c, f], dim=1)), flow], dim=1)


class ConvGRU(nn.Module):
    """Convolutional GRU cell."""

    def __init__(self, cin: int, hidden: int, kernel: tuple[int, int]) -> None:
        """Build the cell; the padding keeps the size."""
        super().__init__()
        pad = (kernel[0] // 2, kernel[1] // 2)
        self.convz = nn.Conv2d(hidden + cin, hidden, kernel, padding=pad)
        self.convr = nn.Conv2d(hidden + cin, hidden, kernel, padding=pad)
        self.convq = nn.Conv2d(hidden + cin, hidden, kernel, padding=pad)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Updated hidden state."""
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


class RecurrentBlock(nn.Module):
    """One or two ConvGRUs applied in turn (large: 1 x 5 then 5 x 1)."""

    def __init__(self, cin: int, hidden: int, kernels: tuple[tuple[int, int], ...]) -> None:
        """Build the GRUs."""
        super().__init__()
        self.convgru1 = ConvGRU(cin, hidden, kernels[0])
        if len(kernels) > 1:
            self.convgru2 = ConvGRU(cin, hidden, kernels[1])

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Updated hidden state."""
        h = self.convgru1(h, x)
        return self.convgru2(h, x) if hasattr(self, "convgru2") else h


class FlowHead(nn.Module):
    """Two 3 x 3 convs: hidden state -> delta flow."""

    def __init__(self, cin: int, hidden: int) -> None:
        """Build the head."""
        super().__init__()
        self.conv1 = nn.Conv2d(cin, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Delta flow `[B, 2, h, w]`."""
        return self.conv2(self.relu(self.conv1(x)))


class UpdateBlock(nn.Module):
    """Motion encoder, recurrent block and flow head."""

    def __init__(self, cfg: Config, corr_channels: int) -> None:
        """Build the block."""
        super().__init__()
        context = cfg.context_layers[-1] - cfg.hidden
        self.motion_encoder = MotionEncoder(cfg, corr_channels)
        self.recurrent_block = RecurrentBlock(cfg.motion_out + context, cfg.hidden, cfg.gru_kernels)
        self.flow_head = FlowHead(cfg.hidden, cfg.flow_head_hidden)

    def forward(
        self, h: torch.Tensor, context: torch.Tensor, corr: torch.Tensor, flow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Updated hidden state and delta flow."""
        x = torch.cat([context, self.motion_encoder(flow, corr)], dim=1)
        h = self.recurrent_block(h, x)
        return h, self.flow_head(h)


class MaskPredictor(nn.Module):
    """Upsampling mask: 3 x 3 conv-relu, 1 x 1 conv to 8 * 8 * 9 weights, scaled by 0.25."""

    def __init__(self, cin: int, hidden: int) -> None:
        """Build the predictor."""
        super().__init__()
        self.convrelu = _cna(cin, hidden, 3)
        self.conv = nn.Conv2d(hidden, FACTOR * FACTOR * 9, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mask logits `[B, 576, h, w]`."""
        return 0.25 * self.conv(self.convrelu(x))


def _coords(b: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    """Pixel coordinates `[B, 2, h, w]` (x, y), float32."""
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([xs, ys])[None].expand(b, -1, -1, -1)


def _pyramid(fmap1: torch.Tensor, fmap2: torch.Tensor, levels: int) -> list[torch.Tensor]:
    """All-pairs correlation `[B * h * w, 1, h, w]`, average-pooled `levels - 1` times."""
    b, c, h, w = fmap1.shape
    corr = torch.matmul(fmap1.reshape(b, c, h * w).transpose(1, 2), fmap2.reshape(b, c, h * w))
    corr = corr.reshape(b * h * w, 1, h, w) / math.sqrt(c)
    out = [corr]
    for _ in range(levels - 1):
        corr = F.avg_pool2d(corr, kernel_size=2, stride=2)
        out.append(corr)
    return out


def _lookup(pyramid: list[torch.Tensor], coords: torch.Tensor, radius: int) -> torch.Tensor:
    """Correlation windows around `coords` `[B, 2, h, w]` on every level -> `[B, levels * (2r + 1)^2, h, w]`."""
    side = 2 * radius + 1
    d = torch.linspace(-radius, radius, side, device=coords.device)
    di, dj = torch.meshgrid(d, d, indexing="ij")
    delta = torch.stack([di, dj], dim=-1).view(1, side, side, 2)  # as torchvision: (di, dj) offsets (x, y)
    b, _, h, w = coords.shape
    centroids = coords.permute(0, 2, 3, 1).reshape(b * h * w, 1, 1, 2)
    out = []
    for corr in pyramid:
        ch, cw = corr.shape[-2:]
        grid = centroids + delta
        gx, gy = grid.split([1, 1], dim=-1)
        gx = 2 * gx / (cw - 1) - 1
        if ch > 1:
            gy = 2 * gy / (ch - 1) - 1
        sampled = F.grid_sample(corr, torch.cat([gx, gy], dim=-1), mode="bilinear", align_corners=True)
        out.append(sampled.view(b, h, w, -1))
        centroids = centroids / 2
    return torch.cat(out, dim=-1).permute(0, 3, 1, 2).contiguous()


def _upsample(flow: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Flow at 1/8 -> full resolution: convex combination of 3 x 3 neighbours, or bilinear without a mask."""
    b, c, h, w = flow.shape
    if mask is None:
        return FACTOR * F.interpolate(flow, size=(h * FACTOR, w * FACTOR), mode="bilinear", align_corners=True)
    m = torch.softmax(mask.view(b, 1, 9, FACTOR, FACTOR, h, w), dim=2)
    up = F.unfold(FACTOR * flow, kernel_size=3, padding=1).view(b, c, 9, 1, 1, h, w)
    up = torch.sum(m * up, dim=2)
    return up.permute(0, 1, 4, 2, 5, 3).reshape(b, c, h * FACTOR, w * FACTOR)


class RAFT(nn.Module):
    """RAFT: `(image1, image2)` in `[-1, 1]`, H and W multiples of 8 -> flow `[B, 2, H, W]` from image1 to image2."""

    def __init__(self, cfg: Config) -> None:
        """Build the network for one size (`LARGE`, `SMALL`)."""
        super().__init__()
        self.cfg = cfg
        corr_channels = cfg.corr_levels * (2 * cfg.corr_radius + 1) ** 2
        self.feature_encoder = FeatureEncoder(cfg.feature_layers, "instance", bottleneck=cfg.bottleneck)
        self.context_encoder = FeatureEncoder(
            cfg.context_layers, "batch" if cfg.context_norm else None, bottleneck=cfg.bottleneck
        )
        self.update_block = UpdateBlock(cfg, corr_channels)
        if cfg.mask:
            self.mask_predictor = MaskPredictor(cfg.hidden, 256)
        self.fp16 = False  # fp16 autocast for the encoders and the update block (CUDA)

    def _amp(self, device: torch.device) -> contextlib.AbstractContextManager:
        return torch.autocast(device.type, torch.float16) if self.fp16 else contextlib.nullcontext()

    def forward(self, image1: torch.Tensor, image2: torch.Tensor, iters: int) -> torch.Tensor:
        """Flow after `iters` updates, float32 `[B, 2, H, W]`."""
        cfg = self.cfg
        b, _, h, w = image1.shape
        hs, ws = h // FACTOR, w // FACTOR
        with self._amp(image1.device):
            fmaps = self.feature_encoder(torch.cat([image1, image2], dim=0))
            ctx = self.context_encoder(image1)
        fmap1, fmap2 = fmaps.float().chunk(2, dim=0)
        pyramid = _pyramid(fmap1, fmap2, cfg.corr_levels)
        hidden, context = torch.split(ctx, [cfg.hidden, ctx.shape[1] - cfg.hidden], dim=1)
        hidden, context = torch.tanh(hidden), F.relu(context)
        coords0 = _coords(b, hs, ws, image1.device)
        coords1 = coords0
        for _ in range(iters):
            corr = _lookup(pyramid, coords1, cfg.corr_radius)
            with self._amp(image1.device):
                hidden, delta = self.update_block(hidden, context, corr, coords1 - coords0)
            coords1 = coords1 + delta.float()
        mask = None
        if cfg.mask:
            with self._amp(image1.device):
                mask = self.mask_predictor(hidden)
            mask = mask.float()
        return _upsample(coords1 - coords0, mask)
