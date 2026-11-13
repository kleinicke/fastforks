"""MemFlowNet (Dong & Fu, CVPR 2024) as upstream DQiaole/MemFlow runs a pair: the first step of its `InferenceCore`.

Inference only. Ported from DQiaole/MemFlow (Apache-2.0, see `LICENSE`): `core/Networks/MemFlowNet/{MemFlow,cnn,
corr,sk2,gma}.py` and `inference/{inference_core_skflow,memory_manager_skflow}.py`, with the same module tree, so
upstream's checkpoints load as they are (the unused relative position embedding `att.pos_emb.*` is dropped).

- Feature encoder (instance norm) on both frames, context encoder (batch norm) on the first: stride 8, 256 channels.
  The context splits into the initial hidden state (tanh) and the context features (relu); a 1 x 1 conv of the
  context gives the attention query and key.
- All-pairs correlation of the two feature maps (dot product / sqrt(C)), average-pooled into a 4-level pyramid; each
  iteration samples a 9 x 9 window around the current match on every level (bilinear, `align_corners=True`).
- Each iteration: a motion encoder (depthwise-separable "PCBlock"s) on the correlation and the flow, its value
  projection, the memory readout (softmax attention of the query over the keys of the memory frames, scaled by
  `dim^-0.5 * log(N) / log(train_avg_length)` for N keys, as upstream), the update block (PCBlock instead of a GRU) ->
  delta flow and the upsampling mask.
- The flow is upsampled once after the last iteration (convex combination of 3 x 3 neighbours), as upstream's
  inference.

A pair has an empty memory (upstream's first frame of a video, or its last with `end=True`): the keys and values are
the first frame's own. With `fp16=True` the encoders and the update block run under fp16 autocast, the correlation
volume, its lookup and the flow stay float32.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable

FACTOR = 8  # fixed: the encoders downsample by 8
DIM = 128  # fixed: hidden, context and attention width
LEVELS = 4  # fixed: correlation pyramid levels
RADIUS = 4  # fixed: lookup radius


class ResidualBlock(nn.Module):
    """Upstream `cnn.ResidualBlock`: two 3 x 3 conv-norm-relu, a residual (1 x 1 conv + norm when strided), relu."""

    def __init__(self, cin: int, cout: int, norm: str, stride: int) -> None:
        """Build the block (upstream module names)."""
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.norm1, self.norm2 = _norm(norm, cout), _norm(norm, cout)
        self.downsample: nn.Module | None = None
        if stride != 1:
            self.norm3 = _norm(norm, cout)
            self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride), self.norm3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual block."""
        y = self.relu(self.norm2(self.conv2(self.relu(self.norm1(self.conv1(x))))))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


def _norm(kind: str, c: int) -> nn.Module:
    return nn.BatchNorm2d(c) if kind == "batch" else nn.InstanceNorm2d(c)


class BasicEncoder(nn.Module):
    """Upstream `cnn.BasicEncoder`: 7 x 7 stride-2 stem, stages of 64 / 96 / 128 channels (strides 1, 2, 2), 1 x 1."""

    def __init__(self, norm: str, out: int = 256) -> None:
        """Build the encoder."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.norm1 = _norm(norm, 64)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(ResidualBlock(64, 64, norm, 1), ResidualBlock(64, 64, norm, 1))
        self.layer2 = nn.Sequential(ResidualBlock(64, 96, norm, 2), ResidualBlock(96, 96, norm, 1))
        self.layer3 = nn.Sequential(ResidualBlock(96, 128, norm, 2), ResidualBlock(128, 128, norm, 1))
        self.conv2 = nn.Conv2d(128, out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, 3, H, W]` -> `[B, 256, H / 8, W / 8]`."""
        x = self.relu1(self.norm1(self.conv1(x)))
        return self.conv2(self.layer3(self.layer2(self.layer1(x))))


class PCBlock(nn.Module):
    """Upstream `sk2.PCBlock4_Deep_nopool_res`: a residual 1.5x MLP, residual depthwise convs, a residual 1 x 1, MLP."""

    def __init__(self, cin: int, cout: int, kernels: tuple[int, ...]) -> None:
        """Build the block (depthwise kernels `k x k`)."""
        super().__init__()
        mid = int(1.5 * cin)
        self.conv_list = nn.ModuleList([nn.Conv2d(cin, cin, k, padding=k // 2, groups=cin) for k in kernels])
        self.ffn1 = nn.Sequential(nn.Conv2d(cin, mid, 1), nn.GELU(), nn.Conv2d(mid, cin, 1))
        self.pw = nn.Conv2d(cin, cin, 1)
        self.ffn2 = nn.Sequential(nn.Conv2d(cin, mid, 1), nn.GELU(), nn.Conv2d(mid, cout, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The block."""
        x = F.gelu(x + self.ffn1(x))
        for conv in self.conv_list:
            x = F.gelu(x + conv(x))
        return self.ffn2(F.gelu(x + self.pw(x)))


ENCODER_KERNELS = (1, 15)  # fixed: upstream k_conv
UPDATE_KERNELS = (1, 7)  # fixed: upstream PCUpdater_conv


class MotionEncoder(nn.Module):
    """Upstream `SKMotionEncoder6_Deep_nopool_res_Mem_skflow`; the flow is appended to the output."""

    def __init__(self) -> None:
        """Build the encoder."""
        super().__init__()
        planes = LEVELS * (2 * RADIUS + 1) ** 2
        self.convc1 = PCBlock(planes, 256, ENCODER_KERNELS)
        self.convc2 = PCBlock(256, 192, ENCODER_KERNELS)
        self.convf1 = nn.Conv2d(2, 128, 1)
        self.convf2 = PCBlock(128, 64, ENCODER_KERNELS)
        self.conv = PCBlock(64 + 192, DIM - 2, ENCODER_KERNELS)

    def forward(self, flow: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """Motion features `[B, 128, h, w]`."""
        cor = self.convc2(F.gelu(self.convc1(corr)))
        flo = self.convf2(self.convf1(flow))
        return torch.cat([self.conv(torch.cat([cor, flo], dim=1)), flow], dim=1)


class Aggregate(nn.Module):
    """The weights upstream's `gma.Aggregate` keeps in this network: the value projection and the readout gain."""

    def __init__(self) -> None:
        """Build the projection."""
        super().__init__()
        self.to_v = nn.Conv2d(DIM, DIM, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))


class UpdateBlock(nn.Module):
    """Upstream `SKUpdateBlock6_Deep_nopoolres_AllDecoder2_Mem_skflow`."""

    def __init__(self) -> None:
        """Build the block."""
        super().__init__()
        self.encoder = MotionEncoder()
        self.gru = PCBlock(4 * DIM, DIM, UPDATE_KERNELS)
        self.flow_head = PCBlock(DIM, 2, ENCODER_KERNELS)
        self.mask = nn.Sequential(nn.Conv2d(DIM, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 64 * 9, 1))
        self.aggregator = Aggregate()

    def forward(
        self, net: torch.Tensor, inp: torch.Tensor, motion: torch.Tensor, global_motion: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The new hidden state and the delta flow."""
        net = self.gru(torch.cat([net, inp, motion, global_motion], dim=1))
        return net, self.flow_head(net)


class Attention(nn.Module):
    """Upstream `gma.Attention` as MemFlow uses it: the query / key projection of the context (one head of 128)."""

    def __init__(self) -> None:
        """Build the projection."""
        super().__init__()
        self.to_qk = nn.Conv2d(DIM, 2 * DIM, 1, bias=False)


def coords_grid(b: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    """`[B, 2, h, w]` pixel coordinates (x, y)."""
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    return torch.stack([x, y]).float()[None].expand(b, -1, -1, -1)


class CorrBlock:
    """Upstream `corr.CorrBlock`: the all-pairs volume, its 4-level pyramid and the 9 x 9 window lookup (float32)."""

    def __init__(self, f1: torch.Tensor, f2: torch.Tensor) -> None:
        """Build the pyramid of `f1 . f2 / sqrt(C)`."""
        b, c, h, w = f1.shape
        corr = (f1.flatten(2).transpose(1, 2) @ f2.flatten(2)) / math.sqrt(c)
        corr = corr.reshape(b * h * w, 1, h, w)
        self.pyramid = [corr]
        for _ in range(LEVELS - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.pyramid.append(corr)
        d = torch.linspace(-RADIUS, RADIUS, 2 * RADIUS + 1, device=f1.device)
        self.delta = torch.stack(torch.meshgrid(d, d, indexing="ij"), dim=-1).view(1, 2 * RADIUS + 1, -1, 2)

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """`[B, 4 * 81, h, w]` windows around `coords` `[B, 2, h, w]` (upstream's (dy, dx) grid used as (x, y))."""
        b, _, h, w = coords.shape
        centre = coords.permute(0, 2, 3, 1).reshape(b * h * w, 1, 1, 2)
        out = []
        for i, corr in enumerate(self.pyramid):
            xy = centre / 2**i + self.delta
            hh, ww = corr.shape[-2:]
            grid = torch.cat([2 * xy[..., :1] / (ww - 1) - 1, 2 * xy[..., 1:] / (hh - 1) - 1], dim=-1)
            out.append(F.grid_sample(corr, grid, align_corners=True).view(b, h, w, -1))
        return torch.cat(out, dim=-1).permute(0, 3, 1, 2).contiguous()


def upsample(flow: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Upstream `upsample_flow`: `[B, 2, h, w]` -> `[B, 2, 8h, 8w]`, convex combination of the 3 x 3 neighbours."""
    b, _, h, w = flow.shape
    mask = torch.softmax(mask.view(b, 1, 9, FACTOR, FACTOR, h, w), dim=2)
    up = F.unfold(FACTOR * flow, (3, 3), padding=1).view(b, 2, 9, 1, 1, h, w)
    up = torch.sum(mask * up, dim=2).permute(0, 1, 4, 2, 5, 3)
    return up.reshape(b, 2, FACTOR * h, FACTOR * w)


class MemFlowNet(nn.Module):
    """MemFlowNet with BasicEncoder feature and context networks (upstream `cnet = fnet = 'basicencoder'`)."""

    def __init__(self, train_avg_length: float) -> None:
        """Build the network; `train_avg_length` is upstream's config value the attention scale depends on."""
        super().__init__()
        self.cnet = BasicEncoder("batch")
        self.fnet = BasicEncoder("instance")
        self.update_block = UpdateBlock()
        self.att = Attention()
        self.train_avg_length = train_avg_length

    def forward_pair(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        iters: int,
        corr_fn: Callable[[torch.Tensor, torch.Tensor], Callable[[torch.Tensor], torch.Tensor]] | None = None,
        *,
        fp16: bool = False,
    ) -> torch.Tensor:
        """Flow `[B, 2, H, W]` (float32) from `image1` to `image2` (`[-1, 1]`, H and W multiples of 8)."""
        ub = self.update_block
        with torch.autocast("cuda", torch.float16, enabled=fp16):
            net, inp = self.cnet(image1).split([DIM, DIM], dim=1)
            net, inp = torch.tanh(net), torch.relu(inp)
            q, k = self.att.to_qk(inp).chunk(2, dim=1)
            fmaps = self.fnet(torch.cat([image1, image2])).float()
        f1, f2 = fmaps.chunk(2)
        lookup = corr_fn(f1, f2) if corr_fn is not None else CorrBlock(f1, f2)
        b, _, h, w = f1.shape
        # upstream memory readout with one memory frame (the current one): scale dim^-0.5 * log_{train_avg}(h * w)
        scale = DIM**-0.5 * math.log(h * w, self.train_avg_length)
        # [B, 1, N, C] contiguous: a transposed view sends SDPA to the math path (an N x N matrix, ~4 GB at 1080p)
        q, k = (t.flatten(2).transpose(1, 2)[:, None].contiguous() for t in (q, k))
        coords0 = coords_grid(b, h, w, image1.device)
        coords1 = coords0
        for _ in range(iters):
            corr = lookup(coords1)
            flow = coords1 - coords0
            with torch.autocast("cuda", torch.float16, enabled=fp16):
                motion = ub.encoder(flow, corr)
                v = ub.aggregator.to_v(motion).flatten(2).transpose(1, 2)[:, None].contiguous()
                readout = F.scaled_dot_product_attention(q, k, v, scale=scale)
                readout = readout[:, 0].transpose(1, 2).reshape(motion.shape)
                net, delta = ub(net, inp, motion, motion + ub.aggregator.gamma * readout)
            coords1 = coords1 + delta.float()
        with torch.autocast("cuda", torch.float16, enabled=fp16):
            mask = 0.25 * ub.mask(net)
        return upsample(coords1 - coords0, mask.float())


def convert(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Upstream's state dict without the relative position embedding the network does not use at inference."""
    return {k: v for k, v in sd.items() if not k.startswith("att.pos_emb.")}
