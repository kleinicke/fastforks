"""RAFT forward for the MLX port: the same function as `net.RAFT.forward`, shaped for MLX on an Apple GPU.

Built from a loaded `RAFT` (its weights, regrouped). What differs from the portable forward, and why (M1 Pro,
1080 x 1920, 12 updates):

- No correlation volume: `corr_windows` takes each window's dot products from fmap1 and the average-pooled fmap2
  levels (the mean of the correlations is the correlation with the mean). The volume was the peak memory (4.2 GB in
  float32 before its pyramid) and its `grid_sample` lookups most of the time.
- The feature encoder runs per image: an instance norm over one sample is a view of the channels-last map, over two
  it is two transposing copies in the MLX replay.
- The GRU convolutions over `[hidden, context, motion]` split by input: the context part does not change over the
  updates, so it is convolved once (with the bias); each update convolves `[hidden, motion]` only, z and r in one
  convolution. Motion features are zero-padded to a multiple of 32 channels (small: 82 -> 96), which is what MLX's
  fast 3 x 3 path needs.
- The motion encoder's last convolution has zero output channels up to what that fast path takes (large 126 -> 128,
  small 80 -> 128); they are the padding above, and the flow takes the last two channels.

All of these are exact regroupings in real arithmetic; outputs differ from the portable forward by rounding only.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.volume_ops import corr_windows

from .net import FACTOR, RAFT, ConvGRU, _coords, _upsample

_ALIGN = 32  # fixed: MLX's fast 3 x 3 convolution needs input and output channels in multiples of 32
_WINOGRAD = 256  # fixed: ... and at least this many input + output channels (float16, 128 -> 80: 3.6 -> 2.1 ms)


def _up(n: int) -> int:
    return -(-n // _ALIGN) * _ALIGN


class _SplitGRU(nn.Module):
    """A `ConvGRU` over `x = [context, motion]` with the context part of its convolutions taken out of the loop."""

    def __init__(self, gru: ConvGRU, hidden: int, context: int, motion: int) -> None:
        """Regroup the weights: `[hidden | motion, zeros to `_up(motion)` with the flow last]` and `[context]`."""
        super().__init__()
        self.hidden = hidden
        self.pad = gru.convz.padding
        pad = _up(motion) - motion

        def loop(w: torch.Tensor) -> torch.Tensor:  # [O, hidden + context + motion, kh, kw] -> without the context
            h, _, m = w.split([hidden, context, motion], dim=1)
            return torch.cat([h, m[:, :-2], m.new_zeros(m.shape[0], pad, *m.shape[2:]), m[:, -2:]], dim=1)

        def ctx(w: torch.Tensor) -> torch.Tensor:
            return w[:, hidden : hidden + context]

        wz, wr, wq = gru.convz.weight.detach(), gru.convr.weight.detach(), gru.convq.weight.detach()
        self.register_buffer("w_zr", torch.cat([loop(wz), loop(wr)]).contiguous(), persistent=False)
        self.register_buffer("w_q", loop(wq).contiguous(), persistent=False)
        self.register_buffer("w_ctx", torch.cat([ctx(wz), ctx(wr), ctx(wq)]).contiguous(), persistent=False)
        bias = torch.cat([gru.convz.bias.detach(), gru.convr.bias.detach(), gru.convq.bias.detach()])
        self.register_buffer("b_ctx", bias, persistent=False)

    def context(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The context's share of the (z, r) and q pre-activations, biases included: once per call."""
        c = F.conv2d(context, self.w_ctx, self.b_ctx, padding=self.pad)
        return c[:, : 2 * self.hidden], c[:, 2 * self.hidden :]

    def forward(self, h: torch.Tensor, motion: torch.Tensor, c_zr: torch.Tensor, c_q: torch.Tensor) -> torch.Tensor:
        """Updated hidden state."""
        zr = torch.sigmoid(F.conv2d(torch.cat([h, motion], dim=1), self.w_zr, padding=self.pad) + c_zr)
        z, r = zr[:, : self.hidden], zr[:, self.hidden :]
        q = torch.tanh(F.conv2d(torch.cat([r * h, motion], dim=1), self.w_q, padding=self.pad) + c_q)
        return (1 - z) * h + z * q


class MlxRAFT(nn.Module):
    """`RAFT` regrouped for MLX; the same signature and result as `RAFT.forward` for one pair per call."""

    def __init__(self, net: RAFT) -> None:
        """Take over `net`'s modules; regroup the update block's weights."""
        super().__init__()
        cfg = net.cfg
        self.cfg = cfg
        self.feature_encoder = net.feature_encoder
        self.context_encoder = net.context_encoder
        self.motion_encoder = net.update_block.motion_encoder
        self.flow_head = net.update_block.flow_head
        self.mask_predictor = net.mask_predictor if cfg.mask else None
        context = cfg.context_layers[-1] - cfg.hidden
        rb = net.update_block.recurrent_block
        grus = [rb.convgru1] + ([rb.convgru2] if hasattr(rb, "convgru2") else [])
        self.grus = nn.ModuleList(_SplitGRU(g, cfg.hidden, context, cfg.motion_out) for g in grus)
        conv = self.motion_encoder.conv[0]  # -> motion_out - 2 channels, then zero channels
        w, b = conv.weight.detach(), conv.bias.detach()
        self.motion = _up(cfg.motion_out) - 2  # channels of it the GRUs take: the encoder's, then zeros
        extra = max(_up(w.shape[0]), _WINOGRAD - w.shape[1]) - w.shape[0]
        self.register_buffer("w_motion", torch.cat([w, w.new_zeros(extra, *w.shape[1:])]), persistent=False)
        self.register_buffer("b_motion", torch.cat([b, b.new_zeros(extra)]), persistent=False)

    def _motion(self, flow: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """Motion features `[B, _up(motion_out), h, w]`: the encoder's, zeros, then the flow."""
        enc = self.motion_encoder
        c = enc.convcorr2(enc.convcorr1(corr))
        f = enc.convflow2(enc.convflow1(flow))
        m = F.relu(F.conv2d(torch.cat([c, f], dim=1), self.w_motion, self.b_motion, padding=1))
        return torch.cat([m[:, : self.motion], flow], dim=1)

    def forward(self, image1: torch.Tensor, image2: torch.Tensor, iters: int) -> torch.Tensor:
        """Flow after `iters` updates, float32 `[B, 2, H, W]`."""
        cfg = self.cfg
        b, _, h, w = image1.shape
        hs, ws = h // FACTOR, w // FACTOR
        fmap1 = self.feature_encoder(image1).float()
        levels = [self.feature_encoder(image2).float()]
        for _ in range(cfg.corr_levels - 1):
            levels.append(F.avg_pool2d(levels[-1], kernel_size=2, stride=2))
        scale = 1 / math.sqrt(fmap1.shape[1])
        ctx = self.context_encoder(image1)
        hidden, context = torch.split(ctx, [cfg.hidden, ctx.shape[1] - cfg.hidden], dim=1)
        hidden, context = torch.tanh(hidden), F.relu(context)
        shares = [gru.context(context) for gru in self.grus]
        coords0 = _coords(b, hs, ws, image1.device)
        coords1 = coords0
        for _ in range(iters):
            corr = corr_windows(fmap1, levels, coords1, cfg.corr_radius, scale)
            motion = self._motion(coords1 - coords0, corr)
            for gru, (c_zr, c_q) in zip(self.grus, shares, strict=True):
                hidden = gru(hidden, motion, c_zr, c_q)
            coords1 = coords1 + self.flow_head(hidden).float()
        mask = self.mask_predictor(hidden).float() if self.mask_predictor is not None else None
        return _upsample(coords1 - coords0, mask)
