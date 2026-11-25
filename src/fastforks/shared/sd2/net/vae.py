# fastforks: the Stable Diffusion 2 VAE of `diffusers.AutoencoderKL` (huggingface/diffusers, Apache-2.0, see
# ../LICENSE), re-implemented without diffusers for Marigold's vae/config.json (channels 128 / 256 / 512 /
# 512, two resnets per encoder block, three per decoder block, one single-head attention in each mid block). Module and
# parameter names are those of the checkpoint (diffusers 0.8 layout: the mid-block attention is `query` / `key` /
# `value` / `proj_attn`; newer checkpoints' `to_q` ... `to_out.0` are renamed on load). Only what Marigold and Lotus
# call: `encoder` + `quant_conv` (the posterior) and `post_quant_conv` + `decoder`.
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .unet import Downsample2D, ResnetBlock2D, Upsample2D, _tracing, attention

EPS = 1e-6
_UPSAMPLE_TILE_MIN_PIXELS = 256 * 256  # below this the temporary is small enough to use one convolution
MPS_STRIP_MIN_BYTES = 512 << 20
"""Bytes of one activation from which the MPS VAE runs its resnets and output convs in row strips (eager only)."""
MPS_STRIP_BYTES = 96 << 20
"""Bytes of one strip's activation. A resnet then holds its input, one full intermediate and a few strips, where the
plain forward holds five to six full activations (DepthFM 1024x1472 float32 decode: 8 GiB, here 3 GiB)."""


def _mps_strips(x: torch.Tensor, channels: int) -> int:
    """Rows per strip of `x` with `channels` channels on MPS in an eager no-grad run, 0 where the plain forward runs."""
    b, _, h, w = x.shape
    row = b * channels * w * x.element_size()
    if x.device.type != "mps" or h * row < MPS_STRIP_MIN_BYTES or _tracing() or torch.is_grad_enabled():
        return 0
    return max(1, min(MPS_STRIP_BYTES // row, (h + 1) // 2))


def _group_stats(gn: nn.GroupNorm, x: torch.Tensor, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`gn`'s per-channel mean and scale (`weight / sqrt(var + eps)`) of `x` as `[B, C, 1, 1]`, summed strip by strip.

    Float32 sums in two passes (the mean, then the squared deviations), as `F.group_norm` normalises: over each
    group's channels and all pixels.
    """
    b, c, h, w = x.shape
    g = gn.num_groups
    n = (c // g) * h * w
    parts = [x[:, :, i : i + rows] for i in range(0, h, rows)]
    mean = torch.stack([p.float().reshape(b, g, -1).sum(-1) for p in parts]).sum(0) / n
    dev = [(p.float().reshape(b, g, -1) - mean[:, :, None]).square_().sum(-1) for p in parts]
    scale = (torch.stack(dev).sum(0) / n + gn.eps).rsqrt()
    mean, scale = (t.repeat_interleave(c // g, dim=1) for t in (mean, scale))
    return mean.to(x.dtype)[:, :, None, None], (scale * gn.weight.float()).to(x.dtype)[:, :, None, None]


def _norm_silu_conv(  # noqa: PLR0913 - the two layers, their input, the strip height and the residual
    gn: nn.GroupNorm,
    conv: nn.Conv2d,
    x: torch.Tensor,
    rows: int,
    skip: torch.Tensor | None = None,
    shortcut: nn.Module | None = None,
) -> torch.Tensor:
    """`conv(silu(gn(x)))` (3x3, padding 1) in strips of `rows` rows with one source row of halo on internal edges.

    `skip`: the resnet's input, added strip by strip, through its 1x1 `shortcut` conv if it has one. Without a
    shortcut the sum is written into `skip` (no further full-size buffer).
    """
    h = x.shape[-2]
    mean, scale = _group_stats(gn, x, rows)
    bias = gn.bias.to(x.dtype)[None, :, None, None]
    out: torch.Tensor | None = None
    for start in range(0, h, rows):
        stop = min(start + rows, h)
        first, last = max(0, start - 1), min(h, stop + 1)
        t = (x[:, :, first:last] - mean).mul_(scale).add_(bias)
        pred = conv(F.silu(t))[:, :, start - first : start - first + stop - start]
        del t
        if skip is not None and shortcut is None:
            pred = pred.add_(skip[:, :, start:stop])
            out = skip
        elif skip is not None:
            pred = pred.add_(shortcut(skip[:, :, start:stop]))
        if out is None:
            out = pred.new_empty((x.shape[0], pred.shape[1], h, x.shape[-1]))
        out[:, :, start:stop].copy_(pred)
        del pred
    assert out is not None  # noqa: S101 - the input has a positive height
    return out


def _resnet(r: ResnetBlock2D, x: torch.Tensor) -> torch.Tensor:
    """`r(x)` of a VAE resnet (no time embedding); on MPS in row strips from `MPS_STRIP_MIN_BYTES` per activation.

    The strips of a resnet without a shortcut conv write their sum into `x`: the encoder and decoder only pass
    intermediates nobody else reads.
    """
    rows = _mps_strips(x, max(r.conv1.in_channels, r.conv1.out_channels))
    if not rows:
        return r(x)
    h = _norm_silu_conv(r.norm1, r.conv1, x, rows)
    return _norm_silu_conv(r.norm2, r.conv2, h, rows, x, r.conv_shortcut)


def _upsample_tiled(up: Upsample2D, x: torch.Tensor, rows: int = 0) -> torch.Tensor:
    """Nearest 2x and the 3x3 convolution in strips of `rows` source rows (default: four strips).

    Retain a source row on each internal edge.
    """
    h, w = x.shape[-2:]
    rows = rows or (h + 3) // 4
    out: torch.Tensor | None = None
    for start in range(0, h, rows):
        stop = min(start + rows, h)
        first, last = max(0, start - 1), min(h, stop + 1)
        resized = F.interpolate(x[..., first:last, :], scale_factor=2.0, mode="nearest")
        pred = up.conv(resized)
        if out is None:
            out = pred.new_empty((x.shape[0], pred.shape[1], 2 * h, 2 * w))
        offset = 2 * (start - first)
        out[..., 2 * start : 2 * stop, :].copy_(pred[..., offset : offset + 2 * (stop - start), :])
        del resized, pred
    assert out is not None  # noqa: S101 - the convolution input has a positive height
    return out


class AttentionBlock(nn.Module):
    """Single-head self-attention over all pixels with a GroupNorm in front and a residual (diffusers `Attention`)."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(32, c, eps=EPS)
        self.query = nn.Linear(c, c)
        self.key = nn.Linear(c, c)
        self.value = nn.Linear(c, c)
        self.proj_attn = nn.Linear(c, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        t = self.group_norm(x.view(b, c, h * w)).transpose(1, 2)
        q, k, v = (m(t)[:, None] for m in (self.query, self.key, self.value))
        o = attention(q, k, v)[:, 0]
        return self.proj_attn(o).transpose(1, 2).reshape(b, c, h, w) + x


class MidBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock2D(c, c, None, EPS) for _ in range(2)])
        self.attentions = nn.ModuleList([AttentionBlock(c)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))


class EncoderBlock(nn.Module):
    """`DownEncoderBlock2D`: two resnets, then a stride-2 conv after a (0, 1, 0, 1) pad (not in the last block)."""

    def __init__(self, cin: int, cout: int, *, down: bool) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock2D(cin if i == 0 else cout, cout, None, EPS) for i in range(2)])
        self.downsamplers = nn.ModuleList([Downsample2D(cout, 0)]) if down else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for r in self.resnets:
            x = _resnet(r, x)
        return x if self.downsamplers is None else self.downsamplers[0](x)


class DecoderBlock(nn.Module):
    """`UpDecoderBlock2D`: three resnets, then nearest x2 + conv (not in the last block)."""

    def __init__(self, cin: int, cout: int, *, up: bool, tile_upsample: bool = False) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock2D(cin if i == 0 else cout, cout, None, EPS) for i in range(3)])
        self.upsamplers = nn.ModuleList([Upsample2D(cout)]) if up else None
        self.tile_upsample = tile_upsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for r in self.resnets:
            x = _resnet(r, x)
        if self.upsamplers is None:
            return x
        up = self.upsamplers[0]
        rows = _mps_strips(x, 4 * x.shape[1])  # the output's bytes
        if rows:
            return _upsample_tiled(up, x, rows)
        if (
            self.tile_upsample
            and not torch.compiler.is_compiling()
            and not torch.jit.is_tracing()
            and x.device.type == "cuda"
            and x.shape[-2] * x.shape[-1] >= _UPSAMPLE_TILE_MIN_PIXELS
            and x.is_contiguous()
            and not torch.is_grad_enabled()
        ):
            return _upsample_tiled(up, x)
        return up(x)


class Encoder(nn.Module):
    def __init__(self, channels: tuple[int, ...], latent: int) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(3, channels[0], 3, padding=1)
        n = len(channels)
        self.down_blocks = nn.ModuleList(
            [EncoderBlock(channels[max(i - 1, 0)], c, down=i < n - 1) for i, c in enumerate(channels)]
        )
        self.mid_block = MidBlock(channels[-1])
        self.conv_norm_out = nn.GroupNorm(32, channels[-1], eps=EPS)
        self.conv_out = nn.Conv2d(channels[-1], 2 * latent, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for blk in self.down_blocks:
            x = blk(x)
        x = self.mid_block(x)
        return self.conv_out(F.silu(self.conv_norm_out(x)))


class Decoder(nn.Module):
    def __init__(self, channels: tuple[int, ...], latent: int) -> None:
        super().__init__()
        rc = channels[::-1]
        n = len(channels)
        self.conv_in = nn.Conv2d(latent, rc[0], 3, padding=1)
        self.mid_block = MidBlock(rc[0])
        self.up_blocks = nn.ModuleList(
            [DecoderBlock(rc[max(i - 1, 0)], c, up=i < n - 1, tile_upsample=i == n - 2) for i, c in enumerate(rc)]
        )
        self.conv_norm_out = nn.GroupNorm(32, channels[0], eps=EPS)
        self.conv_out = nn.Conv2d(channels[0], 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.mid_block(self.conv_in(z))
        for blk in self.up_blocks:
            x = blk(x)
        rows = _mps_strips(x, x.shape[1])
        if rows:
            return _norm_silu_conv(self.conv_norm_out, self.conv_out, x, rows)
        return self.conv_out(F.silu(self.conv_norm_out(x)))


def sample(moments: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """diffusers `DiagonalGaussianDistribution(moments).sample()` with the standard normal `noise` given."""
    mean, logvar = moments.chunk(2, dim=1)
    return mean + torch.exp(0.5 * logvar.clamp(-30.0, 20.0)) * noise


class AutoencoderKL(nn.Module):
    """SD2 VAE: image `[-1, 1]` <-> 4-channel latent at 1/8 resolution (unscaled; Marigold scales by 0.18215)."""

    def __init__(self, channels: tuple[int, ...] = (128, 256, 512, 512), latent: int = 4) -> None:
        super().__init__()
        self.encoder = Encoder(channels, latent)
        self.decoder = Decoder(channels, latent)
        self.quant_conv = nn.Conv2d(2 * latent, 2 * latent, 1)
        self.post_quant_conv = nn.Conv2d(latent, latent, 1)

    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Mean of the latent posterior (Marigold `encode_rgb` before scaling)."""
        return self.quant_conv(self.encoder(x)).chunk(2, dim=1)[0]

    def encode_moments(self, x: torch.Tensor) -> torch.Tensor:
        """The latent posterior's mean and log-variance, concatenated `[B, 2 * latent, h, w]` (see `sample`)."""
        return self.quant_conv(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Latent -> 3-channel output."""
        return self.decoder(self.post_quant_conv(z))
