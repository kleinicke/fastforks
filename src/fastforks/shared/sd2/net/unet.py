# fastforks: the Stable Diffusion 2 UNet of `diffusers.UNet2DConditionModel` (huggingface/diffusers, Apache-2.0, see
# ../LICENSE), re-implemented without diffusers for the configuration Marigold ships (unet/config.json:
# CrossAttnDownBlock2D x 3 + DownBlock2D, UNetMidBlock2DCrossAttn, UpBlock2D + CrossAttnUpBlock2D x 3, linear
# projections, GEGLU feed-forward, flip_sin_to_cos timestep embedding). Module and parameter names are diffusers', so
# the checkpoint loads unchanged. Inference only: no dropout (identity in eval), no gradient checkpointing, no LoRA /
# adapter / ControlNet hooks; attention is `scaled_dot_product_attention` (diffusers' default `AttnProcessor2_0`).
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ResnetBlock2D(nn.Module):
    """diffusers `ResnetBlock2D` (time_embedding_norm "default", output_scale_factor 1, 1x1 shortcut)."""

    def __init__(self, cin: int, cout: int, temb: int | None, eps: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, cin, eps=eps)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.time_emb_proj = nn.Linear(temb, cout) if temb else None
        self.norm2 = nn.GroupNorm(32, cout, eps=eps)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(cin, cout, 1) if cin != cout else None

    def forward(self, x: torch.Tensor, temb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        if self.time_emb_proj is not None and temb is not None:
            h = h + self.time_emb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class Downsample2D(nn.Module):
    """diffusers `Downsample2D` with a conv: stride 2, padding 1 (UNet) or 0 after a (0, 1, 0, 1) zero pad (VAE)."""

    def __init__(self, c: int, padding: int) -> None:
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding == 0:
            x = F.pad(x, (0, 1, 0, 1))
        return self.conv(x)


class Upsample2D(nn.Module):
    """diffusers `Upsample2D` with a conv: nearest x2 (or to `size`), then a 3x3 conv."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x: torch.Tensor, size: tuple[int, int] | None = None) -> torch.Tensor:
        if size is None:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        else:
            x = F.interpolate(x, size=size, mode="nearest")
        return self.conv(x)


class Attention(nn.Module):
    """diffusers `Attention` (no q/k/v bias, output bias) with `AttnProcessor2_0`; `ctx` = cross-attention states."""

    def __init__(self, dim: int, heads: int, kv_dim: int | None = None) -> None:
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(kv_dim or dim, dim, bias=False)
        self.to_v = nn.Linear(kv_dim or dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Identity()])  # [1] = dropout

    def forward(self, x: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        b, n, c = x.shape
        ctx = x if ctx is None else ctx
        q = self.to_q(x).view(b, n, self.heads, -1).transpose(1, 2)
        k = self.to_k(ctx).view(b, ctx.shape[1], self.heads, -1).transpose(1, 2)
        v = self.to_v(ctx).view(b, ctx.shape[1], self.heads, -1).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.to_out[0](o.transpose(1, 2).reshape(b, n, c))


class GEGLU(nn.Module):
    def __init__(self, dim: int, inner: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, inner * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, gate = self.proj(x).chunk(2, dim=-1)
        return h * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.ModuleList([GEGLU(dim, 4 * dim), nn.Identity(), nn.Linear(4 * dim, dim)])  # [1] = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.net:
            x = m(x)
        return x


class BasicTransformerBlock(nn.Module):
    """Self-attention, cross-attention, GEGLU feed-forward; pre-LayerNorm with residuals."""

    def __init__(self, dim: int, heads: int, cross: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = Attention(dim, heads, cross)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), ctx) + x
        return self.ff(self.norm3(x)) + x


class Transformer2DModel(nn.Module):
    """diffusers `Transformer2DModel` with `use_linear_projection=True` and one block."""

    def __init__(self, dim: int, heads: int, cross: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, dim, eps=1e-6)
        self.proj_in = nn.Linear(dim, dim)
        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(dim, heads, cross)])
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        t = self.norm(x).permute(0, 2, 3, 1).reshape(b, h * w, c)
        t = self.proj_in(t)
        for blk in self.transformer_blocks:
            t = blk(t, ctx)
        t = self.proj_out(t)
        return t.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous() + x


class DownBlock(nn.Module):
    """`CrossAttnDownBlock2D` (`heads` set) or `DownBlock2D`; returns the output and the skip states."""

    def __init__(self, cin: int, cout: int, temb: int, heads: int | None, cross: int, *, down: bool) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock2D(cin if i == 0 else cout, cout, temb, 1e-5) for i in range(2)])
        if heads:
            self.attentions = nn.ModuleList([Transformer2DModel(cout, heads, cross) for _ in range(2)])
        self.downsamplers = nn.ModuleList([Downsample2D(cout, 1)]) if down else None

    def forward(self, x: torch.Tensor, temb: torch.Tensor, ctx: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips = []
        for i, r in enumerate(self.resnets):
            x = r(x, temb)
            if hasattr(self, "attentions"):
                x = self.attentions[i](x, ctx)
            skips.append(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
            skips.append(x)
        return x, skips


class UpBlock(nn.Module):
    """`CrossAttnUpBlock2D` (`heads` set) or `UpBlock2D`: three resnets on the concatenated skip states."""

    def __init__(
        self, cin: int, cprev: int, cout: int, temb: int, heads: int | None, cross: int, *, up: bool
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D((cprev if i == 0 else cout) + (cin if i == 2 else cout), cout, temb, 1e-5)
                for i in range(3)
            ]
        )
        if heads:
            self.attentions = nn.ModuleList([Transformer2DModel(cout, heads, cross) for _ in range(3)])
        self.upsamplers = nn.ModuleList([Upsample2D(cout)]) if up else None

    def forward(
        self,
        x: torch.Tensor,
        skips: list[torch.Tensor],
        temb: torch.Tensor,
        ctx: torch.Tensor,
        size: tuple[int, int] | None,
    ) -> torch.Tensor:
        for i, r in enumerate(self.resnets):
            x = r(torch.cat([x, skips.pop()], dim=1), temb)
            if hasattr(self, "attentions"):
                x = self.attentions[i](x, ctx)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x, size)
        return x


class MidBlock(nn.Module):
    """`UNetMidBlock2DCrossAttn`: resnet, transformer, resnet."""

    def __init__(self, c: int, temb: int, heads: int, cross: int) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock2D(c, c, temb, 1e-5) for _ in range(2)])
        self.attentions = nn.ModuleList([Transformer2DModel(c, heads, cross)])

    def forward(self, x: torch.Tensor, temb: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        x = self.resnets[0](x, temb)
        x = self.attentions[0](x, ctx)
        return self.resnets[1](x, temb)


class TimestepEmbedding(nn.Module):
    def __init__(self, cin: int, dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(cin, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(x)))


def timestep_features(t: torch.Tensor, dim: int) -> torch.Tensor:
    """diffusers `get_timestep_embedding(t, dim, flip_sin_to_cos=True, downscale_freq_shift=0)`, float32 `[B, dim]`."""
    half = dim // 2
    exponent = -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    emb = t[:, None].float() * torch.exp(exponent)[None, :]
    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


class UNet2DConditionModel(nn.Module):
    """The SD2 UNet (Marigold: 8 input channels = image latent + depth latent, 4 output channels, v-prediction)."""

    def __init__(
        self,
        cin: int = 8,
        cout: int = 4,
        channels: tuple[int, ...] = (320, 640, 1280, 1280),
        heads: tuple[int, ...] = (5, 10, 20, 20),
        cross: int = 1024,
    ) -> None:
        super().__init__()
        temb = channels[0] * 4
        self.conv_in = nn.Conv2d(cin, channels[0], 3, padding=1)
        self.time_embedding = TimestepEmbedding(channels[0], temb)
        last = len(channels) - 1
        self.down_blocks = nn.ModuleList(
            [
                DownBlock(
                    channels[max(i - 1, 0)], c, temb, heads[i] if i < last else None, cross, down=i < last
                )
                for i, c in enumerate(channels)
            ]
        )
        self.mid_block = MidBlock(channels[-1], temb, heads[-1], cross)
        rc, rh = channels[::-1], heads[::-1]
        self.up_blocks = nn.ModuleList(
            [
                UpBlock(
                    rc[min(i + 1, last)], rc[max(i - 1, 0)], c, temb, rh[i] if i > 0 else None, cross, up=i < last
                )
                for i, c in enumerate(rc)
            ]
        )
        self.conv_norm_out = nn.GroupNorm(32, channels[0], eps=1e-5)
        self.conv_out = nn.Conv2d(channels[0], cout, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """`x` `[B, 8, h, w]`, integer timestep `t` (0-dim), text states `ctx` `[B, n, 1024]` -> `[B, 4, h, w]`."""
        odd = any(s % 2 ** (len(self.up_blocks) - 1) for s in x.shape[-2:])  # forward_upsample_size
        tf = timestep_features(t.reshape(1).expand(x.shape[0]), self.conv_in.out_channels).to(x.dtype)
        temb = self.time_embedding(tf)
        x = self.conv_in(x)
        skips = [x]
        for blk in self.down_blocks:
            x, s = blk(x, temb, ctx)
            skips += s
        x = self.mid_block(x, temb, ctx)
        for blk in self.up_blocks:
            mine = skips[-len(blk.resnets) :]
            del skips[-len(blk.resnets) :]
            size = tuple(skips[-1].shape[2:]) if odd and blk.upsamplers is not None else None
            x = blk(x, mine, temb, ctx, size)
        return self.conv_out(F.silu(self.conv_norm_out(x)))
