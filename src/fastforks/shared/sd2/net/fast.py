# fastforks: written for fastforks (not vendored).
"""Max-mode forward passes of the SD2 UNet and VAE (fp16, channels-last).

Same modules and weights as `unet.py` / `vae.py`, run functionally: convolutions as cuDNN fp16-accumulate graphs
(`cudnn_hacc`), GroupNorm (+ SiLU) on the channels-last map (`group_norm`), no layout copies around the
transformers, the time-embedding projections of every resnet and
the cross-attention keys / values of the (constant) empty prompt computed once at load.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks.kernels import cudnn_hacc

from .unet import (
    BasicTransformerBlock,
    DownBlock,
    MidBlock,
    ResnetBlock2D,
    Transformer2DModel,
    UNet2DConditionModel,
    UpBlock,
    timestep_features,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .vae import AttentionBlock, AutoencoderKL


def group_norm(x: torch.Tensor, m: torch.nn.GroupNorm, *, silu: bool = False) -> torch.Tensor:
    """`m(x)` (then SiLU) on a channels-last fp16 map, without layout copies.

    `F.group_norm` makes a channels-last input contiguous and returns NCHW, which the next conv copies back: two full
    copies per norm (on the full-rate A10 they made the kernels path slower than portable). Here the group statistics
    come from the channels-last view (the reduction accumulates in fp32) and are folded into a per-channel scale and
    shift. Few launches (7 with SiLU): on fast cards (L40S) the kernels forward is launch-bound.
    """
    n, c, h, w = x.shape
    g, cg = m.num_groups, c // m.num_groups
    var, mean = torch.var_mean(x.permute(0, 2, 3, 1).reshape(n, h * w, g, cg), dim=(1, 3), correction=0)
    scale = (var + m.eps).rsqrt_()[:, :, None] * m.weight.view(g, cg)  # [n, g, cg]
    shift = torch.addcmul(m.bias.view(g, cg), mean[:, :, None], scale, value=-1)
    y = torch.addcmul(shift.view(n, c, 1, 1), x, scale.view(n, c, 1, 1))
    return F.silu(y, inplace=True) if silu else y


class Fast:
    """The fp16 channels-last network plus the per-timestep and per-prompt constants."""

    def __init__(
        self, unet: UNet2DConditionModel, text: torch.Tensor, timesteps: Sequence[float], labels: torch.Tensor | None = None
    ) -> None:
        """`timesteps`: the UNet's timesteps (diffusion steps, or flow-matching times in [0, 1]).

        `labels`: the `[1, class_embed]` class labels of a UNet with a class embedding (constant).
        """
        self.unet = unet
        self.kv: dict[BasicTransformerBlock, tuple[torch.Tensor, torch.Tensor]] = {}
        for m in unet.modules():
            if isinstance(m, BasicTransformerBlock):
                a = m.attn2
                k = a.to_k(text).view(1, text.shape[1], a.heads, -1).transpose(1, 2)
                v = a.to_v(text).view(1, text.shape[1], a.heads, -1).transpose(1, 2)
                self.kv[m] = (k, v)
        resnets = [m for m in unet.modules() if isinstance(m, ResnetBlock2D) and m.time_emb_proj is not None]
        self.temb: dict[float, dict[ResnetBlock2D, torch.Tensor]] = {}
        for t in timesteps:
            tf = timestep_features(torch.tensor([t], device=text.device), unet.conv_in.out_channels).to(text.dtype)
            e = unet.time_embedding(tf)
            if unet.class_embedding is not None:
                e = e + unet.class_embedding(labels.to(text.dtype))
            e = F.silu(e)
            self.temb[t] = {r: r.time_emb_proj(e)[:, :, None, None] for r in resnets}

    def conv(self, x: torch.Tensor, c: torch.nn.Conv2d) -> torch.Tensor:
        """A 2-D conv as a cuDNN fp16-accumulate graph, channels-last out."""
        return cudnn_hacc.conv_half_acc(x, c.weight, c.bias, list(c.padding), list(c.stride), list(c.dilation))

    def resnet(self, r: ResnetBlock2D, x: torch.Tensor, proj: torch.Tensor | None) -> torch.Tensor:
        """`ResnetBlock2D.forward`."""
        h = self.conv(group_norm(x, r.norm1, silu=True), r.conv1)
        if proj is not None:
            h = h.add_(proj)
        h = self.conv(group_norm(h, r.norm2, silu=True), r.conv2)
        if r.conv_shortcut is not None:
            x = self.conv(x, r.conv_shortcut)
        return h.add_(x)

    def transformer(self, m: Transformer2DModel, x: torch.Tensor) -> torch.Tensor:
        """`Transformer2DModel.forward` on a channels-last map (the token views need no copies)."""
        b, c, h, w = x.shape
        t = m.proj_in(group_norm(x, m.norm).permute(0, 2, 3, 1).reshape(b, h * w, c))
        blk = m.transformer_blocks[0]
        t = t.add_(blk.attn1(blk.norm1(t)))
        a = blk.attn2
        q = a.to_q(blk.norm2(t)).view(b, h * w, a.heads, -1).transpose(1, 2)
        k, v = self.kv[blk]
        o = F.scaled_dot_product_attention(q, k.expand(b, -1, -1, -1), v.expand(b, -1, -1, -1))
        t = t.add_(a.to_out[0](o.transpose(1, 2).reshape(b, h * w, c)))
        geglu, out = blk.ff.net[0], blk.ff.net[2]
        u, gate = geglu.proj(blk.norm3(t)).chunk(2, dim=-1)
        t = t.add_(out(F.gelu(gate).mul_(u)))
        return m.proj_out(t).view(b, h, w, c).permute(0, 3, 1, 2).add_(x)

    def down(self, blk: DownBlock, x: torch.Tensor, temb: dict) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """`DownBlock.forward`."""
        skips = []
        for i, r in enumerate(blk.resnets):
            x = self.resnet(r, x, temb[r])
            if hasattr(blk, "attentions"):
                x = self.transformer(blk.attentions[i], x)
            skips.append(x)
        if blk.downsamplers is not None:
            x = self.conv(x, blk.downsamplers[0].conv)
            skips.append(x)
        return x, skips

    def up(
        self, blk: UpBlock, x: torch.Tensor, skips: list[torch.Tensor], temb: dict, size: tuple | None
    ) -> torch.Tensor:
        """`UpBlock.forward`."""
        for i, r in enumerate(blk.resnets):
            x = self.resnet(r, torch.cat([x, skips.pop()], dim=1), temb[r])
            if hasattr(blk, "attentions"):
                x = self.transformer(blk.attentions[i], x)
        if blk.upsamplers is not None:
            if size is None:
                x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            else:
                x = F.interpolate(x, size=size, mode="nearest")
            x = self.conv(x, blk.upsamplers[0].conv)
        return x

    def mid(self, blk: MidBlock, x: torch.Tensor, temb: dict) -> torch.Tensor:
        """`MidBlock.forward`."""
        x = self.resnet(blk.resnets[0], x, temb[blk.resnets[0]])
        x = self.transformer(blk.attentions[0], x)
        return self.resnet(blk.resnets[1], x, temb[blk.resnets[1]])

    def encode(self, x: torch.Tensor, t: float) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """`UNet2DConditionModel.encode` (mid output, skips) for one of the load-time timesteps."""
        u, temb = self.unet, self.temb[t]
        x = self.conv(x.contiguous(memory_format=torch.channels_last), u.conv_in)
        skips = [x]
        for blk in u.down_blocks:
            x, s = self.down(blk, x, temb)
            skips += s
        return self.mid(u.mid_block, x, temb), skips

    def unet_forward(
        self, x: torch.Tensor, t: float, residuals: tuple[list[torch.Tensor], torch.Tensor] | None = None
    ) -> torch.Tensor:
        """`UNet2DConditionModel.forward(x, t, empty prompt, residuals=...)` for one of the load-time timesteps."""
        u, temb = self.unet, self.temb[t]
        odd = any(s % 2 ** (len(u.up_blocks) - 1) for s in x.shape[-2:])
        x, skips = self.encode(x, t)
        if residuals is not None:
            skips = [s.add_(r) for s, r in zip(skips, residuals[0], strict=True)]
            x = x.add_(residuals[1])
        for blk in u.up_blocks:
            mine = skips[-len(blk.resnets) :]
            del skips[-len(blk.resnets) :]
            size = tuple(skips[-1].shape[2:]) if odd and blk.upsamplers is not None else None
            x = self.up(blk, x, mine, temb, size)
        return self.conv(group_norm(x, u.conv_norm_out, silu=True), u.conv_out)

    def control(self, x: torch.Tensor, t: float) -> tuple[list[torch.Tensor], torch.Tensor]:
        """`ControlNetEncoder.forward` (this `Fast` holds the ControlNet as `unet`)."""
        x, skips = self.encode(x, t)
        return skips, x

    def vae_attention(self, m: AttentionBlock, x: torch.Tensor) -> torch.Tensor:
        """`vae.AttentionBlock.forward` on a channels-last map."""
        b, c, h, w = x.shape
        t = group_norm(x, m.group_norm).permute(0, 2, 3, 1).reshape(b, h * w, c)
        q, k, v = (lin(t)[:, None] for lin in (m.query, m.key, m.value))
        o = F.scaled_dot_product_attention(q, k, v)[:, 0]
        return m.proj_attn(o).view(b, h, w, c).permute(0, 3, 1, 2).add_(x)

    def vae_mid(self, m: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """`vae.MidBlock.forward`."""
        x = self.resnet(m.resnets[0], x, None)
        x = self.vae_attention(m.attentions[0], x)
        return self.resnet(m.resnets[1], x, None)

    def encode_mean(self, vae: AutoencoderKL, x: torch.Tensor) -> torch.Tensor:
        """`AutoencoderKL.encode_mean`."""
        return self.encode_moments(vae, x).chunk(2, dim=1)[0]

    def encode_moments(self, vae: AutoencoderKL, x: torch.Tensor) -> torch.Tensor:
        """`AutoencoderKL.encode_moments`."""
        e = vae.encoder
        x = self.conv(x.contiguous(memory_format=torch.channels_last), e.conv_in)
        for blk in e.down_blocks:
            for r in blk.resnets:  # ty: ignore[not-iterable] - ModuleList
                x = self.resnet(r, x, None)
            if blk.downsamplers is not None:
                x = self.conv(F.pad(x, (0, 1, 0, 1)), blk.downsamplers[0].conv)
        x = self.vae_mid(e.mid_block, x)
        x = self.conv(group_norm(x, e.conv_norm_out, silu=True), e.conv_out)
        return self.conv(x, vae.quant_conv)

    def decode(self, vae: AutoencoderKL, z: torch.Tensor) -> torch.Tensor:
        """`AutoencoderKL.decode`."""
        d = vae.decoder
        x = self.conv(self.conv(z.contiguous(memory_format=torch.channels_last), vae.post_quant_conv), d.conv_in)
        x = self.vae_mid(d.mid_block, x)
        for blk in d.up_blocks:
            for r in blk.resnets:  # ty: ignore[not-iterable] - ModuleList
                x = self.resnet(r, x, None)
            if blk.upsamplers is not None:
                x = self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"), blk.upsamplers[0].conv)
        return self.conv(group_norm(x, d.conv_norm_out, silu=True), d.conv_out)
