"""UniK3D network (lpiccinelli-eth/UniK3D 29f7862, CC BY-NC-SA 4.0), with upstream's parameter names.

A re-implementation of the inference path of `UniK3D` with the released ViT-L config: a DINOv2 ViT-L/14 encoder
(no registers, LayerScale, final norm on the four used layers 6/12/18/24) and the camera-prompted decoder. The angular
module turns the four class tokens into a field of view, a principal point and degree-1..3 spherical-harmonic
coefficients, from which the per-pixel rays follow; the radial module conditions the patch features on the rays
(cross-attention to their Fourier embedding) and decodes the log-radius and the log-confidence with residual
upsampling convs. Training-only parts (gradient choker, level / sine position embeddings, losses) are left out.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.checkpoint import load_strict
from fastforks.shared.depth_anything_v2 import vit_large

HIDDEN = 512  # decoder width
HEADS = 8
EXPANSION = 4
PATCH = 14
LAYERS = (5, 11, 17, 23)  # encoder blocks feeding the decoder ("last" stacking of output_idx 6, 12, 18, 24)
UPS = ((512, 512), (512, 256), (256, 128))  # residual upsampling stages: (channels, output channels)
RATIO_BOUNDS = (0.5, 2.5)  # released config data.shape_constraints
PIXELS_BOUNDS = (200000.0, 600000.0)
SH_PARAMS = (3, 3, 5, 7)  # angular module tokens: pinhole (hfov, cx, cy), SH degree 1, 2, 3


def vit() -> Any:  # noqa: ANN401 - the DINOv2 module (duck-typed attributes)
    """The ViT-L/14 encoder (518 position table, LayerScale, no registers; upstream's LayerNorms use eps 1e-5)."""
    m = vit_large(patch_size=PATCH, img_size=518, init_values=1.0, block_chunks=0, interpolate_offset=0.0)
    for ln in m.modules():
        if isinstance(ln, nn.LayerNorm):
            ln.eps = 1e-5
    return m


# ------------------------------------------------------------------------------------------------ geometry
def get_paddings(shape: tuple[int, int]) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """Upstream `get_paddings`: zero padding (left, right, top, bottom) to the aspect-ratio bounds, padded size."""
    h, w = shape
    ratio = w / h
    target = min(RATIO_BOUNDS[1], max(RATIO_BOUNDS[0], ratio))
    if ratio > target:
        hn = round(w / target)
        top = (hn - h) // 2
        return (0, 0, top, hn - h - top), (hn, w)
    if ratio < target:
        wn = round(h * target)
        left = (wn - w) // 2
        return (left, wn - w - left, 0, 0), (h, wn)
    return (0, 0, 0, 0), (h, w)


def get_resize_factor(shape: tuple[int, int], level: int) -> tuple[float, tuple[int, int]]:
    """Upstream `get_resize_factor` with `infer`'s `resolution_level` bounds: factor, network size (multiple of 14)."""
    lo, hi = PIXELS_BOUNDS
    step = (hi - lo) / 10
    lo, hi = level * step + lo, (level + 1) * step + lo
    h, w = shape
    n = w * h
    factor = (min(hi, max(lo, n)) / n) ** 0.5
    nh, nw = round(h * factor), round(w * factor)
    return factor, (math.ceil(nh / PATCH) * PATCH, math.ceil(nw / PATCH) * PATCH)


def coords_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Pixel-centre coordinates `[2, H, W]` (x, y), as upstream `coords_grid`."""
    x = torch.linspace(0.5, w - 0.5, w, device=device)
    y = torch.linspace(0.5, h - 0.5, h, device=device)
    return torch.stack([x.repeat(h, 1), y.repeat(w, 1).t()])


def pinhole_rays(k: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Unit rays `[B, 3, H, W]` of pinhole intrinsics `[B, 3, 3]` (upstream `Pinhole.get_rays`, float32)."""
    uv = coords_grid(h, w, k.device).reshape(2, -1)
    uv = torch.cat([uv, torch.ones_like(uv[:1])])
    xyz = torch.inverse(k.float()) @ uv
    xyz = xyz / xyz[:, -1:].clip(min=1e-4)
    xyz = xyz.reshape(-1, 3, h, w)
    return xyz / torch.norm(xyz, dim=1, keepdim=True).clamp(min=1e-4)


def rsh_cart_3(xyz: torch.Tensor) -> torch.Tensor:
    """Real spherical harmonics of degree 1-3 (upstream `rsh_cart_3` without the constant one) `[..., 15]`."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    x2, y2, z2 = x**2, y**2, z**2
    xy, xz, yz = x * y, x * z, y * z
    return torch.stack(
        [
            -0.48860251190292 * y,
            0.48860251190292 * z,
            -0.48860251190292 * x,
            1.09254843059208 * xy,
            -1.09254843059208 * yz,
            0.94617469575756 * z2 - 0.31539156525252,
            -1.09254843059208 * xz,
            0.54627421529604 * x2 - 0.54627421529604 * y2,
            -0.590043589926644 * y * (3.0 * x2 - y2),
            2.89061144264055 * xy * z,
            0.304697199642977 * y * (1.5 - 7.5 * z2),
            1.24392110863372 * z * (1.5 * z2 - 0.5) - 0.497568443453487 * z,
            0.304697199642977 * x * (1.5 - 7.5 * z2),
            1.44530572132028 * z * (x2 - y2),
            -0.590043589926644 * x * (x2 - 3.0 * y2),
        ],
        -1,
    )


def fourier_features(x: torch.Tensor, dim: int, max_freq: int) -> torch.Tensor:
    """Upstream `generate_fourier_features(use_log=True, use_cos=False, cat_orig=False)`."""
    bands = dim // x.shape[-1]
    scales = 2.0 ** torch.linspace(0.0, math.log2(max_freq), steps=bands, device=x.device, dtype=x.dtype)
    return (x.unsqueeze(-1) * scales * math.pi).sin().flatten(-2)


# ------------------------------------------------------------------------------------------------ layers
class LayerScale(nn.Module):
    """`x * gamma`."""

    def __init__(self, dim: int) -> None:
        """Per-channel scale."""
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale."""
        return x * self.gamma


class MLP(nn.Module):
    """LayerNorm, Linear, GELU, Linear."""

    def __init__(self, dim: int, expansion: int, output_dim: int | None = None) -> None:
        """Hidden width `dim * expansion`."""
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj1 = nn.Linear(dim, dim * expansion)
        self.proj2 = nn.Linear(dim * expansion, output_dim or dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MLP."""
        return self.proj2(self.act(self.proj1(self.norm(x))))


class AttentionBlock(nn.Module):
    """Pre-norm (cross-)attention block: `x + ls1(attn(x, context))`, `x + ls2(mlp(x))`."""

    def __init__(self, dim: int, *, layer_scale: bool) -> None:
        """Bias-free q / kv / out projections; LayerScale only if `layer_scale`."""
        super().__init__()
        self.num_heads = HEADS
        self.mlp = MLP(dim, EXPANSION)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.q = nn.Linear(dim, dim, bias=False)
        self.norm_attnx = nn.LayerNorm(dim)
        self.norm_attnctx = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, dim, bias=False)
        self.ls1_1 = LayerScale(dim) if layer_scale else nn.Identity()
        self.ls2 = LayerScale(dim) if layer_scale else nn.Identity()

    def attn(self, x: torch.Tensor, context: torch.Tensor, pos_embed: torch.Tensor | None) -> torch.Tensor:
        """Multi-head attention of `x` to `context` (`pos_embed` added to the queries)."""
        b, n, c = x.shape
        h = self.num_heads
        kv = self.kv(self.norm_attnctx(context)).view(b, context.shape[1], 2, h, c // h)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        q = self.q(self.norm_attnx(x)).view(b, n, h, c // h).transpose(1, 2)
        if pos_embed is not None:
            q = q + pos_embed.view(pos_embed.shape[0], n, h, c // h).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.out(o.transpose(1, 2).reshape(b, n, c))

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None, pos_embed: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Block (self-attention without `context`)."""
        x = self.ls1_1(self.attn(x, x if context is None else context, pos_embed)) + x
        return self.ls2(self.mlp(x)) + x


class AttentionLayer(nn.Module):
    """One cross-attention block without LayerScale (the camera prompt)."""

    def __init__(self) -> None:
        """`layers.0`."""
        super().__init__()
        self.layers = nn.ModuleList([AttentionBlock(HIDDEN, layer_scale=False)])

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Condition `x` on `context`."""
        for layer in self.layers:
            x = layer(x, context)
        return x


class ListAdapter(nn.Module):
    """One Linear per encoder level."""

    def __init__(self) -> None:
        """Four 1024 -> 512 projections."""
        super().__init__()
        self.input_adapters = nn.ModuleList([nn.Linear(1024, HIDDEN) for _ in LAYERS])

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Project each level."""
        return [a(x) for a, x in zip(self.input_adapters, xs, strict=True)]


class ResidualConvUnit(nn.Module):
    """`x + gamma * conv2(leaky(conv1(leaky(x))))`."""

    def __init__(self, dim: int) -> None:
        """Two 3x3 convs with bias, LeakyReLU 0.01, per-channel gamma."""
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.activation = nn.LeakyReLU()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual unit."""
        return self.gamma * self.conv2(self.activation(self.conv1(self.activation(x)))) + x


class ResUpsampleBil(nn.Module):
    """Two residual units, a 1x1 conv and a 2x bilinear upsampling."""

    def __init__(self, dim: int, output_dim: int) -> None:
        """`convs.{0,1}`, `up.0`."""
        super().__init__()
        self.convs = nn.ModuleList([ResidualConvUnit(dim) for _ in range(2)])
        self.up = nn.Sequential(
            nn.Conv2d(dim, output_dim, 1), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine and upsample."""
        for conv in self.convs:
            x = conv(x)
        return self.up(x)


def _head(dim: int) -> tuple[nn.Module, nn.Module, nn.Module]:
    """(LayerNorm + Linear, low-resolution 3x3 conv, high-resolution conv head) of a log-depth / log-confidence map."""
    mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
    lr = nn.Conv2d(dim, dim // 2, 3, padding=1, padding_mode="reflect")
    hr = nn.Sequential(
        nn.Conv2d(dim // 2, 32, 3, padding=1, padding_mode="reflect"), nn.LeakyReLU(), nn.Conv2d(32, 1, 1)
    )
    return mlp, lr, hr


# ------------------------------------------------------------------------------------------------ decoder
class AngularModule(nn.Module):
    """Camera head: class tokens -> field of view, principal point and spherical-harmonic ray coefficients."""

    def __init__(self) -> None:
        """Token projections, two self-attention blocks, output MLPs."""
        super().__init__()
        self.aggregate1 = AttentionBlock(HIDDEN, layer_scale=True)
        self.aggregate2 = AttentionBlock(HIDDEN, layer_scale=True)
        self.latents_pos = nn.Parameter(torch.zeros(1, sum(SH_PARAMS), HIDDEN))
        self.project_pin = nn.Linear(HIDDEN, SH_PARAMS[0] * HIDDEN, bias=False)
        self.project_deg1 = nn.Linear(HIDDEN, SH_PARAMS[1] * HIDDEN, bias=False)
        self.project_deg2 = nn.Linear(HIDDEN, SH_PARAMS[2] * HIDDEN, bias=False)
        self.project_deg3 = nn.Linear(HIDDEN, SH_PARAMS[3] * HIDDEN, bias=False)
        self.out_pinhole = MLP(HIDDEN, 1, 1)
        self.out_deg1 = MLP(HIDDEN, 1, 3)
        self.out_deg2 = MLP(HIDDEN, 1, 3)
        self.out_deg3 = MLP(HIDDEN, 1, 3)

    def forward(self, cls_tokens: torch.Tensor, shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """`[B, 4, 512]` tokens -> (hfov, vfov, cx, cy) `[B, 4]` and SH coefficients `[B, 15, 3]` (network size)."""
        b = cls_tokens.shape[0]
        pos = self.latents_pos.expand(b, -1, -1)
        projs = (self.project_pin, self.project_deg1, self.project_deg2, self.project_deg3)
        tokens = torch.cat(
            [p(t).view(b, n, HIDDEN) for p, t, n in zip(projs, cls_tokens.chunk(4, 1), SH_PARAMS, strict=True)], 1
        )
        tokens = self.aggregate2(self.aggregate1(tokens, pos_embed=pos), pos_embed=pos)
        t_pin, t1, t2, t3 = torch.split(tokens, list(SH_PARAMS), 1)
        x = self.out_pinhole(t_pin).squeeze(-1)
        sh = torch.cat([self.out_deg1(t1), self.out_deg2(t2), self.out_deg3(t3)], 1)
        hfov, cx, cy = x.unbind(-1)
        h, w = shape
        hfov = torch.sigmoid(hfov - 1.1)
        k = torch.stack([hfov, hfov * (h / w), torch.sigmoid(cx), torch.sigmoid(cy)], 1)
        return torch.tensor([2 * math.pi, 2 * math.pi, w, h], device=x.device, dtype=x.dtype) * k, sh


def camera_rays(fov: torch.Tensor, sh: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Upstream `run_camera`: predicted unit rays `[B, H * W, 3]`."""
    uv = coords_grid(h, w, sh.device)
    lon = (uv[0] - fov[:, 2].view(-1, 1, 1)) / w * fov[:, 0].view(-1, 1, 1)
    lat = (uv[1] - fov[:, 3].view(-1, 1, 1)) / h * fov[:, 1].view(-1, 1, 1)
    sphere = torch.stack([torch.cos(lat) * torch.sin(lon), -torch.sin(lat), torch.cos(lat) * torch.cos(lon)], -1)
    sphere = sphere / torch.norm(sphere, dim=-1, keepdim=True).clip(min=1e-5)
    rays = torch.einsum("bhwc,bcd->bhwd", rsh_cart_3(sphere), sh)
    rays = rays / torch.norm(rays, dim=-1, keepdim=True).clip(min=1e-5)
    return rays.reshape(rays.shape[0], h * w, 3)


def embed_rays(rays: torch.Tensor, size: tuple[int, int], grid: tuple[int, int]) -> torch.Tensor:
    """Upstream `RadialModule.embed_rays`: rays `[B, H * W, 3]` -> Fourier embedding `[B, h * w, 512]` of the grid."""
    b = rays.shape[0]
    if size != grid:
        t = rays.view(b, *size, 3).permute(0, 3, 1, 2)
        t = F.interpolate(t, size=grid, mode="bilinear", align_corners=False, antialias=True)
        rays = t.reshape(b, 3, -1).permute(0, 2, 1)
    rays = rays / torch.norm(rays, dim=-1, keepdim=True).clip(min=1e-4)
    x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
    polar = torch.acos(z)
    x_clipped = x.abs().clip(min=1e-3) * (2 * (x >= 0).int() - 1)
    azimuth = torch.atan2(y, x_clipped)
    return fourier_features(torch.stack([polar, azimuth], -1), HIDDEN, max(grid) // 2)


class RadialModule(nn.Module):
    """Camera-prompted dense decoder: log-radius and log-confidence at the network resolution."""

    def __init__(self) -> None:
        """Upstream layout for hidden 512, depths (2, 2, 2), out_dim 64."""
        super().__init__()
        self.to_latents = nn.Linear(HIDDEN, HIDDEN)
        self.prompt_camera = nn.ModuleList([AttentionLayer() for _ in LAYERS])
        self.process_features = nn.ModuleList(
            [nn.ConvTranspose2d(HIDDEN, c, max(1, 2 * i), max(1, 2 * i)) for i, (c, _) in enumerate(UPS)]
        )
        self.ups = nn.ModuleList([ResUpsampleBil(c, o) for c, o in UPS])
        dim = UPS[-1][1]
        mlp, self.to_depth_lr, self.to_depth_hr = _head(dim)
        self.depth_mlp = nn.ModuleList([nn.Identity(), nn.Identity(), mlp])
        self.confidence_mlp, self.to_confidence_lr, self.to_confidence_hr = _head(dim)

    def forward(
        self, features: list[torch.Tensor], rays: torch.Tensor, size: tuple[int, int], *, confidence: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Level features `[B, h, w, 512]`, rays `[B, H * W, 3]` -> log-radius (and log-confidence) `[B, 1, H, W]`."""
        b, h, w, _ = features[0].shape
        emb = embed_rays(rays, size, (h, w))
        cond = [p(f.reshape(b, h * w, -1), emb) for p, f in zip(self.prompt_camera, features, strict=True)]
        x = self.to_latents(cond[0]).view(b, h, w, -1).permute(0, 3, 1, 2)
        cond = [c.view(b, h, w, -1).permute(0, 3, 1, 2) for c in cond]
        for i, up in enumerate(self.ups):
            x = up(x + self.process_features[i](cond[i + 1]))
        heads = [(self.depth_mlp[-1], self.to_depth_lr, self.to_depth_hr)]
        if confidence:
            heads.append((self.confidence_mlp, self.to_confidence_lr, self.to_confidence_hr))
        outs = []
        for mlp, lr, hr in heads:  # (upstream's same-size align_corners interpolation of the depth features: identity)
            y = lr(mlp(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
            outs.append(hr(F.interpolate(y, size=size, mode="bilinear", align_corners=True)))
        return outs[0], outs[1] if confidence else None


class Decoder(nn.Module):
    """`pixel_decoder`."""

    def __init__(self) -> None:
        """Adapters, angular and radial modules."""
        super().__init__()
        self.input_adapter = ListAdapter()
        self.camera_token_adapter = ListAdapter()
        self.angular_module = AngularModule()
        self.radial_module = RadialModule()


class UniK3DNet(nn.Module):
    """`UniK3D` (encoder + decoder) for inference."""

    def __init__(self) -> None:
        """ViT-L/14 encoder and the decoder."""
        super().__init__()
        self.pixel_encoder = vit()
        self.pixel_decoder = Decoder()

    def load(self, sd: dict[str, torch.Tensor], what: str = "UniK3D checkpoint") -> UniK3DNet:
        """Load upstream's state dict strictly (assigned), dropping the unused register / mask tokens and level embeds.

        Raises:
            WeightsError: The state dict does not fit (names `what` and the first keys).
        """
        unused = ("pixel_encoder.register_tokens", "pixel_encoder.mask_token", "pixel_decoder.level_embeds")
        sd = {k: v for k, v in sd.items() if k not in unused}
        load_strict(self, sd, what, not_in_checkpoint=("pixel_encoder.mask_token",), assign=True)
        return self

    def pos_table(self, h: int, w: int) -> torch.Tensor:
        """Upstream `interpolate_pos_encoding` of an `h x w` grid: fp32 bicubic by output size, `[1, 1 + hw, C]`."""
        pe = self.pixel_encoder.pos_embed.float()
        m = math.isqrt(pe.shape[1] - 1)
        p = F.interpolate(pe[:, 1:].reshape(1, m, m, -1).permute(0, 3, 1, 2), size=(h, w), mode="bicubic")
        return torch.cat([pe[:, :1], p.permute(0, 2, 3, 1).reshape(1, h * w, -1)], 1)

    def encode(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Normalised image `[B, 3, H, W]` -> per level the normed tokens `[B, h, w, C]` and cls token `[B, 1, C]`."""
        vit = self.pixel_encoder
        b, _, hh, ww = x.shape
        h, w = hh // PATCH, ww // PATCH
        t = vit.patch_embed(x)
        t = torch.cat((vit.cls_token.expand(b, -1, -1), t), 1)
        t = t + self.pos_table(h, w).to(t.dtype)
        feats, cls = [], []
        for i, blk in enumerate(vit.blocks[: LAYERS[-1] + 1]):
            t = blk(t)
            if i in LAYERS:
                n = vit.norm(t)
                feats.append(n[:, 1:].reshape(b, h, w, -1))
                cls.append(n[:, :1])
        return feats, cls

    def forward(
        self, x: torch.Tensor, rays: torch.Tensor | None, *, confidence: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Network input -> (log-radius, log-confidence or None `[B, 1, H, W]`, unit rays `[B, H * W, 3]`).

        `rays`: given camera rays `[B, H * W, 3]`, else the angular module predicts them.
        """
        size = (x.shape[-2], x.shape[-1])
        dec = self.pixel_decoder
        feats, cls = self.encode(x)
        feats = dec.input_adapter(feats)
        if rays is None:
            fov, sh = dec.angular_module(torch.cat(dec.camera_token_adapter(cls), 1), size)
            rays = camera_rays(fov, sh, *size)
        logr, logc = dec.radial_module(feats, rays, size, confidence=confidence)
        return logr, logc, rays
