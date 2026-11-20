"""nnFoundation runners: the pretrained encoders of nnssl's `nnFoundationCNN` (ResEnc-L) and `nnFoundationViT` (Primus).

Both checkpoints are masked-autoencoder pretrainings; only the encoder is used (the reconstruction decoder is not
loaded). Pipeline: nnssl's `ZScoreNormalization` over the whole volume (no mask, no resampling), zero padding at the
far end of every axis (CNN: to a multiple of 32; ViT: to its fixed 192^3 input), the encoder, and the feature maps
cropped to `ceil(n / stride)` of the input.

- CNN: dynamic_network_architectures' `ResidualEncoder` (`fastforks.volume._nnunet.Encoder`, the ResEnc-L plan: 6
  stages of 1, 3, 4, 6, 6, 6 basic blocks, 32..320 channels); the output of every stage (strides 1..32).
- ViT: `Primus` (patch embedding 8^3, absolute + 3D rotary positions, 40 EVA blocks of width 1056 with SwiGLU MLPs,
  LayerScale and the inner attention / MLP norms of timm's `EvaBlock`, final LayerNorm); the tokens as one map
  `[1056, D / 8, H / 8, W / 8]`. Head size 66 is padded to 72 for SDPA's flash kernels (zeros: same scores).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.checkpoint import load_strict
from fastforks._core.errors import OptionsError, WeightsError
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import override
from fastforks._core.runner import Forward
from fastforks.shared import mlx_wrap
from fastforks.shared.host_blocks import StreamedModuleList
from fastforks.volume._nnunet import Encoder, Norm, Plan, UNet
from fastforks.volume._nnunet_runner import fused_norm_act
from fastforks.volume._runner import EncoderRunner

if TYPE_CHECKING:
    from fastforks.volume._runner import Context

__all__ = ["RUNNERS", "Params", "Primus", "normalise", "res_enc_l"]

_PAD_HEAD = 8  # fixed: SDPA's flash / efficient kernels take head sizes in multiples of 8


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params`.

    Attributes:
        arch: `"cnn"` (ResEnc-L) or `"vit"` (Primus).
        stream: Stage CNN levels or ViT blocks from host memory on CUDA.
    """

    arch: str = "cnn"
    stream: bool = False


def normalise(x: torch.Tensor) -> torch.Tensor:
    """`ZScoreNormalization` of nnssl without a mask: `(x - mean) / max(std, 1e-8)` per channel (float64 stats)."""
    out = torch.empty_like(x, dtype=torch.float32)
    for c in range(x.shape[0]):
        d = x[c].cpu().double()  # MPS has no float64
        out[c] = (x[c].float() - d.mean().float().to(x.device)) / max(float(d.std(correction=0)), 1e-8)
    return out


def res_enc_l() -> Plan:
    """The ResEnc-L encoder of nnssl's `ResEncL` (dynamic_network_architectures `ResidualEncoderUNet`)."""
    k3, s2 = (3, 3, 3), (2, 2, 2)
    return Plan(
        patch_size=(192, 192, 192),
        spacing=(1.0, 1.0, 1.0),
        norms=(Norm("ZScoreNormalization", use_mask=False),),
        features=(32, 64, 128, 256, 320, 320),
        kernel_sizes=(k3,) * 6,
        strides=((1, 1, 1), s2, s2, s2, s2, s2),
        n_conv_encoder=(1, 3, 4, 6, 6, 6),
        n_conv_decoder=(1, 1, 1, 1, 1),
        labels={},
        network="resenc",
    )


def _rope(size: int, dim: int, temperature: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """`RotaryEmbeddingCat(dim, in_pixels=False)` of timm on a `size^3` grid: sin, cos `[N, 1.5 dim]`.

    The feature shape is the reference shape, so its `t / feat * ref` rescaling is the identity.
    """
    n_bands = dim // 4
    bands = 1.0 / (temperature ** (torch.arange(n_bands, dtype=torch.float32) / n_bands))
    t = torch.arange(size, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(t, t, t, indexing="ij"), dim=-1).unsqueeze(-1) * bands  # [s, s, s, 3, bands]
    sin = grid.sin().reshape(size**3, -1).repeat_interleave(2, -1)
    cos = grid.cos().reshape(size**3, -1).repeat_interleave(2, -1)
    return sin, cos


def _rotate(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """`apply_rot_embed_cat` of timm (interleaved pairs): `x cos + rot(x) sin`, `rot(x) = (-x1, x0, -x3, x2, ...)`."""
    rot = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).reshape(x.shape)
    return x * cos + rot * sin


def _ln(norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    """LayerNorm in float32 (as under the original's autocast), returned in the weights' dtype."""
    w, b = norm.weight, norm.bias
    return F.layer_norm(x.float(), norm.normalized_shape, w.float(), b.float(), norm.eps).to(w.dtype)


class _Attention(nn.Module):
    """timm's `EvaAttention` (separate q / k / v, k without bias, inner LayerNorm), no q / k norm."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        h = self.heads
        d = c // h
        q = _rotate(self.q_proj(x).view(b, n, h, d).transpose(1, 2), sin, cos).to(x.dtype)
        k = _rotate(self.k_proj(x).view(b, n, h, d).transpose(1, 2), sin, cos).to(x.dtype)
        v = self.v_proj(x).view(b, n, h, d).transpose(1, 2)
        pad = -d % _PAD_HEAD
        if pad:
            q, k, v = (F.pad(t, (0, pad)) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, scale=d**-0.5)[..., :d]
        return self.proj(_ln(self.norm, o.transpose(1, 2).reshape(b, n, c)))


class _Mlp(nn.Module):
    """timm's `SwiGLU` with the inner norm (`scale_mlp`): `fc2(norm(silu(fc1_g(x)) * fc1_x(x)))`."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1_g = nn.Linear(dim, hidden)
        self.fc1_x = nn.Linear(dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(_ln(self.norm, F.silu(self.fc1_g(x)) * self.fc1_x(x)))


class _Block(nn.Module):
    """timm's `EvaBlock` in eval mode: pre-norm attention and MLP, each scaled by its LayerScale `gamma`."""

    def __init__(self, dim: int, heads: int, hidden: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, heads)
        self.gamma_1 = nn.Parameter(torch.ones(dim))
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, hidden)
        self.gamma_2 = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        """`x`: the float32 residual stream (as under autocast); the branches run in the weights' dtype."""
        x = torch.addcmul(x, self.gamma_1.float(), self.attn(_ln(self.norm1, x), sin, cos).float())
        return torch.addcmul(x, self.gamma_2.float(), self.mlp(_ln(self.norm2, x)).float())


class _Eva(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, hidden: int, tokens: int) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, tokens, dim))
        self.blocks = nn.ModuleList(_Block(dim, heads, hidden) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)


class _PatchEmbed(nn.Module):
    def __init__(self, patch: int, dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(channels, dim, patch, patch)


class Primus(nn.Module):
    """dynamic_network_architectures' `Primus` encoder (upstream keys): `[B, C, s, s, s]` -> `[B, dim, s / p, ...]`."""

    def __init__(  # noqa: PLR0913 - the architecture
        self,
        *,
        size: int = 192,
        patch: int = 8,
        dim: int = 1056,
        depth: int = 40,
        heads: int = 16,
        hidden: int = 2816,
        channels: int = 1,
    ) -> None:
        """Build the encoder for a fixed `size^3` input (the absolute position table has one row per patch)."""
        super().__init__()
        self.size, self.grid = size, size // patch
        self.down_projection = _PatchEmbed(patch, dim, channels)
        self.eva = _Eva(dim, depth, heads, hidden, self.grid**3)
        self.rope_dim = round(dim // heads / 1.5)
        sin, cos = _rope(self.grid, self.rope_dim)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Token map of a normalised `[B, C, size, size, size]` volume."""
        t = self.down_projection.proj(x).flatten(2).transpose(1, 2)  # "b c w h d -> b (w h d) c"
        t = t.float() + self.eva.pos_embed.float()
        sin, cos = self.rope_sin.float(), self.rope_cos.float()
        for blk in self.eva.blocks:
            t = blk(t, sin, cos)
        t = _ln(self.eva.norm, t)
        g = self.grid
        return t.transpose(1, 2).reshape(x.shape[0], -1, g, g, g)


def _encoder_keys(weights: dict[str, torch.Tensor], keep: tuple[str, ...], strip: str = "") -> dict[str, torch.Tensor]:
    sd = UNet.state_dict_of(weights)
    return {k.removeprefix(strip): v for k, v in sd.items() if k.startswith(keep)}


class Portable(EncoderRunner):
    """Plain PyTorch: float16 weights / activations on CUDA (CNN: `channels_last_3d`; ViT: SDPA, padded heads)."""

    dtypes = ("float32", "float16")
    Params = Params
    fused = False

    def __init__(self, ctx: Context) -> None:
        """Build the encoder and load the encoder weights of the checkpoint."""
        super().__init__(ctx)
        arch = ctx.params.arch
        if ctx.params.stream and ctx.device.type != "cuda":
            msg = "stream=True requires CUDA for nnFoundation layer transfers"
            raise OptionsError(msg)
        weights = ctx.state_dict(0, key="network_weights", strip="")
        self.half = ctx.dtype == "float16"
        if arch == "cnn":
            net: nn.Module = Encoder(res_enc_l(), 1)
            sd = _encoder_keys(weights, ("encoder.",), "encoder.")
            self.strides, self.widths, self.multiple = (1, 2, 4, 8, 16, 32), res_enc_l().features, 32
        elif arch == "vit":
            net = Primus()
            sd = _encoder_keys(weights, ("down_projection.", "eva."))
            self.strides, self.widths, self.multiple = (8,), (1056,), 8
            self.max_size = net.size
        else:
            msg = f"{ctx.variant.ref}: unknown nnFoundation arch {arch!r}"
            raise WeightsError(msg)
        load_strict(net, sd, ctx.variant.ref)
        self.arch = arch
        net = net.eval()
        if self.half:
            net = net.half()
            if arch == "vit":  # the rotary tables stay float32 (the original rotates in float32 under autocast)
                net.rope_sin, net.rope_cos = _rope(cast("int", net.grid), cast("int", net.rope_dim))
        if arch == "cnn" and ctx.device.type == "cuda":
            net = net.to(memory_format=torch.channels_last_3d)
        self.net = self._prepare(net)

    def _prepare(self, net: nn.Module) -> nn.Module:
        """Install inference methods before the host packer moves stage shells to meta."""
        if self.fused and isinstance(net, Encoder):
            for module in net.modules():
                if hasattr(module, "norm_act"):
                    override(module, fused_norm_act, "norm_act")
        if self.ctx.params.stream:
            if isinstance(net, Encoder):
                net.stages = StreamedModuleList(net.stages, self.ctx.device)  # ty: ignore[invalid-assignment]
            else:
                eva = cast("Primus", net).eva
                eva.blocks = StreamedModuleList(eva.blocks, self.ctx.device)  # ty: ignore[invalid-assignment]
        return net.to(self.ctx.device)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """`[1, C, *padded]` of a normalised `[C, D, H, W]`: zeros at the far end (ViT: to the fixed input size)."""
        if self.arch == "vit":
            target = [self.max_size or 0] * 3
        else:
            target = [-(-n // self.multiple) * self.multiple for n in x.shape[1:]]
        pads = [v for n, t in zip(reversed(x.shape[1:]), reversed(target), strict=True) for v in (0, t - n)]
        return F.pad(x, pads)[None]

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        dt = torch.float16 if self.half else torch.float32
        if self.arch == "cnn":
            mf = torch.channels_last_3d if x.is_cuda else torch.contiguous_format
            return list(self.net(x.to(dt).contiguous(memory_format=mf)))
        return [self.net(x.to(dt))]

    def __call__(self, volume: torch.Tensor) -> dict[str, tuple[torch.Tensor, ...]]:
        """Feature maps of one `[C, D, H, W]` volume, cropped to the input."""
        x = self._pad(normalise(volume))
        maps = self._encode(x)
        del x
        if len(maps) != len(self.strides):
            msg = f"expected {len(self.strides)} nnFoundation feature maps, got {len(maps)}"
            raise ValueError(msg)
        maps.reverse()
        shape = volume.shape[1:]
        out = []
        for s in self.strides:
            t = maps.pop()
            crop = tuple(slice(0, -(-n // s)) for n in shape)
            cropped = t[(slice(None), slice(None), *crop)]
            fmt = torch.preserve_format
            if t.dtype != torch.float32 and t.is_contiguous(memory_format=torch.channels_last_3d):
                fmt = torch.channels_last_3d
            out.append(cropped.to(dtype=torch.float32, memory_format=fmt)[0])
            del cropped, t
        return {"features": tuple(out)}

    def module(self) -> nn.Module:
        """The encoder: a normalised, padded `[1, C, D, H, W]` volume -> the feature maps (CNN: a tuple of six)."""
        if self.ctx.params.stream:
            msg = "Export requires resident weights; reload with stream=False"
            raise NotImplementedError(msg)
        return Forward(self.net, lambda _net, x: tuple(self._encode(x)))


class Kernels(Portable):
    """Portable + CNN: InstanceNorm + LeakyReLU as one in-place kernel (`vol_norm`); ViT: fp16-accumulate GEMMs."""

    fused = True

    def __init__(self, ctx: Context) -> None:
        """Load the kernels first, then the encoder."""
        if ctx.params.arch == "cnn":
            from fastforks.kernels import vol_norm  # noqa: PLC0415 - kernels only

            vol_norm.load(ctx.device)
        super().__init__(ctx)

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self.arch == "vit":
            with fp16_accumulate():
                return super()._encode(x)
        return super()._encode(x)


RUNNERS = {"portable": Portable, "kernels": Kernels, "portable_mlx": mlx_wrap.port(Portable)}
