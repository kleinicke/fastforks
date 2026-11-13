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
# fastforks: upstream `tapnet/tapnext/tapnext_torch.py` (+ `restore_model_from_jax_checkpoint` of
# tapnext_torch_utils.py as `from_jax`), time-major for one video (`[T, N, C]` tokens), einops removed, the RG-LRU scan
# and the temporal conv carrying an explicit state so a video runs in chunks of frames, SDPA attention. Parameter names
# as upstream (its state dicts load unchanged).
"""TAPNext (TRecViT-B point tracker), time-major: one video, `[T, N, C]` tokens."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from fastforks._core.checkpoint import mismatch_details
from fastforks._core.errors import CheckpointMismatchError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

WIDTH, HEADS, DEPTH, PATCH, SIZE = 768, 12, 12, 8, 256
CONV_WIDTH = 4


def posemb_sincos_2d(h: int, w: int, width: int, temperature: float = 10_000.0) -> torch.Tensor:
    """MoCo v3 2D sin-cos embedding `[1, h * w, width]` (float64 math, float32 result, as upstream's numpy)."""
    y, x = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij")
    omega = 1.0 / temperature ** (torch.arange(width // 4, dtype=torch.float64) / (width // 4 - 1))
    y, x = y.flatten()[:, None] * omega, x.flatten()[:, None] * omega
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], 1).float()[None]


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6) * (self.scale.float() + 1)).to(x.dtype)


class BlockDiagonalLinear(nn.Module):
    def __init__(self, width: int, blocks: int) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(blocks, width // blocks, width // blocks))
        self.b = nn.Parameter(torch.zeros(blocks, width // blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, i, _ = self.w.shape
        y = torch.einsum("...hi,hij->...hj", x.unflatten(-1, (h, i)), self.w) + self.b
        return y.flatten(-2)


class RGLRU(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.a_param = nn.Parameter(torch.zeros(width))
        self.input_gate = BlockDiagonalLinear(width, heads)
        self.a_gate = BlockDiagonalLinear(width, heads)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """`x [T, N, C]` (float32), `h [N, C]` float32 state or `None` at the video start."""
        gate_x = torch.sigmoid(self.input_gate(x))
        log_a = -8.0 * torch.sigmoid(self.a_gate(x)) * F.softplus(self.a_param.float())
        a = log_a.exp()
        mult = torch.sqrt(1 - (2 * log_a).exp())
        if h is None:  # reset at the first frame
            mult = torch.cat([torch.ones_like(mult[:1]), mult[1:]])
            h = torch.zeros_like(x[0], dtype=torch.float32)
        xs = (x * gate_x * mult).float()
        ys = []
        for t in range(x.shape[0]):
            h = a[t] * h + xs[t]
            ys.append(h)
        return torch.stack(ys), h


class RecurrentBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.linear_y = nn.Linear(width, width)
        self.linear_x = nn.Linear(width, width)
        self.linear_out = nn.Linear(width, width)
        self.conv_1d = nn.Module()
        self.conv_1d.w = nn.Parameter(torch.zeros(CONV_WIDTH, width))
        self.conv_1d.b = nn.Parameter(torch.zeros(width))
        self.rg_lru = RGLRU(width, heads)

    def forward(self, x: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor] | None):
        y = F.gelu(self.linear_y(x), approximate="tanh")
        return self.temporal(self.linear_x(x).float(), cache, y, self.linear_out)

    def temporal(self, x: torch.Tensor, cache, y: torch.Tensor, out: Callable[[torch.Tensor], torch.Tensor]):
        """The temporal conv and the RG-LRU scan of the float32 `x` branch, gated by `y`, through `out`."""
        conv_state = x.new_zeros(CONV_WIDTH - 1, *x.shape[1:]) if cache is None else cache[1]
        xp = torch.cat([conv_state, x])  # [T + 3, N, C]
        t = x.shape[0]
        w = self.conv_1d.w.float()
        x = self.conv_1d.b.float() + sum(w[k] * xp[k : k + t] for k in range(CONV_WIDTH))
        x, h = self.rg_lru(x, None if cache is None else cache[0])
        x = out((x * y).to(y.dtype))
        return x, (h, xp[-(CONV_WIDTH - 1) :])


class MLPBlock(nn.Module):
    def __init__(self, width: int, expanded: int) -> None:
        super().__init__()
        self.ffw_up = nn.Module()
        self.ffw_up.w = nn.Parameter(torch.zeros(2, width, expanded))
        self.ffw_up.b = nn.Parameter(torch.zeros(2, 1, 1, expanded))
        self.ffw_down = nn.Linear(expanded, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, b = self.ffw_up.w, self.ffw_up.b
        gate = F.gelu(x @ w[0] + b[0, 0], approximate="tanh")
        return self.ffw_down(gate * (x @ w[1] + b[1, 0]))


class ResidualBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.temporal_pre_norm = RMSNorm(width)
        self.recurrent_block = RecurrentBlock(width, heads)
        self.channel_pre_norm = RMSNorm(width)
        self.mlp_block = MLPBlock(width, 4 * width)

    def forward(self, x: torch.Tensor, cache):
        y, cache = self.recurrent_block(self.temporal_pre_norm(x), cache)
        x = x + y
        return x + self.mlp_block(self.channel_pre_norm(x)), cache


class Attention(nn.Module):
    """`nn.MultiheadAttention` parameters (packed in-projection), SDPA forward."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.in_proj_weight = nn.Parameter(torch.zeros(3 * width, width))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * width))
        self.out_proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).view(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        return self.out_proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c))


class EncoderBlock(nn.Module):
    """torchvision's ViT `EncoderBlock` (LayerNorm eps 1e-6, exact GELU)."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(width, eps=1e-6)
        self.self_attention = Attention(width, heads)
        self.ln_2 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Identity(), nn.Linear(4 * width, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class TRecViTBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.ssm_block = ResidualBlock(width, heads)
        self.vit_block = EncoderBlock(width, heads)

    def forward(self, x: torch.Tensor, cache):
        x, cache = self.ssm_block(x, cache)  # over time, per token
        return self.vit_block(x), cache  # over tokens, per frame


def _head(width: int, out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(width, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
        nn.Linear(256, out),
    )


class TAPNext(nn.Module):
    """Parameter names as upstream's torch `TAPNext` (its state dicts load as they are)."""

    def __init__(self) -> None:
        super().__init__()
        g = SIZE // PATCH
        self.lin_proj = nn.Conv2d(3, WIDTH, PATCH, PATCH)
        self.blocks = nn.ModuleList(TRecViTBlock(WIDTH, HEADS) for _ in range(DEPTH))
        self.encoder_norm = nn.LayerNorm(WIDTH)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, WIDTH))
        self.unknown_token = nn.Parameter(torch.zeros(1, 1, WIDTH))
        self.point_query_token = nn.Parameter(torch.zeros(1, 1, 1, WIDTH))
        self.image_pos_emb = nn.Parameter(torch.zeros(1, g * g, WIDTH))
        self.register_buffer("query_pos_embed", posemb_sincos_2d(SIZE, SIZE, WIDTH))
        self.visible_head = _head(WIDTH, 1)
        self.coordinate_head = _head(WIDTH, 2 * SIZE)

    def query_embedding(self, pos: torch.Tensor) -> torch.Tensor:
        """Point-query tokens `[Q, C]` of query positions `[Q, 2]` `(y, x)` in model pixels."""
        table = self.query_pos_embed.view(1, SIZE, SIZE, WIDTH).permute(0, 3, 2, 1)  # [1, C, x, y]
        grid = (pos / SIZE * 2 - 1).to(table.dtype)[None, None]  # (y, x): y indexes the last dim
        emb = F.grid_sample(table, grid, align_corners=False)[0, :, 0].T  # [Q, C]
        return (self.point_query_token[0, 0] + emb).to(self.mask_token.dtype)

    def query_tokens(self, t0: int, t: int, qt: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        """`[t, Q, C]` tokens of frames `t0 .. t0 + t - 1` for queries on frames `qt [Q]` with tokens `point`."""
        frames = torch.arange(t0, t0 + t, device=qt.device, dtype=qt.dtype)[:, None, None]
        qt = qt[None, :, None]
        tok = torch.where(frames == qt, point[None], self.mask_token[0])
        return torch.where(frames < qt, self.unknown_token[0], tok)

    def forward(self, video: torch.Tensor, qt: torch.Tensor, point: torch.Tensor, t0: int, state: list | None):
        """`video [T, 3, 256, 256]` in [-1, 1], frames `t0 ..`; returns `(y, x)` tracks `[T, Q, 2]`, visible logits
        `[T, Q]`, track logits `[T, Q, 512]` and the new state."""
        t = video.shape[0]
        x = self.lin_proj(video).flatten(2).transpose(1, 2) + self.image_pos_emb  # [T, 1024, C]
        n = x.shape[1]
        x = torch.cat([x, self.query_tokens(t0, t, qt, point).to(x.dtype)], 1)
        new = []
        for blk, cache in zip(self.blocks, state or [None] * len(self.blocks), strict=True):
            x, c = blk(x, cache)
            new.append(c)
        p = self.encoder_norm(x[:, n:])
        logits = self.coordinate_head(p)
        return soft_argmax(logits.float()), self.visible_head(p)[..., 0].float(), logits, new


def soft_argmax(logits: torch.Tensor, radius: int = 20, temperature: float = 0.5) -> torch.Tensor:
    """Upstream's windowed soft-argmax over the two 256-bin halves; `(y, x)` + 0.5 in model pixels."""
    pos = logits.unflatten(-1, (2, -1))  # [..., 2, 256]
    idx = torch.arange(pos.shape[-1], device=logits.device, dtype=logits.dtype)
    near = (idx - pos.argmax(-1, keepdim=True)).abs() <= radius
    p = F.softmax(pos * temperature, -1) * near
    return (p * idx).sum(-1) / p.sum(-1) + 0.5


class _Read(dict):
    """The JAX checkpoint; records the keys read and names a missing one as a checkpoint mismatch."""

    def __init__(self, ckpt: Mapping[str, torch.Tensor]) -> None:
        super().__init__(ckpt)
        self.used: set[str] = set()

    def __getitem__(self, k: str) -> torch.Tensor:
        if k not in self:
            raise CheckpointMismatchError("checkpoint", mismatch_details([k], [], []))
        self.used.add(k)
        return super().__getitem__(k)


def from_jax(jax: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Upstream's `restore_model_from_jax_checkpoint` as a state-dict mapping; keys it does not read are kept as they
    are (the strict load reports them as unexpected)."""
    ckpt = _Read(jax)
    t = lambda a: a.clone()  # noqa: E731
    sd = {
        "lin_proj.weight": t(ckpt["backbone/embedding/kernel"][0]).permute(3, 2, 0, 1),
        "lin_proj.bias": t(ckpt["backbone/embedding/bias"]),
        "mask_token": t(ckpt["backbone/mask_token"]),
        "point_query_token": t(ckpt["backbone/point_query_token"]),
        "unknown_token": t(ckpt["backbone/unknown_token"]),
        "image_pos_emb": t(ckpt["backbone/pos_embedding"]),
        "encoder_norm.weight": t(ckpt["backbone/Transformer/encoder_norm/scale"]),
        "encoder_norm.bias": t(ckpt["backbone/Transformer/encoder_norm/bias"]),
        "query_pos_embed": posemb_sincos_2d(SIZE, SIZE, WIDTH),
    }
    ssm = (
        "temporal_pre_norm.scale", "channel_pre_norm.scale", "recurrent_block.linear_y.weight",
        "recurrent_block.linear_y.bias", "recurrent_block.linear_x.weight", "recurrent_block.linear_x.bias",
        "recurrent_block.linear_out.weight", "recurrent_block.linear_out.bias", "recurrent_block.conv_1d.w",
        "recurrent_block.conv_1d.b", "recurrent_block.rg_lru.a_param", "recurrent_block.rg_lru.input_gate.w",
        "recurrent_block.rg_lru.input_gate.b", "recurrent_block.rg_lru.a_gate.w", "recurrent_block.rg_lru.a_gate.b",
        "mlp_block.ffw_up.w", "mlp_block.ffw_up.b", "mlp_block.ffw_down.weight", "mlp_block.ffw_down.bias",
    )
    for i in range(DEPTH):
        pre = f"backbone/Transformer/encoderblock_{i}"
        for k in ssm:
            a = t(ckpt[f"{pre}/ssm_block/" + k.replace(".", "/").replace("weight", "kernel")])
            sd[f"blocks.{i}.ssm_block.{k}"] = a.T if "weight" in k else a
        v = f"{pre}/vit_block/"
        att = v + "MultiHeadDotProductAttention_0/"
        b = f"blocks.{i}.vit_block."
        sd[b + "ln_1.weight"], sd[b + "ln_1.bias"] = t(ckpt[v + "LayerNorm_0/scale"]), t(ckpt[v + "LayerNorm_0/bias"])
        sd[b + "ln_2.weight"], sd[b + "ln_2.bias"] = t(ckpt[v + "LayerNorm_1/scale"]), t(ckpt[v + "LayerNorm_1/bias"])
        for j, d in ((0, 0), (3, 1)):
            sd[b + f"mlp.{j}.weight"] = t(ckpt[v + f"MlpBlock_0/Dense_{d}/kernel"]).T
            sd[b + f"mlp.{j}.bias"] = t(ckpt[v + f"MlpBlock_0/Dense_{d}/bias"])
        sd[b + "self_attention.in_proj_weight"] = torch.cat(
            [t(ckpt[att + f"{n}/kernel"]).reshape(WIDTH, WIDTH).T for n in ("query", "key", "value")]
        )
        sd[b + "self_attention.in_proj_bias"] = torch.cat(
            [t(ckpt[att + f"{n}/bias"]).flatten() for n in ("query", "key", "value")]
        )
        sd[b + "self_attention.out_proj.weight"] = t(ckpt[att + "out/kernel"]).reshape(WIDTH, WIDTH).T
        sd[b + "self_attention.out_proj.bias"] = t(ckpt[att + "out/bias"]).flatten()
    for head in ("visible_head", "coordinate_head"):
        for j in (0, 3, 6):
            sd[f"{head}.{j}.weight"] = t(ckpt[f"{head}/layers_{j}/kernel"]).T
            sd[f"{head}.{j}.bias"] = t(ckpt[f"{head}/layers_{j}/bias"])
        for j in (1, 4):
            sd[f"{head}.{j}.weight"] = t(ckpt[f"{head}/layers_{j}/scale"])
            sd[f"{head}.{j}.bias"] = t(ckpt[f"{head}/layers_{j}/bias"])
    return sd | {k: v for k, v in jax.items() if k not in ckpt.used}
