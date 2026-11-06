"""DINOv3 ViT in plain PyTorch, with the parameter names of the Hugging Face checkpoints (`DINOv3ViTModel`).

Same maths as `transformers` `DINOv3ViTModel` in eval: patch 16 conv, then [cls, 4 registers, patches] (no absolute
position embedding); pre-norm blocks with LayerScale; attention with RoPE on the q / k of the patch tokens only (theta
100, patch centres in [-1, 1], `rotate_half`); GELU MLP, or the gated SiLU MLP (ViT-H+); final LayerNorm.

fp16 range: the residual stream of ViT-L / H+ reaches 1e5 (> 65504; HF in fp16 gives NaN), ViT-B 1.5e4. `stream_scale`
divides the whole stream by a power of two through the embedding and LayerScale weights (LayerNorm eps / scale^2):
exact up to the eps rounding, no extra op at run time. `hidden_states` are multiplied back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.checkpoint import load_strict

PATCH = 16
REGISTERS = 4
_ROPE_SIZES = 8  # fixed: cache policy, RoPE tables kept (one per patch grid)


@dataclass(frozen=True)
class Config:
    """The `config.json` entries that differ between the sizes."""

    hidden_size: int
    num_heads: int
    num_layers: int
    intermediate_size: int
    gated: bool  # gated SiLU MLP (ViT-H+) instead of GELU
    rope_theta: float = 100.0
    eps: float = 1e-5


CONFIGS = {
    "vits16": Config(384, 6, 12, 1536, gated=False),
    "vitb16": Config(768, 12, 12, 3072, gated=False),
    "vitl16": Config(1024, 16, 24, 4096, gated=False),
    "vith16plus": Config(1280, 20, 32, 5120, gated=True),
}


class _Embeddings(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.empty(1, 1, c))
        self.mask_token = nn.Parameter(torch.empty(1, 1, c))  # pre-training only (kept: strict loading)
        self.register_tokens = nn.Parameter(torch.empty(1, REGISTERS, c))
        self.patch_embeddings = nn.Conv2d(3, c, PATCH, PATCH)


class _Attention(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.q_proj, self.k_proj = nn.Linear(c, c), nn.Linear(c, c, bias=False)
        self.v_proj, self.o_proj = nn.Linear(c, c), nn.Linear(c, c)


class _Scale(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(torch.empty(c))


class _Mlp(nn.Module):
    def __init__(self, c: int, hidden: int, *, gated: bool) -> None:
        super().__init__()
        self.up_proj, self.down_proj = nn.Linear(c, hidden), nn.Linear(hidden, c)
        self.gate_proj = nn.Linear(c, hidden) if gated else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_proj is not None:
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return self.down_proj(F.gelu(self.up_proj(x)))


class _Layer(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        c = cfg.hidden_size
        self.norm1, self.norm2 = nn.LayerNorm(c, eps=cfg.eps), nn.LayerNorm(c, eps=cfg.eps)
        self.attention = _Attention(c)
        self.layer_scale1, self.layer_scale2 = _Scale(c), _Scale(c)
        self.mlp = _Mlp(c, cfg.intermediate_size, gated=cfg.gated)


def rope_table(h: int, w: int, head_dim: int, theta: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 `(cos, sin)` `[h * w, head_dim]` of an `h x w` patch grid, as HF `DINOv3ViTRopePositionEmbedding` (eval)."""
    ch = torch.arange(0.5, h, dtype=torch.float32, device=device) / h
    cw = torch.arange(0.5, w, dtype=torch.float32, device=device) / w
    coords = 2.0 * torch.stack(torch.meshgrid(ch, cw, indexing="ij"), -1).flatten(0, 1) - 1.0
    inv = 1 / theta ** torch.arange(0, 1, 4 / head_dim, dtype=torch.float32, device=device)
    ang = (2 * math.pi * coords[:, :, None] * inv[None, None, :]).flatten(1, 2).tile(2)
    return ang.cos(), ang.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """`[x1, x2] -> [-x2, x1]` over the last dimension."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class DINOv3ViT(nn.Module):
    """`forward(x)`: normalised `[B, 3, H, W]` (multiples of 16) -> `last_hidden_state`; `run` also hidden states."""

    def __init__(self, cfg: Config) -> None:
        """Uninitialised network (load a checkpoint with `load`)."""
        super().__init__()
        self.cfg = cfg
        self.embeddings = _Embeddings(cfg.hidden_size)
        self.layer = nn.ModuleList(_Layer(cfg) for _ in range(cfg.num_layers))
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.eps)
        self.stream_scale = 1.0
        self._rope: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def load(
        self, sd: dict[str, torch.Tensor], stream_scale: float = 1.0, what: str = "DINOv3 checkpoint"
    ) -> DINOv3ViT:
        """Assign the HF state dict strictly, folding `stream_scale` (a power of two) into the stream weights.

        Raises:
            WeightsError: The state dict does not fit (names `what` and the first keys).
        """
        s = float(stream_scale)
        if s != 1.0:
            folded = ("embeddings.cls_token", "embeddings.register_tokens", "embeddings.patch_embeddings.")
            sd = {k: v / s if k.startswith(folded) or k.endswith(".lambda1") else v for k, v in sd.items()}
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eps = self.cfg.eps / s**2
        load_strict(self, sd, what, assign=True)
        self.stream_scale = s
        return self

    def blocks(self) -> list[_Layer]:
        """The transformer blocks (typed view of `layer`)."""
        return [cast("_Layer", b) for b in self.layer]

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """`[cls, registers, patches]` (HF `hidden_states[0]`, divided by the stream scale)."""
        e = self.embeddings
        p = e.patch_embeddings(x).flatten(2).transpose(1, 2)
        b = p.shape[0]
        return torch.cat([e.cls_token.expand(b, -1, -1), e.register_tokens.expand(b, -1, -1), p], 1)

    def pack_qkv(self) -> DINOv3ViT:
        """Replace the q / k / v projections of each block by one `[3C, C]` GEMM (zero k bias): fewer launches."""
        for blk in self.blocks():
            a = blk.attention
            w = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]).detach()
            b = torch.cat([a.q_proj.bias, torch.zeros_like(a.q_proj.bias), a.v_proj.bias]).detach()
            a.register_buffer("qkv_weight", w, persistent=False)
            a.register_buffer("qkv_bias", b, persistent=False)
            del a.q_proj, a.k_proj, a.v_proj
        return self

    def attention(self, blk: _Layer, y: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """RoPE SDPA attention + output projection of the normed `y`; its temporaries are freed on return."""
        b, n, c = y.shape
        h = self.cfg.num_heads
        a = blk.attention
        if hasattr(a, "qkv_weight"):
            w, bias = cast("torch.Tensor", a.qkv_weight), cast("torch.Tensor", a.qkv_bias)
            qkv = F.linear(y, w, bias).view(b, n, 3, h, c // h)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        else:
            q, k, v = (p(y).view(b, n, h, c // h).transpose(1, 2) for p in (a.q_proj, a.k_proj, a.v_proj))
        cos, sin = rope
        np_ = cos.shape[0]
        q = torch.cat([q[:, :, :-np_], q[:, :, -np_:] * cos + rotate_half(q[:, :, -np_:]) * sin], 2)
        k = torch.cat([k[:, :, :-np_], k[:, :, -np_:] * cos + rotate_half(k[:, :, -np_:]) * sin], 2)
        return a.o_proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c))

    def block(self, blk: _Layer, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """One pre-norm block (SDPA attention); `rope` = `(cos, sin)` in the dtype of `x`."""
        if torch.onnx.is_in_onnx_export():  # ONNX addcmul promotes to fp32: plain ops for the export
            x = x + self.attention(blk, blk.norm1(x), rope) * blk.layer_scale1.lambda1
            return x + blk.mlp(blk.norm2(x)) * blk.layer_scale2.lambda1
        x = torch.addcmul(x, self.attention(blk, blk.norm1(x), rope), blk.layer_scale1.lambda1)
        return torch.addcmul(x, blk.mlp(blk.norm2(x)), blk.layer_scale2.lambda1)

    def rope(self, h: int, w: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """`(cos, sin)` of an `h x w` patch grid in `dtype`; in inference mode kept for the last `_ROPE_SIZES` grids."""
        cache = not torch.compiler.is_compiling() and torch.is_inference_mode_enabled()  # not in traces / autograd
        key = (h, w, dtype, device)
        if cache and key in self._rope:
            return self._rope[key]
        cfg = self.cfg
        cos, sin = rope_table(h, w, cfg.hidden_size // cfg.num_heads, cfg.rope_theta, device)
        cs = cos.to(dtype), sin.to(dtype)
        if cache:
            if len(self._rope) >= _ROPE_SIZES:
                del self._rope[next(iter(self._rope))]
            self._rope[key] = cs
        return cs

    def run(self, x: torch.Tensor, layers: tuple[int, ...] = ()) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """(`last_hidden_state`, the HF `hidden_states` at `layers` (float32, un-normed, stream scale undone))."""
        t = self.tokens(x)
        rope = self.rope(x.shape[-2] // PATCH, x.shape[-1] // PATCH, t.dtype, x.device)
        hs = [t.float() * self.stream_scale] if 0 in layers else []  # fp32: ViT-L / H+ unscaled exceed fp16
        for i, blk in enumerate(self.blocks(), 1):
            t = self.block(blk, t, rope)
            if i in layers:
                hs.append(t.float() * self.stream_scale)
        return self.norm(t), hs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`last_hidden_state` `[B, 5 + h * w, C]`."""
        return self.run(x)[0]
