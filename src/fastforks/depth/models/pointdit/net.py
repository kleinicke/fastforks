"""PointDiT network (upstream `model.py` + `denoiser.generate`), with the per-call constants folded at load time.

Upstream conditions a DiT with adaLN on the timestep only (no class), and the timesteps of the sampler are fixed
(`linspace(0, 1, steps + 1)`, the ODE starts from zeros): every adaLN modulation is a constant per step and block.
They are computed once in float32 and folded:
- `modulate(norm(x), shift, scale)` -> `rms(x) * (g * (1 + scale))`, the shift folded into the bias of the next linear;
- the final layer's `g * (1 + scale)` and shift folded into its weight and bias (one small weight per step);
- the gates are per-channel vectors (the kernels GEMM epilogue's `scale`); the adaLN weights are not kept.
The input side is linear up to the transformer: `concat_proj([dino + pos_y, proj2(proj1(z)) + pos])` is
`dino @ Wy^T + patches(z) @ (Wx W2 W1)^T + K`, with `K` = the (interpolated) position terms and the biases; `Wy` runs
once per image and the point map stays in token layout `[B, N, (p, q, c)]` between steps (step 0: z = 0).

RoPE: upstream rotates interleaved pairs (2i, 2i + 1) of each head (pairs 0-15: the row angle, 16-31: the column
angle). The q / k rows of `qkv` (and the q / k RMSNorm weights) are permuted to the rotate-half layout (dims j and
j + 32 share an angle), which leaves q k^T unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

PATCH = 16  # fixed: patch size of every released model
GRID = 32  # fixed: the position-embedding grid (512 px training)
HEAD = 64  # fixed: head width
EPS = 1e-6  # fixed: upstream RMSNorm eps
LN_EPS = 1e-5  # fixed: the DINOv3 final LayerNorm eps ("layernormbf16")
FREQ = 256  # fixed: timestep frequency embedding size
THETA = 10000.0  # fixed: RoPE and timestep base


@dataclass(frozen=True)
class Config:
    """One PointDiT size."""

    depth: int
    hidden: int
    heads: int
    enc_depth: int  # DINOv3 blocks
    enc_hidden: int


CONFIGS = {
    "b": Config(12, 768, 12, 12, 768),  # PointDiT-B/16 on DINOv3 ViT-B/16
    "l": Config(24, 1024, 16, 24, 1024),  # PointDiT-L/16 on DINOv3 ViT-L/16
}


def encoder_layers(cfg: Config) -> tuple[int, ...]:
    """The HF `hidden_states` indices of upstream's 4 equally spaced DINOv3 blocks (`int((i + 1) * depth / 4) - 1`)."""
    return tuple(int((i + 1) * cfg.enc_depth / 4) for i in range(4))


def timesteps(steps: int) -> list[tuple[float, float]]:
    """Per Euler step `(t_next - t, 1 - t)` as upstream's float32 `linspace` gives them."""
    t = torch.linspace(0.0, 1.0, steps + 1)
    return [(float(t[i + 1] - t[i]), float((1.0 - t[i]).clamp_min(0.0))) for i in range(steps)]


def _timestep_embedding(t: torch.Tensor) -> torch.Tensor:
    half = FREQ // 2
    freqs = torch.exp(-math.log(THETA) * torch.arange(half, dtype=torch.float32) / half).to(t.device)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], -1)


def checkpoint_shapes(cfg: Config) -> dict[str, tuple[int, ...]]:
    """Upstream `model_ema1` keys (without `net.`) and shapes of `cfg` (DINOv3 not included: "nodinov3")."""
    c, cy, f = cfg.hidden, 4 * cfg.enc_hidden, int(4 * cfg.hidden * 2 / 3)
    out: dict[str, tuple[int, ...]] = {
        "pos_embed": (1, GRID**2, c), "pos_embed_y": (1, GRID**2, cy),
        "t_embedder.mlp.0.weight": (c, FREQ), "t_embedder.mlp.0.bias": (c,),
        "t_embedder.mlp.2.weight": (c, c), "t_embedder.mlp.2.bias": (c,),
        "x_embedder.proj1.weight": (128, 3, PATCH, PATCH), "x_embedder.proj2.weight": (c, 128, 1, 1),
        "x_embedder.proj2.bias": (c,), "concat_proj.weight": (c, c + cy), "concat_proj.bias": (c,),
        "final_layer.norm_final.weight": (c,), "final_layer.linear.weight": (PATCH**2 * 3, c),
        "final_layer.linear.bias": (PATCH**2 * 3,), "final_layer.adaLN_modulation.1.weight": (2 * c, c),
        "final_layer.adaLN_modulation.1.bias": (2 * c,),
    }  # fmt: skip
    for i in range(cfg.depth):
        p = f"blocks.{i}."
        out |= {
            p + "norm1.weight": (c,), p + "attn.q_norm.weight": (HEAD,), p + "attn.k_norm.weight": (HEAD,),
            p + "attn.qkv.weight": (3 * c, c), p + "attn.qkv.bias": (3 * c,), p + "attn.proj.weight": (c, c),
            p + "attn.proj.bias": (c,), p + "norm2.weight": (c,), p + "mlp.w12.weight": (2 * f, c),
            p + "mlp.w12.bias": (2 * f,), p + "mlp.w3.weight": (c, f), p + "mlp.w3.bias": (c,),
            p + "adaLN_modulation.1.weight": (6 * c, c), p + "adaLN_modulation.1.bias": (6 * c,),
        }  # fmt: skip
    return out


def _half_perm(heads: int) -> torch.Tensor:
    """Row order of one q (or k) projection: per head, interleaved pairs -> rotate-half layout."""
    d = torch.arange(HEAD)
    head = torch.cat([d[0::2], d[1::2]])  # new dim j <- old 2j, new dim j + 32 <- old 2j + 1
    return (torch.arange(heads)[:, None] * HEAD + head[None]).reshape(-1)


def rope_angles(h: int, w: int, device: torch.device) -> torch.Tensor:
    """`[h * w, 32]` float32 angles of the rotate-half layout: 16 row then 16 column frequencies."""
    quarter = HEAD // 4
    freqs = 1.0 / (THETA ** (torch.arange(0, HEAD // 2, 2, device=device)[:quarter].float() / (HEAD // 2)))
    fh = torch.arange(h, device=device).float()[:, None] * freqs  # [h, 16]
    fw = torch.arange(w, device=device).float()[:, None] * freqs  # [w, 16]
    return torch.cat([fh[:, None].expand(-1, w, -1), fw[None].expand(h, -1, -1)], -1).reshape(h * w, -1)


class Block(nn.Module):
    """One block's weights; `steps` x per-channel constants (float32)."""

    qkv_w: torch.Tensor
    proj_w: torch.Tensor
    proj_b: torch.Tensor
    w12: torch.Tensor
    w3: torch.Tensor
    b3: torch.Tensor
    qk_norm: torch.Tensor
    a1: torch.Tensor
    qkv_b: torch.Tensor
    g1: torch.Tensor
    a2: torch.Tensor
    b12: torch.Tensor
    g2: torch.Tensor
    g1h: torch.Tensor
    g2h: torch.Tensor

    def __init__(self, sd: dict[str, torch.Tensor], i: int, cfg: Config, mods: torch.Tensor) -> None:
        """`mods`: `[steps, 6 * C]` float32 adaLN outputs of this block."""
        super().__init__()
        c = cfg.hidden
        p = f"blocks.{i}."
        perm = _half_perm(cfg.heads).to(mods.device)
        rows = torch.cat([perm, perm + c, torch.arange(2 * c, 3 * c, device=mods.device)])
        qkv_w = sd[p + "attn.qkv.weight"].float()[rows]
        qkv_b = sd[p + "attn.qkv.bias"].float()[rows]
        w12, b12 = sd[p + "mlp.w12.weight"].float(), sd[p + "mlp.w12.bias"].float()
        sh1, sc1, g1, sh2, sc2, g2 = mods.chunk(6, -1)
        head = torch.cat([torch.arange(0, HEAD, 2), torch.arange(1, HEAD, 2)]).to(mods.device)
        self.register_buffer("qkv_w", qkv_w)
        self.register_buffer("proj_w", sd[p + "attn.proj.weight"].float())
        self.register_buffer("proj_b", sd[p + "attn.proj.bias"].float())
        self.register_buffer("w12", w12)
        self.register_buffer("w3", sd[p + "mlp.w3.weight"].float())
        self.register_buffer("b3", sd[p + "mlp.w3.bias"].float())
        qk = torch.stack([sd[p + "attn.q_norm.weight"], sd[p + "attn.k_norm.weight"]]).float()[:, head]
        self.register_buffer("qk_norm", qk[:, None])  # [2, 1, 64]: broadcasts over heads
        # per step: norm scales g * (1 + scale), the shifts through the next linear, gates
        self.register_buffer("a1", sd[p + "norm1.weight"].float() * (1 + sc1))
        self.register_buffer("qkv_b", qkv_b + sh1 @ qkv_w.T)
        self.register_buffer("g1", g1.contiguous())  # float32: the kernels epilogue scale
        self.register_buffer("a2", sd[p + "norm2.weight"].float() * (1 + sc2))
        self.register_buffer("b12", b12 + sh2 @ w12.T)
        self.register_buffer("g2", g2.contiguous())


class PointDiT(nn.Module):
    """The folded network (float32 buffers; `half()` for fp16). See the module docstring."""

    final_w: torch.Tensor
    final_b: torch.Tensor
    wy: torch.Tensor
    wp: torch.Tensor
    pos: torch.Tensor
    const: torch.Tensor
    ln_w: torch.Tensor
    ln_b: torch.Tensor

    def __init__(
        self, sd: dict[str, torch.Tensor], ln: tuple[torch.Tensor, torch.Tensor], cfg: Config, steps: int
    ) -> None:
        """`sd`: upstream `model_ema1` without `net.` (float32, on the target device); `ln`: DINOv3 final LayerNorm."""
        super().__init__()
        self.cfg, self.schedule = cfg, timesteps(steps)
        dev = sd["concat_proj.weight"].device
        c, cy = cfg.hidden, 4 * cfg.enc_hidden
        t = torch.linspace(0.0, 1.0, steps + 1, device=dev)[:-1]
        emb = sd["t_embedder.mlp.0.weight"].float()
        temb = F.linear(_timestep_embedding(t), emb, sd["t_embedder.mlp.0.bias"].float())
        temb = F.linear(F.silu(temb), sd["t_embedder.mlp.2.weight"].float(), sd["t_embedder.mlp.2.bias"].float())
        cond = F.silu(temb)  # [steps, C]: SiLU of the adaLN input
        layers = [
            Block(
                sd,
                i,
                cfg,
                F.linear(
                    cond, sd[f"blocks.{i}.adaLN_modulation.1.weight"].float(), sd[f"blocks.{i}.adaLN_modulation.1.bias"]
                ),
            )
            for i in range(cfg.depth)
        ]
        self.blocks = nn.ModuleList(layers)  # registered (state_dict, .to)
        self.layers = layers  # the same modules, typed
        f = "final_layer."
        shift, scale = F.linear(
            cond, sd[f + "adaLN_modulation.1.weight"].float(), sd[f + "adaLN_modulation.1.bias"]
        ).chunk(2, -1)
        lw, lb = sd[f + "linear.weight"].float(), sd[f + "linear.bias"].float()
        self.register_buffer("final_w", lw[None] * (sd[f + "norm_final.weight"].float() * (1 + scale))[:, None])
        self.register_buffer("final_b", lb + shift @ lw.T)
        # input side: Wy, Wp = Wx W2 W1 (columns in the (p, q, c) order of the output tokens), position grid, constant
        wc = sd["concat_proj.weight"].float()
        wy, wx = wc[:, :cy], wc[:, cy:]
        w1 = (
            sd["x_embedder.proj1.weight"]
            .float()
            .permute(0, 2, 3, 1)
            .reshape(sd["x_embedder.proj1.weight"].shape[0], -1)
        )
        w2 = sd["x_embedder.proj2.weight"].float()[:, :, 0, 0]
        self.register_buffer("wy", wy.contiguous())
        self.register_buffer("wp", (wx @ w2 @ w1).contiguous())
        g = sd["pos_embed_y"].float()[0] @ wy.T + sd["pos_embed"].float()[0] @ wx.T  # [32 * 32, C]
        self.register_buffer("pos", g.T.reshape(1, c, GRID, GRID).contiguous())
        self.register_buffer("const", wx @ sd["x_embedder.proj2.bias"].float() + sd["concat_proj.bias"].float())
        self.register_buffer("ln_w", ln[0].float())
        self.register_buffer("ln_b", ln[1].float())
        self._pos: dict[tuple[int, int], torch.Tensor] = {}
        self._rope: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def cast(self, dtype: torch.dtype) -> PointDiT:
        """Weights, biases and gates (`g1h`, `g2h`) in `dtype`; norm scales, positions, kernels gates stay float32."""
        for name in ("wy", "wp", "final_w", "final_b"):
            setattr(self, name, getattr(self, name).to(dtype))
        for blk in self.layers:
            for name in ("qkv_w", "qkv_b", "proj_w", "proj_b", "w12", "b12", "w3", "b3"):
                setattr(blk, name, getattr(blk, name).to(dtype))
            blk.register_buffer("g1h", blk.g1.to(dtype))
            blk.register_buffer("g2h", blk.g2.to(dtype))
        return self

    def _position(self, h: int, w: int) -> torch.Tensor:
        g = self.pos
        if (h, w) != (GRID, GRID):
            g = F.interpolate(g, size=(h, w), mode="bicubic", align_corners=False)
        return g[0].reshape(g.shape[1], -1).T + self.const

    def position(self, h: int, w: int) -> torch.Tensor:
        """`[h * w, C]` float32: the position terms (bicubic from the 32 x 32 grid as upstream) + the biases."""
        if torch.compiler.is_compiling():
            return self._position(h, w)
        if (h, w) not in self._pos:
            if len(self._pos) >= 8:  # noqa: PLR2004 - cache policy
                self._pos.clear()
            with torch.inference_mode(mode=False):  # normal tensors: the cache may meet autograd (module() traced)
                self._pos[h, w] = self._position(h, w)
        return self._pos[h, w]

    @staticmethod
    def _rope_table(h: int, w: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        a = rope_angles(h, w, device)
        a = torch.cat([a, a], -1)[:, None, None]
        return a.cos(), a.sin()

    def rope(self, h: int, w: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """`(cos, sin)` `[h * w, 1, 1, 64]` float32 of the rotate-half layout (broadcast over q / k and heads)."""
        if torch.compiler.is_compiling():
            return self._rope_table(h, w, device)
        if (h, w) not in self._rope:
            if len(self._rope) >= 8:  # noqa: PLR2004 - cache policy
                self._rope.clear()
            with torch.inference_mode(mode=False):
                self._rope[h, w] = self._rope_table(h, w, device)
        return self._rope[h, w]

    def condition(self, hidden: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """DINOv3 hidden states `[B, 4, P + N, Cenc]` float32 -> the per-image token constant `[B, N, C]` float32."""
        b, _, n, ce = hidden.shape
        y = F.layer_norm(hidden[:, :, n - h * w :], (ce,), self.ln_w, self.ln_b, LN_EPS)  # [B, 4, N, Cenc]
        y = y.permute(0, 2, 1, 3).reshape(b, h * w, 4 * ce)
        return F.linear(y.to(self.wy.dtype), self.wy).float() + self.position(h, w)


def rms(x: torch.Tensor, a: torch.Tensor | float) -> torch.Tensor:
    """RMSNorm times `a` over the last dim with float32 statistics, in the dtype of `x`."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * a).to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """`[-x2, x1]` of the two halves of the last dim."""
    x1, x2 = x.chunk(2, -1)
    return torch.cat([-x2, x1], -1)


def block(net: PointDiT, blk: Block, s: int, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """One block of step `s` in plain PyTorch (SDPA)."""
    b, n, c = x.shape
    heads = net.cfg.heads
    qkv = F.linear(rms(x, blk.a1[s]), blk.qkv_w, blk.qkv_b[s]).view(b, n, 3, heads, HEAD)
    qk = qkv[:, :, :2].float()
    qk = qk * torch.rsqrt(qk.pow(2).mean(-1, keepdim=True) + EPS) * blk.qk_norm
    qk = (qk * rope[0] + rotate_half(qk) * rope[1]).to(x.dtype)
    q, k, v = qk[:, :, 0].transpose(1, 2), qk[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c)
    x = torch.addcmul(x, F.linear(o, blk.proj_w, blk.proj_b), blk.g1h[s])
    g, u = F.linear(rms(x, blk.a2[s]), blk.w12, blk.b12[s]).chunk(2, -1)
    return torch.addcmul(x, F.linear(F.silu(g) * u, blk.w3, blk.b3), blk.g2h[s])


def sample(net: PointDiT, cond: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Upstream `generate` from the token constant `[B, N, C]` float32 -> point tokens `[B, N, 3 * 16 * 16]` float32."""
    dt = net.wy.dtype
    rope = net.rope(h, w, cond.device)
    z = None
    for s, (step, one_minus_t) in enumerate(net.schedule):
        x = cond if z is None else cond + F.linear(z.to(dt), net.wp).float()
        x = x.to(dt)
        for blk in net.layers:
            x = block(net, blk, s, x, rope)
        out = F.linear(rms(x, 1.0), net.final_w[s], net.final_b[s]).float()
        z = step * (out / one_minus_t) if z is None else z + step * ((out - z) / one_minus_t)
    if z is None:  # steps >= 1 (the card's sampler)
        msg = "no sampling steps"
        raise ValueError(msg)
    return z


def unpatchify(z: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Point tokens `[B, h * w, (p, q, c)]` -> points `[B, h * 16, w * 16, 3]`."""
    b = z.shape[0]
    return z.view(b, h, w, PATCH, PATCH, 3).permute(0, 1, 3, 2, 4, 5).reshape(b, h * PATCH, w * PATCH, 3)
