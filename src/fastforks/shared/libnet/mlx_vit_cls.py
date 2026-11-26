"""Vision Transformer classifiers (timm `VisionTransformer` / `Eva`) written in MLX, with `fastforks.shared.mlx_vit`.

ViT, DeiT, SigLIP, EVA-02, the ViT / EVA-02 WD taggers and the Falconsai ViT: patch embedding as one matmul, the
position table (with the patch bias and the cls token folded in), `norm_pre`, the blocks, the final norm, the pool
(cls token, patch mean or `AttentionPoolLatent`), `fc_norm` and the head, all one `mx.compile`d forward. At these
token counts (197-785) that is 1.5-5.5 % faster warm than `mlx_vit.blocks(bounded=True)` (M1 Pro, float16:
wd-vit-large 189 -> 179 ms, wd-eva02-large 207 -> 195 ms).

EVA-02's blocks: the separate q / k / v projections packed into one (k has no bias), the interleaved 2D RoPE turned
into `mlx_vit.rotate`'s half layout by reordering each head's q and k channels (evens, then odds; `q . k` sums over
them, so one permutation of both leaves the attention as it is) and its tables alike (float32, as timm rotates, with
identity rows for the cls token), the SwiGLU's two input projections packed into `w12` with the LayerNorm before its
output projection (`ffn_norm`).

The exported-graph translation spent 0.8-2.8 s of the first call exporting and translating; written out, the first
call builds the shaders only. The float32 conversion is cached (`converted`; a cache hit reads neither the checkpoint
nor builds the twin, the configuration comes from a meta-device network). A miss converts the twin's memory-mapped
checkpoint (`lazy_build`) block by block in float64; the cache is cast to the run dtype one by one.
"""

# ruff: noqa: ANN401 - timm's architecture-specific modules

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from fastforks._core.mlx import converted, mx, to_mx, to_torch
from fastforks.shared.mlx_vit import _putter, blocks, patches, rope_tables, transformer_params

from .mlx_features import _act, _deinterleave, _qk_order

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks._core.runner import Context

__all__ = ["Network", "supports"]

LAYOUT = "vit-cls-v1"  # the cached float32 layout of `_params` (change it with `_params`)


@dataclass(frozen=True)
class Config:
    """What the forward needs of the timm network besides its weights."""

    patch: int
    grid: tuple[int, int]
    heads: int
    eps: float  # the blocks' LayerNorms
    act: Callable[[mx.array], mx.array]
    prefix: int  # cls token rows in front of the patches (0 or 1)
    hidden_size: int
    depth: int
    pre_eps: float | None  # `norm_pre`, None: Identity
    norm_eps: float | None  # the final `norm`, None: Identity
    pool: str  # "token", "avg" or "map" (`AttentionPoolLatent`)
    pool_prefix: bool  # the attention pool reads the prefix tokens too
    fc_norm_eps: float | None  # `fc_norm`, None: Identity
    rope: bool  # Eva: 2D RoPE (interleaved, `RotaryEmbeddingCat`)
    pool_heads: int = 0
    pool_eps: float = 0.0
    pool_act: Callable[[mx.array], mx.array] | None = None


def _name(m: torch.nn.Module | None) -> str:
    return type(m).__name__


def _eps(m: Any) -> float | None:
    return None if _name(m) == "Identity" else m.eps


def supports(net: torch.nn.Module) -> bool:
    """Whether `net` is a timm `VisionTransformer` or `Eva` (the architectures this module writes out)."""
    return _name(net) in {"VisionTransformer", "Eva"}


def _check_blocks(net: Any, *, eva: bool) -> None:
    for blk in net.blocks:
        at, mlp = blk.attn, blk.mlp
        bad = [n for n in ("q_norm", "k_norm", "norm") if _name(getattr(at, n, None)) not in {"Identity", "NoneType"}]
        if getattr(at, "gate", None) is not None or getattr(at, "rotate_half", False):
            bad.append("gate / rotate-half")
        if _name(blk) not in {"Block", "EvaBlock"} or any(
            _name(getattr(blk, n, None)) not in {"Identity", "LayerScale", "NoneType"} for n in ("ls1", "ls2")
        ):
            bad.append(_name(blk))
        if _name(mlp) == "SwiGLU":
            if not eva or _name(mlp.act) != "SiLU":
                bad.append("SwiGLU")
            elif _name(mlp.norm) != "Identity" and mlp.norm.eps != blk.norm1.eps:
                bad.append("SwiGLU norm eps")
        elif _name(mlp) != "Mlp" or _name(mlp.norm) != "Identity":
            bad.append(_name(mlp))
        if bad:
            msg = f"unsupported ViT block for MLX: {sorted(set(bad))}"
            raise NotImplementedError(msg)


def _config(net: Any) -> Config:
    """The configuration of a timm `VisionTransformer` / `Eva` classifier (NotImplementedError otherwise)."""
    eva = _name(net) == "Eva"
    if (
        not supports(net)
        or net.pos_embed is None
        or getattr(net, "reg_token", None) is not None
        or getattr(net, "num_reg_tokens", 0)
        or getattr(net, "no_embed_class", False)
        or getattr(net, "dynamic_img_size", False)
        or (eva and (_name(net.rope) != "RotaryEmbeddingCat" or net.rope.rotate_half))
        or (not eva and getattr(net, "rope", None) is not None)
    ):
        msg = "the MLX ViT classifier supports absolute position tables (EVA: plus interleaved RoPE), no registers"
        raise NotImplementedError(msg)
    _check_blocks(net, eva=eva)
    pool = net.attn_pool
    if pool is None and net.global_pool not in {"token", "avg"}:
        msg = f"unsupported ViT pool {net.global_pool!r}"
        raise NotImplementedError(msg)
    if pool is not None and (pool.pos_embed is not None or pool.latent.shape[1] != 1 or pool.pool != "token"):
        msg = "unsupported attention pool (position embedding, several latents or mean pooling)"
        raise NotImplementedError(msg)
    if _name(net.head) not in {"Linear", "Identity"}:
        msg = f"unsupported ViT head {_name(net.head)}"
        raise NotImplementedError(msg)
    b = net.blocks[0]
    return Config(
        patch=net.patch_embed.proj.kernel_size[0],
        grid=tuple(net.patch_embed.grid_size),
        heads=b.attn.num_heads,
        eps=b.norm1.eps,
        act=(lambda x: x) if _name(b.mlp) == "SwiGLU" else _act(b.mlp.act),
        prefix=int(net.cls_token is not None),
        hidden_size=net.embed_dim,
        depth=len(net.blocks),
        pre_eps=_eps(net.norm_pre),
        norm_eps=_eps(net.norm),
        pool="map" if pool is not None else net.global_pool,
        pool_prefix=eva or getattr(net, "pool_include_prefix", False),
        fc_norm_eps=_eps(net.fc_norm),
        rope=eva,
        pool_heads=0 if pool is None else pool.num_heads,
        pool_eps=0.0 if pool is None else pool.norm.eps,
        pool_act=None if pool is None else _act(pool.mlp.act),
    )


def _block_state(blk: Any, cfg: Config, order: torch.Tensor | None) -> dict[str, torch.Tensor]:
    """One block under `transformer_params`' names (as block 0); an Eva's packed qkv (RoPE-reordered) and SwiGLU."""
    if order is None:
        return {f"blocks.0.{k}": v.detach() for k, v in blk.state_dict().items()}
    state: dict[str, torch.Tensor] = {}
    b, at, mlp = "blocks.0.", blk.attn, blk.mlp
    for n in ("norm1", "norm2"):
        state[f"{b}{n}.weight"], state[f"{b}{n}.bias"] = getattr(blk, n).weight, getattr(blk, n).bias
    if at.qkv is not None:
        w = at.qkv.weight
        bias = at.qkv.bias if at.q_bias is None else torch.cat((at.q_bias, at.k_bias, at.v_bias))
    else:
        w = torch.cat([at.q_proj.weight, at.k_proj.weight, at.v_proj.weight])
        zero = torch.zeros(cfg.hidden_size)
        bias = torch.cat([m.bias if m.bias is not None else zero for m in (at.q_proj, at.k_proj, at.v_proj)])
    if bias is None:
        bias = torch.zeros(3 * cfg.hidden_size)
    state[b + "attn.qkv.weight"], state[b + "attn.qkv.bias"] = w[order], bias[order]
    state[b + "attn.proj.weight"], state[b + "attn.proj.bias"] = at.proj.weight, at.proj.bias
    if _name(mlp) == "SwiGLU":  # gate first: `mlx_vit.block` computes silu(x1) * x2
        state[b + "mlp.w12.weight"] = torch.cat([mlp.fc1_g.weight, mlp.fc1_x.weight])
        state[b + "mlp.w12.bias"] = torch.cat([mlp.fc1_g.bias, mlp.fc1_x.bias])
        state[b + "mlp.w3.weight"], state[b + "mlp.w3.bias"] = mlp.fc2.weight, mlp.fc2.bias
    else:
        for n in ("fc1", "fc2"):
            state[f"{b}mlp.{n}.weight"], state[f"{b}mlp.{n}.bias"] = getattr(mlp, n).weight, getattr(mlp, n).bias
    if getattr(blk, "gamma_1", None) is not None:
        state[b + "ls1.gamma"], state[b + "ls2.gamma"] = blk.gamma_1, blk.gamma_2
    return {k: v.detach() for k, v in state.items()}


def _arr(t: torch.Tensor) -> np.ndarray:
    return t.detach().double().numpy()


def _tail_params(net: Any, cfg: Config, put: Callable[[str, np.ndarray], None]) -> None:
    """The attention pool (its latent's query precomputed) and the head."""
    if cfg.pool == "map":
        a = net.attn_pool
        put("pool.q", _arr(a.latent)[0] @ _arr(a.q.weight).T + _arr(a.q.bias))
        for n, layer in (("kv", a.kv), ("proj", a.proj), ("fc1", a.mlp.fc1), ("fc2", a.mlp.fc2)):
            put(f"pool.{n}.weight", _arr(layer.weight).T)
            put(f"pool.{n}.bias", _arr(layer.bias))
        put("pool.norm.weight", _arr(a.norm.weight))
        put("pool.norm.bias", _arr(a.norm.bias))
    if _name(net.head) == "Linear":
        put("head.weight", _arr(net.head.weight).T)
        put("head.bias", _arr(net.head.bias))


def _params(net: Any, cfg: Config, dtype: mx.Dtype) -> dict[str, mx.array]:
    """The network's weights in `mlx_vit`'s layout, in `dtype` (float64 folding, cast one by one)."""
    out: dict[str, mx.array] = {}
    put = _putter(out, dtype)

    p, c = cfg.patch, cfg.hidden_size
    proj = net.patch_embed.proj
    put("patch.weight", _arr(proj.weight).transpose(2, 3, 1, 0).reshape(p * p * 3, c))
    pos = _arr(net.pos_embed)[0]
    bias = _arr(proj.bias) if proj.bias is not None else np.zeros(c)
    rows = [pos[cfg.prefix :] + bias]
    if cfg.prefix:
        rows.insert(0, _arr(net.cls_token)[0] + pos[:1])
    put("table", np.concatenate(rows))
    for n, eps in (("norm_pre", cfg.pre_eps), ("norm", cfg.norm_eps), ("fc_norm", cfg.fc_norm_eps)):
        if eps is not None:
            put(f"{n}.weight", _arr(getattr(net, n).weight))
            put(f"{n}.bias", _arr(getattr(net, n).bias))
    _tail_params(net, cfg, put)
    order = _qk_order(cfg.hidden_size, cfg.heads) if cfg.rope else None
    for i, blk in enumerate(net.blocks):  # block by block: one block's float64 transients at a time
        b = f"blocks.{i}."
        out.update({b + k[9:]: v for k, v in transformer_params(_block_state(blk, cfg, order), dtype).items()})
        if _name(blk.mlp) == "SwiGLU" and _name(blk.mlp.norm) != "Identity":
            put(b + "ffn_norm.weight", _arr(blk.mlp.norm.weight))
            put(b + "ffn_norm.bias", _arr(blk.mlp.norm.bias))
    if cfg.rope:  # float32 `(cos, sin)` of the fixed grid, in the de-interleaved order
        sin, cos = net.rope.get_embed().float().chunk(2, -1)
        head = torch.from_numpy(_deinterleave(cos.shape[-1]))
        tabs = rope_tables(to_mx(cos[:, head].contiguous()), to_mx(sin[:, head].contiguous()), cfg.prefix)
        out["rope.cos"], out["rope.sin"] = tabs  # kept float32
    return out


def _pool(p: dict[str, mx.array], cfg: Config, x: mx.array) -> mx.array:
    """`AttentionPoolLatent` (one latent, `pool="token"`) on `[B, N, C]` -> `[B, C]`."""
    b, n, c = x.shape
    heads = cfg.pool_heads
    d = c // heads
    kv = mx.addmm(p["pool.kv.bias"], x, p["pool.kv.weight"]).reshape(b, n, 2, heads, d).transpose(2, 0, 3, 1, 4)
    q = mx.broadcast_to(p["pool.q"].reshape(1, heads, 1, d), (b, heads, 1, d))
    y = mx.fast.scaled_dot_product_attention(q, kv[0], kv[1], scale=d**-0.5).reshape(b, c)
    y = mx.addmm(p["pool.proj.bias"], y, p["pool.proj.weight"])
    h = mx.fast.layer_norm(y, p["pool.norm.weight"], p["pool.norm.bias"], cfg.pool_eps)
    act = cfg.pool_act
    assert act is not None  # noqa: S101 - set with the attention pool
    h = act(mx.addmm(p["pool.fc1.bias"], h, p["pool.fc1.weight"]))
    return y + mx.addmm(p["pool.fc2.bias"], h, p["pool.fc2.weight"])


def _head(p: dict[str, mx.array], cfg: Config, t: mx.array) -> mx.array:
    """Tokens after the blocks -> logits (or the pooled features without a head), in the run dtype."""
    if cfg.norm_eps is not None:
        t = mx.fast.layer_norm(t, p["norm.weight"], p["norm.bias"], cfg.norm_eps)
    if cfg.pool == "map":
        x = _pool(p, cfg, t if cfg.pool_prefix else t[:, cfg.prefix :])
    elif cfg.pool == "avg":
        x = mx.mean(t[:, cfg.prefix :].astype(mx.float32), axis=1).astype(t.dtype)
    else:
        x = t[:, 0]
    if cfg.fc_norm_eps is not None:
        x = mx.fast.layer_norm(x, p["fc_norm.weight"], p["fc_norm.bias"], cfg.fc_norm_eps)
    return mx.addmm(p["head.bias"], x, p["head.weight"]) if "head.weight" in p else x


class Network:
    """A timm ViT / EVA classifier in MLX: converted once at load, one compiled forward per call."""

    def __init__(
        self, net: Any, dtype: str, *, ctx: Context | None = None, twin: Callable[[], Any] | None = None
    ) -> None:
        """The (CPU, eval) timm network's weights in `dtype`.

        With `ctx`, the float32 conversion is cached (`converted`): `net` then only gives the configuration (a
        meta-device network will do) and `twin()` the weights, built on a cache miss only.
        """
        self.cfg = cfg = _config(net)
        self.dtype = getattr(mx, dtype)
        if ctx is None:
            p = _params(net, cfg, self.dtype)
        else:
            p32 = converted(ctx, LAYOUT, lambda: _params(net if twin is None else twin(), cfg, None))
            p = {}
            for k in list(p32):  # one by one: a lazy cast keeps its float32 source until it is evaluated
                v = p32.pop(k)
                p[k] = v if k.startswith("rope.") else v.astype(self.dtype)
                mx.eval(p[k])
        self.rope: tuple[mx.array, mx.array] | None = (p.pop("rope.cos"), p.pop("rope.sin")) if cfg.rope else None
        self.p = p
        self.forward = mx.compile(self._forward)

    def _forward(self, x: mx.array, *rope: mx.array) -> mx.array:
        cfg, p = self.cfg, self.p
        table = p["table"]
        t = patches(x.astype(self.dtype), cfg.patch) @ p["patch.weight"] + table[cfg.prefix :]
        if cfg.prefix:
            t = mx.concatenate([mx.broadcast_to(table[:1], (x.shape[0], 1, t.shape[-1])), t], axis=1)
        if cfg.pre_eps is not None:
            t = mx.fast.layer_norm(t, p["norm_pre.weight"], p["norm_pre.bias"], cfg.pre_eps)
        t, _ = blocks(p, cfg.depth, cfg.heads, cfg.eps, t, (), rope=rope or None, act=cfg.act)
        return _head(p, cfg, t)

    def logits(self, x: mx.array) -> mx.array:
        """Normalised NCHW image batch -> logits in the run dtype (lazy)."""
        cfg = self.cfg
        grid = (x.shape[2] // cfg.patch, x.shape[3] // cfg.patch)
        if grid != cfg.grid or x.shape[2] % cfg.patch or x.shape[3] % cfg.patch:
            msg = f"the network takes {cfg.grid[0] * cfg.patch} x {cfg.grid[1] * cfg.patch} images, not {x.shape[2:]}"
            raise ValueError(msg)
        return self.forward(x, *(self.rope or ()))

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Normalised NCHW CPU images -> float32 CPU logits."""
        return to_torch(self.logits(to_mx(image)))
