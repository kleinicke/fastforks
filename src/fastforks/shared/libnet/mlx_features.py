"""timm `VisionTransformer` / `Eva` features on Apple MLX (`device="mlx"`), with `fastforks.shared.mlx_vit`'s blocks.

The maths of `features._Walk` (timm's `forward_features` / `forward_head(pre_logits=True)`) for the absolute-position
ViTs of CLIP and SigLIP 2 and the Perception Encoder `Eva`s (absolute table plus 2D RoPE): patch embedding, position
table, `norm_pre`, the blocks (exact GELU, QuickGELU or tanh GELU), the final norm (if any), and the cls token, the
attention pool (`AttentionPoolLatent`) or the patch mean as `pooler_output`.

Perception Encoder RoPE is timm's interleaved layout (pairs `(2 i, 2 i + 1)` rotate together): the q and k columns of
each head are reordered at conversion, evens then odds, and the tables alike, which turns it into `mlx_vit.rotate`'s
half layout (`q . k` is a sum over the head channels, so one permutation of both leaves the attention as it is). The
tables are float32 (timm rotates fp16 q / k in float32) with identity rows for the cls token.

Prepared once, at conversion (float64, stored float32): the patch convolution as one matmul on unfolded patches, the
blocks in `mlx_vit`'s layout, and the attention pool's query of its constant latent. On a cache hit the timm network
is built on the meta device for its configuration only.

Per patch grid (cached): the embedding table, resampled from the checkpoint's float32 table as timm resamples it, with
the patch bias and the cls token folded in.

A hand-written network instead of the exported-graph port. CLIP / SigLIP 2 evaluate their blocks one by one
(`mlx_vit.blocks(bounded=True)`, not one compiled graph): lower peak, equal or faster warm. The Evas (RoPE: the
Perception Encoders) run one compiled graph per layer selection, 3-5 % faster than the bounded blocks.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from fastforks._core.mlx import converted, mx, to_mx
from fastforks.shared.mlx_vit import blocks, gelu, patches, rope_tables, transformer_params

from . import build, timm

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks._core.runner import Context

__all__ = ["Engine"]

# version of `_prepare`'s parameter layout (part of the conversion cache's file name; the Eva layout is new, so its
# files are new too and the ViTs' cached files stay valid)
LAYOUT = "vit-v1"
TABLES = 8  # embedding tables kept (one per patch grid, least recently used dropped)
# Refs whose float16 residual stream is float32: L/14 DFN carries a channel near 120, where float16 steps are 0.06
# (clip-l14-dfn2b float16 mean 0.0209 against tol 0.02 with a float16 stream, 0.0103 float32). Not for all: it costs
# 5-8 % of the warm time on the L / H CLIPs.
STREAM_F32 = frozenset({"clip-l14-dfn2b"})


def _quick_gelu(x: mx.array) -> mx.array:
    y = x.astype(mx.float32)
    return (y * mx.sigmoid(1.702 * y)).astype(x.dtype)


def _gelu_tanh(x: mx.array) -> mx.array:
    y = x.astype(mx.float32)
    return (0.5 * y * (1.0 + mx.tanh(0.7978845608028654 * (y + 0.044715 * y * y * y)))).astype(x.dtype)


quick_gelu = mx.compile(_quick_gelu, shapeless=True)
gelu_tanh = mx.compile(_gelu_tanh, shapeless=True)


def _act(module: torch.nn.Module) -> Callable[[mx.array], mx.array]:
    """The MLX activation of a timm MLP's `act` (computed in float32, rounded once)."""
    name = type(module).__name__
    if name == "QuickGELU":
        return quick_gelu
    if name == "GELUTanh" or (name == "GELU" and getattr(module, "approximate", "none") == "tanh"):
        return gelu_tanh
    if name == "GELU":
        return gelu
    msg = f"unsupported ViT activation {name}"
    raise NotImplementedError(msg)


@dataclass(frozen=True)
class Config:
    """What the forward needs of the timm network besides its weights."""

    patch: int
    grid: tuple[int, int]
    heads: int
    eps: float  # the blocks' LayerNorms
    norm_eps: float
    pre_eps: float | None  # `norm_pre`'s, None: Identity
    act: Callable[[mx.array], mx.array]
    cls: bool  # a cls token in front of the patches
    no_embed_class: bool  # the position table has no row for the cls token
    hidden_size: int
    depth: int
    pool_heads: int | None  # attention pool's, None: the cls token is the pooler output
    pool_eps: float = 0.0
    pool_act: Callable[[mx.array], mx.array] | None = None
    stream_f32: bool = False  # the residual stream float32 (`STREAM_F32`)
    final_norm: bool = True  # `norm` is a LayerNorm (PE spatial: Identity)
    avg_pool: bool = False  # `pooler_output` is the mean of the patch tokens (PE spatial)
    rope: bool = False  # 2D RoPE tables (Eva `RotaryEmbeddingCat`, interleaved)


def _sub(module: torch.nn.Module, name: str) -> torch.nn.Module | None:
    try:
        return module.get_submodule(name)
    except AttributeError:
        return None


def _config(net: Any) -> Config:  # noqa: ANN401 - a timm VisionTransformer / Eva
    if type(net).__name__ == "Eva":
        return _eva_config(net)
    if (
        getattr(net, "reg_token", None) is not None
        or net.pos_embed is None
        or type(net).__name__ != "VisionTransformer"
    ):
        msg = "the MLX ViT supports timm VisionTransformers with an absolute position table and no register tokens"
        raise NotImplementedError(msg)
    if type(net.fc_norm).__name__ != "Identity" or net.global_pool not in {"token", "map"}:
        msg = f"unsupported ViT head: fc_norm {type(net.fc_norm).__name__}, pool {net.global_pool}"
        raise NotImplementedError(msg)
    b = net.blocks[0]
    pool = net.attn_pool
    extra = [
        n
        for blk in (*net.blocks, *([pool] if pool is not None else []))
        for n in ("attn.q_norm", "attn.k_norm", "attn.norm", "q_norm", "k_norm", "mlp.norm")
        if (m := _sub(blk, n)) is not None and type(m).__name__ != "Identity"
    ]
    if extra:
        msg = f"unsupported ViT normalisations: {sorted(set(extra))}"
        raise NotImplementedError(msg)
    if pool is not None and (pool.pos_embed is not None or pool.latent.shape[1] != 1 or pool.pool != "token"):
        msg = "unsupported attention pool (position embedding, several latents or mean pooling)"
        raise NotImplementedError(msg)
    pre = net.norm_pre
    return Config(
        patch=net.patch_embed.proj.kernel_size[0],
        grid=tuple(net.patch_embed.grid_size),
        heads=b.attn.num_heads,
        eps=b.norm1.eps,
        norm_eps=net.norm.eps,
        pre_eps=None if type(pre).__name__ == "Identity" else pre.eps,
        act=_act(b.mlp.act),
        cls=net.cls_token is not None,
        no_embed_class=net.no_embed_class,
        hidden_size=net.embed_dim,
        depth=len(net.blocks),
        pool_heads=None if pool is None else pool.num_heads,
        pool_eps=0.0 if pool is None else pool.norm.eps,
        pool_act=None if pool is None else _act(pool.mlp.act),
    )


def _eva_config(net: Any) -> Config:  # noqa: ANN401 - a timm Eva
    """The Perception Encoder `Eva`s: absolute table + interleaved 2D RoPE, plain pre-norm GELU blocks."""
    b = net.blocks[0]
    rope = net.rope
    if (
        getattr(net, "num_reg_tokens", 0)
        or net.pos_embed is None
        or type(rope).__name__ != "RotaryEmbeddingCat"
        or rope.rotate_half
        or getattr(rope, "bands", None) is None  # dynamic_img_size: the tables per grid
        or getattr(net, "rope_mixed", False)
    ):
        msg = "the MLX Eva supports absolute position tables with interleaved RoPE (Perception Encoder), no registers"
        raise NotImplementedError(msg)
    for blk in net.blocks:
        at = blk.attn
        if (
            type(blk).__name__ != "EvaBlock"
            or getattr(at, "qkv", None) is None
            or getattr(at, "rotate_half", False)
            or getattr(at, "gate", None) is not None
            or any(type(m).__name__ != "Identity" for m in (at.q_norm, at.k_norm, at.norm, blk.mlp.norm))
            or type(blk.mlp).__name__ != "Mlp"
        ):
            msg = "unsupported Eva block (separate q / k / v, gates, q / k norms, SwiGLU or rotate-half RoPE)"
            raise NotImplementedError(msg)
    pool = net.attn_pool
    if type(net.fc_norm).__name__ != "Identity" or (pool is None and net.global_pool not in {"avg", "token"}):
        msg = f"unsupported Eva head: fc_norm {type(net.fc_norm).__name__}, pool {net.global_pool}"
        raise NotImplementedError(msg)
    if pool is not None and (pool.pos_embed is not None or pool.latent.shape[1] != 1 or pool.pool != "token"):
        msg = "unsupported attention pool (position embedding, several latents or mean pooling)"
        raise NotImplementedError(msg)
    pre, norm = net.norm_pre, net.norm
    return Config(
        patch=net.patch_embed.proj.kernel_size[0],
        grid=tuple(net.patch_embed.grid_size),
        heads=b.attn.num_heads,
        eps=b.norm1.eps,
        norm_eps=getattr(norm, "eps", 0.0),
        pre_eps=None if type(pre).__name__ == "Identity" else pre.eps,
        act=_act(b.mlp.act),
        cls=net.cls_token is not None,
        no_embed_class=net.no_embed_class,
        hidden_size=net.embed_dim,
        depth=len(net.blocks),
        pool_heads=None if pool is None else pool.num_heads,
        pool_eps=0.0 if pool is None else pool.norm.eps,
        pool_act=None if pool is None else _act(pool.mlp.act),
        final_norm=type(norm).__name__ != "Identity",
        avg_pool=pool is None and net.global_pool == "avg",
        rope=True,
    )


def _deinterleave(d: int) -> np.ndarray:
    """The channel order of one head that turns interleaved RoPE pairs into the half layout: evens, then odds."""
    return np.concatenate([np.arange(0, d, 2), np.arange(1, d, 2)])


def _qk_order(c: int, heads: int) -> torch.Tensor:
    """Rows of a packed `[3 C, C]` qkv weight with each head's q and k channels de-interleaved (v as it is)."""
    d = c // heads
    head = _deinterleave(d)
    qk = [np.concatenate([s * c + h * d + head for h in range(heads)]) for s in (0, 1)]
    return torch.from_numpy(np.concatenate([*qk, np.arange(2 * c, 3 * c)]))


def _eva_state(state: dict[str, torch.Tensor], cfg: Config) -> None:
    """Eva's names and RoPE layout -> `transformer_params`' (LayerScale `gamma_*`, q / k columns de-interleaved)."""
    index = _qk_order(cfg.hidden_size, cfg.heads)
    for i in range(cfg.depth):
        b = f"blocks.{i}."
        for g, ls in (("gamma_1", "ls1"), ("gamma_2", "ls2")):
            if b + g in state:
                state[f"{b}{ls}.gamma"] = state.pop(b + g)
        for k in ("attn.qkv.weight", "attn.qkv.bias"):
            state[b + k] = state[b + k][index]


def _prepare(state: dict[str, torch.Tensor], cfg: Config) -> dict[str, mx.array]:
    """The timm state dict -> the float32 arrays the forward uses. Empties `state` as it goes."""
    out: dict[str, mx.array] = {}

    def take(k: str) -> np.ndarray:
        return state.pop(k).double().numpy()

    def put(k: str, v: np.ndarray) -> None:
        out[k] = mx.array(np.ascontiguousarray(v, dtype=np.float32))

    if cfg.rope:
        _eva_state(state, cfg)
    p, c = cfg.patch, cfg.hidden_size
    put("patch.weight", take("patch_embed.proj.weight").transpose(2, 3, 1, 0).reshape(p * p * 3, c))
    put("patch.bias", take("patch_embed.proj.bias") if "patch_embed.proj.bias" in state else np.zeros(c))
    put("pos", take("pos_embed")[0])
    if cfg.cls:
        put("cls", take("cls_token")[0])
    for n in (*(("norm_pre",) if cfg.pre_eps is not None else ()), *(("norm",) if cfg.final_norm else ())):
        put(f"{n}.weight", take(f"{n}.weight"))
        put(f"{n}.bias", take(f"{n}.bias"))
    if cfg.pool_heads is not None:
        a = "attn_pool."
        put("pool.q", take(a + "latent")[0] @ take(a + "q.weight").T + take(a + "q.bias"))
        for n, src in (("kv", "kv"), ("proj", "proj"), ("fc1", "mlp.fc1"), ("fc2", "mlp.fc2")):
            put(f"pool.{n}.weight", take(f"{a}{src}.weight").T)
            put(f"pool.{n}.bias", take(f"{a}{src}.bias"))
        put("pool.norm.weight", take(a + "norm.weight"))
        put("pool.norm.bias", take(a + "norm.bias"))
    for k in [k for k in state if k.startswith("head.")]:  # the projection after `pre_logits`
        del state[k]
    # popped out of `state` (`transformer_params` empties what it gets): PE G/14 is 7.5 GB of float32, and the
    # checkpoint's tensors kept beside the converted arrays would double the conversion's peak
    out.update(transformer_params({k: state.pop(k) for k in [k for k in state if k.startswith("blocks.")]}))
    rest = sorted(state)
    if rest:
        msg = f"unconverted ViT parameters: {rest}"
        raise NotImplementedError(msg)
    state.clear()
    return out


def _pool(p: dict[str, mx.array], cfg: Config, last: mx.array) -> mx.array:
    """The `AttentionPoolLatent` of timm (one latent, `pool="token"`) on `last_hidden_state` `[B, N, C]` -> `[B, C]`."""
    b, n, c = last.shape
    heads = cfg.pool_heads
    assert heads is not None  # noqa: S101 - only called with an attention pool
    d = c // heads
    kv = mx.addmm(p["pool.kv.bias"], last, p["pool.kv.weight"]).reshape(b, n, 2, heads, d).transpose(2, 0, 3, 1, 4)
    q = mx.broadcast_to(p["pool.q"].reshape(1, heads, 1, d), (b, heads, 1, d))
    x = mx.fast.scaled_dot_product_attention(q, kv[0], kv[1], scale=d**-0.5).reshape(b, c)
    x = mx.addmm(p["pool.proj.bias"], x, p["pool.proj.weight"])
    h = mx.fast.layer_norm(x, p["pool.norm.weight"], p["pool.norm.bias"], cfg.pool_eps)
    act = cfg.pool_act
    assert act is not None  # noqa: S101
    h = act(mx.addmm(p["pool.fc1.bias"], h, p["pool.fc1.weight"]))
    return x + mx.addmm(p["pool.fc2.bias"], h, p["pool.fc2.weight"])


def forward(  # noqa: PLR0913 - the network's inputs
    p: dict[str, mx.array],
    cfg: Config,
    layers: tuple[int, ...],
    x: mx.array,
    table: mx.array,
    *,
    rope: tuple[mx.array, mx.array] | None = None,
    bounded: bool = True,
) -> tuple[mx.array, ...]:
    """Normalised `[B, 3, H, W]` -> float32 `(last_hidden_state, pooler_output, *hidden states at layers)`.

    `rope`: the float32 `(cos, sin)` `[prefix + N, head dim]` (`Engine.tables`) of an Eva. `bounded`: the blocks
    evaluated one by one (`mlx_vit.blocks`); off inside `mx.compile`.
    """
    prefix = 1 if cfg.cls else 0
    t = patches(x.astype(table.dtype), cfg.patch) @ p["patch.weight"] + table[prefix:]
    if prefix:
        t = mx.concatenate([mx.broadcast_to(table[:1], (x.shape[0], 1, t.shape[-1])), t], axis=1)
    if cfg.stream_f32:  # the GEMMs stay in the weights' dtype (`mlx_vit.block`)
        t = t.astype(mx.float32)
    if cfg.pre_eps is not None:
        t = mx.fast.layer_norm(t, p["norm_pre.weight"], p["norm_pre.bias"], cfg.pre_eps)
    t, hs = blocks(p, cfg.depth, cfg.heads, cfg.eps, t, layers, rope=rope, act=cfg.act, bounded=bounded)
    last = mx.fast.layer_norm(t, p["norm.weight"], p["norm.bias"], cfg.norm_eps) if cfg.final_norm else t
    if cfg.pool_heads is not None:
        pooled = _pool(p, cfg, last.astype(table.dtype))
    elif cfg.avg_pool:
        pooled = mx.mean(last[:, prefix:].astype(mx.float32), axis=1)
    else:
        pooled = last[:, 0]
    return tuple(a.astype(mx.float32) for a in (last, pooled, *hs))


class Engine:
    """The MLX network of one loaded model: converted weights and embedding tables."""

    def __init__(self, ctx: Context) -> None:
        """Convert (or read the cached) weights for `ctx`; the configuration from timm's (meta) network."""
        self.dtype = getattr(mx, ctx.dtype)
        holder: dict[str, Any] = {}

        def convert() -> dict[str, mx.array]:
            net = build(ctx, dynamic_img_size=True)
            holder["cfg"] = _config(net)
            if holder["cfg"].rope:
                holder["rope"] = copy.deepcopy(net.rope).float()
            state = dict(net.state_dict())
            del net
            return _prepare(state, holder["cfg"])

        p32 = converted(ctx, LAYOUT, convert)
        if "cfg" not in holder:
            with torch.device("meta"):
                net = timm().create_model(ctx.params.arch, pretrained=False, dynamic_img_size=True)
            holder["cfg"] = _config(net)
            if holder["cfg"].rope:  # its frequency bands: a non-persistent buffer, computed (not in the checkpoint)
                rope = copy.deepcopy(net.rope).to_empty(device="cpu")
                rope.init_non_persistent_buffers()
                holder["rope"] = rope.float()
            del net
        self.rope = holder.get("rope")  # Eva: the float32 RoPE module (CPU), for the tables per grid
        self.cfg = replace(holder["cfg"], stream_f32=ctx.variant.ref.split("@")[0] in STREAM_F32)
        host = {k: torch.from_numpy(np.array(p32.pop(k))) for k in ("pos", "patch.bias", "cls") if k in p32}
        self.pos, self.bias, self.cls_token = host["pos"][None], host["patch.bias"], host.get("cls")
        self.p = {}
        for k in list(p32):  # one by one: a lazy cast keeps its float32 source until it is evaluated
            self.p[k] = p32.pop(k).astype(self.dtype)
            mx.eval(self.p[k])
        self.tables: OrderedDict[tuple[int, int], tuple[mx.array, ...]] = OrderedDict()
        self.graphs: dict[tuple[int, ...], Callable[..., tuple[mx.array, ...]]] = {}

    def table(self, h0: int, w0: int) -> tuple[mx.array, ...]:
        """`(embedding table, *RoPE tables)` of an `h0 x w0` patch grid.

        The embedding table `[prefix + h0 * w0, C]` in the run dtype: cls token + its position, then the patch
        positions + bias. An Eva's `(cos, sin)` `[prefix + h0 * w0, head dim]`: float32, in the half layout of the
        de-interleaved q / k columns, identity rows for the cls token.
        """
        key = (h0, w0)
        if key not in self.tables:
            cfg = self.cfg
            npt = 0 if cfg.no_embed_class or not cfg.cls else 1  # prefix rows of the position table
            pos = self.pos
            if (h0, w0) != cfg.grid:
                pos = timm().layers.resample_abs_pos_embed(
                    pos, new_size=(h0, w0), old_size=cfg.grid, num_prefix_tokens=npt
                )
            pos = pos[0]
            rows = [pos[npt:] + self.bias]
            if self.cls_token is not None:  # cfg.cls
                rows.insert(0, self.cls_token + (pos[:1] if npt else 0))
            tabs = (to_mx(torch.cat(rows)).astype(self.dtype),)
            if self.rope is not None:
                sin, cos = self.rope.get_embed(shape=(h0, w0)).chunk(2, -1)  # [N, head dim] each, interleaved
                head = torch.from_numpy(_deinterleave(cos.shape[-1]))
                tabs += rope_tables(to_mx(cos[:, head].contiguous()), to_mx(sin[:, head].contiguous()), int(cfg.cls))
            mx.eval(tabs)
            self.tables[key] = tabs
            while len(self.tables) > TABLES:
                self.tables.popitem(last=False)
        self.tables.move_to_end(key)
        return self.tables[key]

    def compiled(self, layers: tuple[int, ...]) -> Callable[..., tuple[mx.array, ...]]:
        """The compiled forward of an Eva (`(image, table, cos, sin)`) for `layers` (MLX retraces per shape)."""
        fn = self.graphs.get(layers)
        if fn is None:
            p, cfg = self.p, self.cfg

            def run(x: mx.array, table: mx.array, cos: mx.array, sin: mx.array) -> tuple[mx.array, ...]:
                return forward(p, cfg, layers, x, table, rope=(cos, sin), bounded=False)

            fn = self.graphs[layers] = mx.compile(run)
        return fn

    def __call__(self, image: torch.Tensor, layers: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
        """Normalised `[B, 3, H, W]` CPU tensor -> float32 CPU `(last_hidden_state, pooler_output, *hidden states)`."""
        patch = self.cfg.patch
        tabs = self.table(image.shape[2] // patch, image.shape[3] // patch)
        if len(tabs) > 1:  # RoPE: one compiled graph beats the bounded blocks by 3-5 % (no peak to bound there)
            out = self.compiled(tuple(layers))(to_mx(image), *tabs)
        else:
            out = forward(self.p, self.cfg, tuple(layers), to_mx(image), tabs[0])
        mx.eval(*out)
        return tuple(torch.from_numpy(np.array(a)) for a in out)
