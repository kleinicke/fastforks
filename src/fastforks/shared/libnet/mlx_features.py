"""timm `VisionTransformer` features on Apple MLX (`device="mlx"`), with the blocks of `fastforks.shared.mlx_vit`.

The maths of `features._Walk` (timm's `forward_features` / `forward_head(pre_logits=True)`) for the absolute-position
ViTs of CLIP and SigLIP 2: patch embedding, position table, `norm_pre`, the blocks (exact GELU, QuickGELU or tanh
GELU), the final norm, and the cls token or the attention pool (`AttentionPoolLatent`) as `pooler_output`.

Prepared once, at conversion (float64, stored float32): the patch convolution as one matmul on unfolded patches, the
blocks in `mlx_vit`'s layout, and the attention pool's query of its constant latent. On a cache hit the timm network
is built on the meta device for its configuration only.

Per patch grid (cached): the embedding table, resampled from the checkpoint's float32 table as timm resamples it, with
the patch bias and the cls token folded in.

A hand-written network instead of the exported-graph port: the first call compiles only (no export / translation).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from fastforks._core.mlx import converted, mx, to_mx, tuning
from fastforks.shared.mlx_vit import blocks, gelu, patches, transformer_params

from . import build, timm

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks._core.runner import Context

__all__ = ["Engine"]

LAYOUT = "vit-v1"  # version of `_prepare`'s parameter layout (part of the conversion cache's file name)
TABLES = 8  # embedding tables kept (one per patch grid, least recently used dropped)
COMPILED = 8  # compiled forwards kept (one per set of hidden-state layers)
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


def _sub(module: torch.nn.Module, name: str) -> torch.nn.Module | None:
    try:
        return module.get_submodule(name)
    except AttributeError:
        return None


def _config(net: Any) -> Config:  # noqa: ANN401 - a timm VisionTransformer
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


def _prepare(state: dict[str, torch.Tensor], cfg: Config) -> dict[str, mx.array]:
    """The timm state dict -> the float32 arrays the forward uses. Empties `state` as it goes."""
    out: dict[str, mx.array] = {}

    def take(k: str) -> np.ndarray:
        return state.pop(k).double().numpy()

    def put(k: str, v: np.ndarray) -> None:
        out[k] = mx.array(np.ascontiguousarray(v, dtype=np.float32))

    p, c = cfg.patch, cfg.hidden_size
    put("patch.weight", take("patch_embed.proj.weight").transpose(2, 3, 1, 0).reshape(p * p * 3, c))
    put("patch.bias", take("patch_embed.proj.bias") if "patch_embed.proj.bias" in state else np.zeros(c))
    put("pos", take("pos_embed")[0])
    if cfg.cls:
        put("cls", take("cls_token")[0])
    for n in ("norm_pre", "norm") if cfg.pre_eps is not None else ("norm",):
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
    out.update(transformer_params({k: v for k, v in state.items() if k.startswith("blocks.")}))
    rest = sorted(k for k in state if not k.startswith("blocks."))
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


def forward(
    p: dict[str, mx.array], cfg: Config, layers: tuple[int, ...], x: mx.array, table: mx.array
) -> tuple[mx.array, ...]:
    """Normalised `[B, 3, H, W]` -> float32 `(last_hidden_state, pooler_output, *hidden states at layers)`."""
    prefix = 1 if cfg.cls else 0
    t = patches(x.astype(table.dtype), cfg.patch) @ p["patch.weight"] + table[prefix:]
    if prefix:
        t = mx.concatenate([mx.broadcast_to(table[:1], (x.shape[0], 1, t.shape[-1])), t], axis=1)
    if cfg.stream_f32:  # the GEMMs stay in the weights' dtype (`mlx_vit.block`)
        t = t.astype(mx.float32)
    if cfg.pre_eps is not None:
        t = mx.fast.layer_norm(t, p["norm_pre.weight"], p["norm_pre.bias"], cfg.pre_eps)
    t, hs = blocks(p, cfg.depth, cfg.heads, cfg.eps, t, layers, act=cfg.act)
    last = mx.fast.layer_norm(t, p["norm.weight"], p["norm.bias"], cfg.norm_eps)
    pooled = last[:, 0] if cfg.pool_heads is None else _pool(p, cfg, last.astype(table.dtype))
    return tuple(a.astype(mx.float32) for a in (last, pooled, *hs))


class Engine:
    """The MLX network of one loaded model: converted weights, embedding tables and compiled forwards."""

    def __init__(self, ctx: Context) -> None:
        """Convert (or read the cached) weights for `ctx`; the configuration from timm's (meta) network."""
        self.dtype = getattr(mx, ctx.dtype)
        holder: dict[str, Config] = {}

        def convert() -> dict[str, mx.array]:
            net = build(ctx, dynamic_img_size=True)
            holder["cfg"] = _config(net)
            state = dict(net.state_dict())
            del net
            return _prepare(state, holder["cfg"])

        p32 = converted(ctx, LAYOUT, convert)
        if "cfg" not in holder:
            with torch.device("meta"):
                holder["cfg"] = _config(timm().create_model(ctx.params.arch, pretrained=False, dynamic_img_size=True))
        self.cfg = replace(holder["cfg"], stream_f32=ctx.variant.ref.split("@")[0] in STREAM_F32)
        host = {k: torch.from_numpy(np.array(p32.pop(k))) for k in ("pos", "patch.bias", "cls") if k in p32}
        self.pos, self.bias, self.cls_token = host["pos"][None], host["patch.bias"], host.get("cls")
        self.p = {k: v.astype(self.dtype) for k, v in p32.items()}
        p32.clear()
        mx.eval(self.p)
        self.compile = tuning().compile_steps
        self.tables: OrderedDict[tuple[int, int], mx.array] = OrderedDict()
        self.compiled: OrderedDict[tuple[int, ...], Callable[[mx.array, mx.array], tuple[mx.array, ...]]] = (
            OrderedDict()
        )

    def table(self, h0: int, w0: int) -> mx.array:
        """`[prefix + h0 * w0, C]` in the run dtype: cls token + its position, then the patch positions + bias."""
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
            table = to_mx(torch.cat(rows)).astype(self.dtype)
            mx.eval(table)
            self.tables[key] = table
            while len(self.tables) > TABLES:
                self.tables.popitem(last=False)
        self.tables.move_to_end(key)
        return self.tables[key]

    def _forward(self, layers: tuple[int, ...]) -> Callable[[mx.array, mx.array], tuple[mx.array, ...]]:
        if layers not in self.compiled:
            params, cfg = self.p, self.cfg

            def run(x: mx.array, table: mx.array) -> tuple[mx.array, ...]:
                return forward(params, cfg, layers, x, table)

            self.compiled[layers] = mx.compile(run) if self.compile else run
            while len(self.compiled) > COMPILED:
                self.compiled.popitem(last=False)
        self.compiled.move_to_end(layers)
        return self.compiled[layers]

    def __call__(self, image: torch.Tensor, layers: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
        """Normalised `[B, 3, H, W]` CPU tensor -> float32 CPU `(last_hidden_state, pooler_output, *hidden states)`."""
        patch = self.cfg.patch
        table = self.table(image.shape[2] // patch, image.shape[3] // patch)
        out = self._forward(tuple(layers))(to_mx(image), table)
        mx.eval(*out)
        return tuple(torch.from_numpy(np.array(a)) for a in out)
