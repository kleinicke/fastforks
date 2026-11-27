"""DINOv3 ViT on Apple MLX (`device="mlx"`): every size (ViT-S / B / L / H+), blocks of `fastforks.shared.mlx_vit`.

The maths of `net.DINOv3ViT` with the outputs of the torch runners: `last_hidden_state`, and the hidden states HF
numbers 0 (embeddings) to L (after block L, no norm).

Prepared once, at conversion (exact in real arithmetic, done in float64, stored float32):
- q / k / v projections packed into one `[C, 3 C]` matmul (k has no bias: zeros).
- LayerScale folded into the attention's output projection and the MLP's last layer.
- the gated SiLU MLP (ViT-H+) as the shared SwiGLU layout: gate and up projections side by side.
- the patch embedding as one matmul on unfolded patches; cls and register tokens as one prefix `[5, C]`.

Per grid size (cached): the RoPE tables, computed in float32 by the torch runners' `net.rope_table`, with identity
rows for the prefix tokens.

float16: weights and activations fp16, LayerNorm statistics and the attention's softmax in fp32 (inside MLX's
kernels). As in the torch fp16 runners, ViT-B / L / H+ run with the residual stream divided by 8 (it reaches 1e5
unscaled, past the fp16 range): the scale is a power of two folded into the embedding and the residual branches'
last layers when the weights are cast, with the LayerNorm eps divided by 64; hidden states are multiplied back.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np
import torch

from fastforks._core.mlx import converted, mx, to_mx
from fastforks.shared.mlx_vit import blocks, patches, rope_tables

from .net import CONFIGS, PATCH, REGISTERS, Config, rope_table

if TYPE_CHECKING:
    from fastforks.features._runner import Context

LAYOUT = "v1"  # version of `prepare`'s parameter layout (part of the conversion cache's file name)
TABLES = 8  # RoPE tables kept (one per grid size, least recently used dropped)
WIDE = 768  # ViT-B and wider: fp16 residual stream past 65504 without the scale
STREAM_SCALE = 8.0
SCALED = ("prefix", "patch.", "proj.", "fc2.", "w3.")  # what the stream scale divides (embedding, branch outputs)


def prepare(state: dict[str, torch.Tensor]) -> dict[str, mx.array]:
    """Checkpoint tensors (HF names) -> the float32 arrays the forward uses. Empties `state` as it goes."""
    out: dict[str, mx.array] = {}

    def take(k: str) -> np.ndarray:
        return state.pop(k).double().numpy()

    def put(k: str, v: np.ndarray) -> None:
        out[k] = mx.array(np.ascontiguousarray(v, dtype=np.float32))

    depth = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("layer."))
    put("patch.weight", take("embeddings.patch_embeddings.weight").transpose(2, 3, 1, 0).reshape(PATCH * PATCH * 3, -1))
    put("patch.bias", take("embeddings.patch_embeddings.bias"))
    put("prefix", np.concatenate([take("embeddings.cls_token")[0], take("embeddings.register_tokens")[0]]))
    for i in range(depth):
        src, b = f"layer.{i}.", f"blocks.{i}."
        for n in ("norm1", "norm2"):
            put(f"{b}{n}.weight", take(f"{src}{n}.weight"))
            put(f"{b}{n}.bias", take(f"{src}{n}.bias"))
        g1, g2 = take(f"{src}layer_scale1.lambda1"), take(f"{src}layer_scale2.lambda1")
        a = src + "attention."
        q, k, v = (take(f"{a}{n}_proj.weight") for n in "qkv")
        put(f"{b}qkv.weight", np.concatenate([q, k, v]).T)
        qb = take(f"{a}q_proj.bias")
        put(f"{b}qkv.bias", np.concatenate([qb, np.zeros_like(qb), take(f"{a}v_proj.bias")]))
        put(f"{b}proj.weight", take(f"{a}o_proj.weight").T * g1)
        put(f"{b}proj.bias", take(f"{a}o_proj.bias") * g1)
        m = src + "mlp."
        if f"{m}gate_proj.weight" in state:  # gated SiLU (ViT-H+): silu(gate(x)) * up(x)
            put(f"{b}w12.weight", np.concatenate([take(f"{m}gate_proj.weight"), take(f"{m}up_proj.weight")]).T)
            put(f"{b}w12.bias", np.concatenate([take(f"{m}gate_proj.bias"), take(f"{m}up_proj.bias")]))
            last = "w3"
        else:
            put(f"{b}fc1.weight", take(f"{m}up_proj.weight").T)
            put(f"{b}fc1.bias", take(f"{m}up_proj.bias"))
            last = "fc2"
        put(f"{b}{last}.weight", take(f"{m}down_proj.weight").T * g2)
        put(f"{b}{last}.bias", take(f"{m}down_proj.bias") * g2)
    put("norm.weight", take("norm.weight"))
    put("norm.bias", take("norm.bias"))
    return out


def forward(  # noqa: PLR0913, PLR0917 - the network's pieces
    p: dict[str, mx.array],
    cfg: Config,
    layers: tuple[int, ...],
    scale: float,
    x: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> tuple[mx.array, ...]:
    """Normalised `[B, 3, H, W]` -> float32 `(last_hidden_state [B, N, C], *hidden states at layers)`."""
    eps = cfg.eps / scale**2
    prefix = p["prefix"]
    t = patches(x.astype(prefix.dtype), PATCH) @ p["patch.weight"] + p["patch.bias"]
    t = mx.concatenate([mx.broadcast_to(prefix, (x.shape[0], *prefix.shape)), t], axis=1)
    t, hs = blocks(p, cfg.num_layers, cfg.num_heads, eps, t, layers, (cos, sin), bounded=True)
    last = mx.fast.layer_norm(t, p["norm.weight"], p["norm.bias"], eps)
    return (last.astype(mx.float32), *(h.astype(mx.float32) * scale for h in hs))


class Engine:
    """The MLX network of one loaded model: converted weights, RoPE tables and the forward (block by block)."""

    def __init__(self, ctx: Context) -> None:
        """Convert (or read the cached) weights for `ctx`."""
        self.cfg = CONFIGS[ctx.params.arch]
        self.dtype = getattr(mx, ctx.dtype)
        half = ctx.dtype == "float16"
        self.scale = STREAM_SCALE if half and self.cfg.hidden_size >= WIDE else 1.0
        scale = self.scale

        def scaled(k: str) -> bool:  # `blocks.<i>.proj.weight` -> `proj.weight`
            name = k.split(".", 2)[2] if k.startswith("blocks.") else k
            return scale != 1.0 and name.startswith(SCALED)

        def build() -> dict[str, mx.array]:  # the scale (a power of two: exact) before the cache's cast
            return {k: v / scale if scaled(k) else v for k, v in prepare(ctx.state_dict()).items()}

        layout = LAYOUT if scale == 1.0 else f"{LAYOUT}-s{scale:g}"
        self.p = converted(ctx, layout, build, dtype=ctx.dtype)
        self.tables: OrderedDict[tuple[int, int], tuple[mx.array, mx.array]] = OrderedDict()

    def rope(self, h0: int, w0: int) -> tuple[mx.array, mx.array]:
        """`(cos, sin)` `[5 + h0 * w0, head_dim]` in the run dtype (identity rows for cls and registers)."""
        key = (h0, w0)
        if key not in self.tables:
            cfg = self.cfg
            cos, sin = rope_table(h0, w0, cfg.hidden_size // cfg.num_heads, cfg.rope_theta, torch.device("cpu"))
            cs = tuple(a.astype(self.dtype) for a in rope_tables(to_mx(cos), to_mx(sin), 1 + REGISTERS))
            mx.eval(*cs)
            self.tables[key] = (cs[0], cs[1])
            while len(self.tables) > TABLES:
                self.tables.popitem(last=False)
        self.tables.move_to_end(key)
        return self.tables[key]

    def __call__(self, image: torch.Tensor, layers: tuple[int, ...]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Normalised `[B, 3, H, W]` CPU tensor -> float32 CPU `(last_hidden_state, hidden states at layers)`."""
        cos, sin = self.rope(image.shape[2] // PATCH, image.shape[3] // PATCH)
        out = forward(self.p, self.cfg, tuple(layers), self.scale, to_mx(image), cos, sin)
        mx.eval(*out)
        last, *hs = (torch.from_numpy(np.array(a)) for a in out)
        return last, hs
