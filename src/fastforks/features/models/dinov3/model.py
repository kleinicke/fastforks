"""DINOv3 ViT runners: HF-format checkpoints (`facebook/dinov3-*-pretrain-lvd1689m`) on `net.DINOv3ViT`.

Outputs as `transformers` `DINOv3ViTModel`: `hidden_states[0]` = embeddings `[cls, 4 registers, patches]`,
`hidden_states[i]` = output of block i (no norm), `last_hidden_state` = final LayerNorm, `pooler_output` = its cls.
fp16 (portable on CUDA, kernels) folds a stream scale of 8 into the weights of ViT-B / L / H+ (see `net`).
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core import kernels
from fastforks._core.errors import BackendUnavailableError, InputError
from fastforks.features._runner import Forward, Runner
from fastforks.kernels.fastbb import Flash64
from fastforks.kernels.fastbb import load as load_kernels
from fastforks.shared.cuda_graphs import CAPTURE_LOCK, GraphCache, capture_stream
from fastforks.shared.depth_anything_v2.fast_vit import HEAD_DIM, _flash, _fp16_accumulate

from .net import CONFIGS, PATCH, REGISTERS, DINOv3ViT, _Mlp, rope_table

if TYPE_CHECKING:
    from fastforks.features._runner import Context


@dataclass(frozen=True)
class Params:
    """Variant `params`."""

    arch: str  # "vits16", "vitb16", "vitl16" or "vith16plus"
    fp16_accumulate: bool  # portable: cuBLAS fp16 accumulation in the GEMMs (option)


_WIDE = 768  # ViT-B and wider: fp16 residual stream past 65504 without the scale


def _stream_scale(arch: str) -> float:
    return 8.0 if CONFIGS[arch].hidden_size >= _WIDE else 1.0


def _network(ctx: Context, stream_scale: float) -> DINOv3ViT:
    return (
        DINOv3ViT(CONFIGS[ctx.params.arch]).load(ctx.state_dict(), stream_scale, f"{ctx.variant.ref} checkpoint").eval()
    )


def _init(runner: Runner, arch: str) -> None:
    cfg = CONFIGS[arch]
    runner.hidden_size, runner.num_layers = cfg.hidden_size, cfg.num_layers
    runner.num_prefix_tokens, runner.num_register_tokens = 1 + REGISTERS, REGISTERS


def _outputs(last: torch.Tensor, hs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    last = last.float()
    out = {"last_hidden_state": last, "pooler_output": last[:, 0]}
    if hs:
        out["hidden_states"] = torch.stack([h.float() for h in hs], 1)
    return out


class Portable(Runner):
    """Plain PyTorch with SDPA attention: float32 (HF's maths), or fp16 weights / activations (stream scale 8)."""

    dtypes = ("float32", "float16")
    batch = None
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.half = ctx.dtype == "float16"
        self.acc = self.half and ctx.params.fp16_accumulate
        net = _network(ctx, _stream_scale(ctx.params.arch) if self.half else 1.0)
        # fp16 weights (no per-call autocast casts), converted on the CPU; packed qkv
        self.net = (net.half() if self.half else net).pack_qkv().to(ctx.device)
        _init(self, ctx.params.arch)

    def _run(self, x: torch.Tensor, layers: tuple[int, ...]) -> dict[str, torch.Tensor]:
        with _fp16_accumulate() if self.acc else contextlib.nullcontext():
            return _outputs(*self.net.run(x.half() if self.half else x, layers))

    def __call__(self, image: torch.Tensor, layers: tuple[int, ...]) -> dict[str, torch.Tensor]:
        """Features of a resized, normalised batch."""
        return self._run(image, layers)

    def module(self) -> torch.nn.Module:
        """The network: normalised `[B, 3, h, w]` (sides multiples of 16) -> `last_hidden_state` `[B, N, C]`."""
        return Forward(self.net, lambda _net, x: self._run(x, ())["last_hidden_state"])


# flash attention precision (fastbb `flash64_*`): ViT-S / B have q / k outliers up to 2e3 (q k^T 2x / 36x the fp16
# range on COCO: "hh" gives NaN, which `Flash64` would catch, and loses accuracy below the range), so they use "fh"
# (fp32 q k^T); ViT-L / H+ run "hh"
_KERNELS_ATTN = {"vits16": "fh", "vitb16": "fh", "vitl16": "hh", "vith16plus": "hh"}


# CUDA graphs (`-graph`): one per input shape and layer set while the call is CPU launch bound, i.e. up to this
# many activations B * N * C (every ViT up to 512 px, ViT-S up to 1024 px);
# larger calls are GPU bound and run eagerly (the same kernels)
_GRAPH_WORK = 1 << 21  # fixed: launch-bound threshold, measured
# graphs kept per model: at most this many, each with at most `_GRAPH_BYTES / _GRAPHS` of outputs (last_hidden_state +
# the requested hidden states, fp16; the intermediates share one memory pool, not counted). A shape is captured on its
# second call; when the cache is full, a graph unused for `_GRAPH_STALE` graph-sized calls makes room, else the new
# shape runs eagerly
# (shapes cycling through more than `_GRAPHS` keep `_GRAPHS` graphs instead of recapturing on every call)
_GRAPHS = 8  # fixed: graph cache policy, not architecture
_GRAPH_BYTES = 256 << 20  # fixed: graph cache policy
_GRAPH_STALE = 64  # fixed: graph cache policy
_ROPE_SIZES = 8  # fixed: cache policy, RoPE tables kept (one per patch grid; video at one size always hits)
_SEEN = 64  # fixed: shapes remembered for the second-call rule
_INT32 = 2**31 - 1  # fixed: `rope_qk_h` indexes in int32


class Kernels(Runner):
    """fastbb: fp16 weights / activations, fp16-accumulate cuBLAS GEMMs, packed qkv, fused RoPE, flash attention.

    RoPE is one fastbb kernel in place on the packed q / k (patch tokens; q also takes the softmax scale 2^-3, so the
    attention runs with scale 1); LayerNorm, GELU / SiLU gate and the residual + LayerScale run in PyTorch fp16;
    stream scale 8 for ViT-B / L / H+. Outputs float32.

    Small calls (`_GRAPH_WORK`) replay a CUDA graph per input shape and layer set, captured on the shape's second
    call (outputs up to `_GRAPH_BYTES / _GRAPHS`); the model keeps up to `_GRAPHS` of them (see `_GRAPH_STALE`) until
    `release_graphs()` or until it is deleted. `GraphCache` orders calls from several threads or
    streams, so they never read each other's buffers.
    """

    batch = None
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network on the CPU, pack qkv, convert to fp16, move it to the device."""
        super().__init__(ctx)
        arch = ctx.params.arch
        cfg = CONFIGS[arch]
        if cfg.hidden_size // cfg.num_heads != HEAD_DIM:
            msg = f"fastbb flash attention needs head_dim {HEAD_DIM}"
            raise BackendUnavailableError(msg)  # mode="auto" falls back to portable
        net = _network(ctx, _stream_scale(arch))
        self.qkv = []
        for blk in net.blocks():
            a = blk.attention
            w = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight])
            b = torch.cat([a.q_proj.bias, torch.zeros_like(a.q_proj.bias), a.v_proj.bias])
            self.qkv.append((w.half().to(ctx.device), b.half().to(ctx.device)))
            del a.q_proj, a.k_proj, a.v_proj
        self.net = net.half().to(ctx.device)
        self.blocks = net.blocks()
        self.cfg, self.attn = cfg, Flash64(_KERNELS_ATTN[arch])
        self.kernels = load_kernels(ctx.device)
        self._rope: dict[tuple[int, int], torch.Tensor] = {}
        self._layers: tuple[int, ...] = ()  # the layer set of the call in flight (set under `_lock`)
        self._cache = GraphCache(self._graph_forward, graphs=_GRAPHS, stale=_GRAPH_STALE, seen=_SEEN)
        self._lock = threading.Lock()
        _init(self, arch)

    def _rope_table(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """fp32 `[h * w, 2, 32]`: cos | sin of the first half (the second half repeats it), as `rope_qk_h` reads it."""
        if (h, w) not in self._rope:
            if len(self._rope) >= _ROPE_SIZES:
                del self._rope[next(iter(self._rope))]
            cos, sin = rope_table(h, w, HEAD_DIM, self.cfg.rope_theta, device)
            self._rope[h, w] = torch.stack((cos[:, : HEAD_DIM // 2], sin[:, : HEAD_DIM // 2]), 1).contiguous()
        return self._rope[h, w]

    def _block(self, i: int, x: torch.Tensor, cs: torch.Tensor, rope: tuple) -> torch.Tensor:
        blk = self.blocks[i]
        b, n, _ = x.shape
        h = self.cfg.num_heads
        w, bias = self.qkv[i]
        qkv = F.linear(blk.norm1(x), w, bias).view(b, n, 3, h, HEAD_DIM)
        self.kernels.launch("rope_qk_h", *rope[0], qkv, cs, n, n - cs.shape[0], h, rope[1], HEAD_DIM**-0.5)
        o = blk.attention.o_proj(_flash(qkv, 1.0, self.attn))
        x = torch.addcmul(x, o, blk.layer_scale1.lambda1)
        return x.addcmul_(self._mlp(blk.mlp, blk.norm2(x)), blk.layer_scale2.lambda1)

    @staticmethod
    def _mlp(mlp: _Mlp, y: torch.Tensor) -> torch.Tensor:
        """The block MLP; SwiGLU (ViT-H+) in place: two hidden-size tensors live at the peak instead of three."""
        if mlp.gate_proj is None:
            return mlp(y)
        return mlp.down_proj(F.silu(mlp.gate_proj(y), inplace=True).mul_(mlp.up_proj(y)))

    def _forward(self, image: torch.Tensor, layers: tuple[int, ...]) -> list[torch.Tensor]:
        """fp16 `[last_hidden_state, *hidden states at layers]` (hidden states still divided by the stream scale)."""
        net = self.net
        cs = self._rope_table(image.shape[-2] // PATCH, image.shape[-1] // PATCH, image.device)
        t = net.tokens(image.half())
        total = t.shape[0] * t.shape[1] * 2 * self.cfg.num_heads * (HEAD_DIM // 4)  # q / k half2 pairs per half head
        grid = kernels.linear(total)
        if total + grid[0][0] * grid[1][0] > _INT32:  # the grid-stride index must not overflow
            msg = f"batch too large for the fastbb RoPE kernel ({t.shape[0]} x {t.shape[1]} tokens): split it"
            raise InputError(msg)
        hs = [t] if 0 in layers else []
        for i in range(len(self.blocks)):
            t = self._block(i, t, cs, (grid, total))
            if i + 1 in layers:
                hs.append(t)
        return [net.norm(t), *hs]

    def release_graphs(self) -> None:
        """Drop every CUDA graph and free its memory pool on the device (`torch.cuda.empty_cache`)."""
        self._cache.release()

    def _graph_forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        return self._forward(image, self._layers)

    def _float(self, outs: list[torch.Tensor]) -> list[torch.Tensor]:
        """`[last_hidden_state, *hidden states]` as float32, the hidden states times the stream scale."""
        return [outs[0].float(), *(t.float() * self.net.stream_scale for t in outs[1:])]

    def _eager(self, image: torch.Tensor, layers: tuple[int, ...]) -> list[torch.Tensor]:
        """`_forward` on the capture stream: no extra cuBLAS workspace (32 MB) on the caller's stream."""
        cur = torch.cuda.current_stream(image.device)
        side = capture_stream(image.device)
        with CAPTURE_LOCK, torch.cuda.stream(side):
            side.wait_stream(cur)
            outs = self._forward(image, layers)
        cur.wait_stream(side)
        for o in outs:
            o.record_stream(cur)  # freed after the caller's copy-out
        return outs

    def __call__(self, image: torch.Tensor, layers: tuple[int, ...]) -> dict[str, torch.Tensor]:
        """Features of a resized, normalised batch."""
        b, _, h, w = image.shape
        work = b * (1 + REGISTERS + (h // PATCH) * (w // PATCH)) * self.cfg.hidden_size
        with torch.autocast("cuda", enabled=False), _fp16_accumulate():
            if work <= _GRAPH_WORK and (1 + len(layers)) * work * 2 * _GRAPHS <= _GRAPH_BYTES:
                with self._lock:  # `_layers` is the argument of the graph forward
                    self._layers = layers
                    outs = self._cache.run((tuple(image.shape), layers), image, post=self._float)
                if outs is None:  # first call of the shape / cache busy / capture failed
                    outs = self._float(self._eager(image, layers))
            else:
                outs = self._float(self._forward(image, layers))
            return _outputs(outs[0], outs[1:])


RUNNERS = {"portable": Portable, "kernels": Kernels}
