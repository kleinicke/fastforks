"""Feature-extraction runners for timm ViTs (`VisionTransformer`, `Eva`), run through `fastforks._optimize`.

The network is timm's own module with the variant's weights (`libnet.build`, `dynamic_img_size=True`); the blocks run
the optimizer's forwards (portable: SDPA + folded LayerScale; kernels: fused fp16 GEMMs, RoPE kernel). This module
only walks the network the way timm's `forward_features` / `forward_head(pre_logits=True)` do, so it can return the
hidden states:

- `hidden_states[0]`: the token embeddings after `norm_pre` (HF CLIP's `pre_layrnorm` output), `[i]`: block i's output;
- `last_hidden_state`: the final `norm` of the last block's output (prefix tokens, then the patch grid row-major);
- `pooler_output`: the model's pooled embedding before any projection (attention pool / cls / patch mean, as timm).

The absolute position table and the RoPE tables are resampled in float32 from the checkpoint's values (timm resamples
the fp16 copy in fp16 runs), once per patch grid and kept for the `_TABLES` most recently used grids. A CUDA graph
holds the address of the tables it was captured with, so its cache entry keeps them alive (`Kernels._captured`).
"""

from __future__ import annotations

import contextlib
import copy
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from fastforks._core.half import cublas_gemms, fp16_accumulate, graph_sized
from fastforks._optimize import optimize, transformer
from fastforks.features._runner import Forward, Runner
from fastforks.shared.cuda_graphs import GraphCache

from . import build, timm

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks._optimize import FastModel
    from fastforks.features._runner import Context

__all__ = ["Kernels", "MetalKernels", "Params", "Portable"]

_TABLES = 8  # fixed: cache policy, resampled position / RoPE tables kept (the most recently used patch grids)
# CUDA graphs (kernels) while the call is CPU launch bound: up to this many activations B * N * C (ViT-B up to ~460 px,
# ViT-L up to ~300 px, at 16 px patches); larger calls are GPU bound and run eagerly (the same kernels)
_GRAPH_WORK = 1 << 20  # fixed: launch-bound threshold, measured
_GRAPHS = 4  # fixed: graph cache policy, not architecture


@dataclass(frozen=True)
class Params:
    """Card `[params]` / variant `params` / `[options]`."""

    library: str  # "timm"
    arch: str  # `timm.create_model` name with the pretrained tag
    fp16_accumulate: bool  # portable: cuBLAS fp16 accumulation in the GEMMs (option)
    fp32_accumulate: bool  # kernels: every GEMM through cuBLAS with fp32 accumulation (option)


class _Walk:
    """timm's `forward_features` / `forward_head(pre_logits=True)`, step by step, with float32 position tables."""

    def __init__(self, net: Any) -> None:  # noqa: ANN401 - a timm VisionTransformer / Eva (before `optimize`)
        """Keep float32 copies of the position table and the RoPE module (before the fp16 cast)."""
        self.eva = type(net).__name__ == "Eva"
        pos = getattr(net, "pos_embed", None)
        self.pos32 = None if pos is None else pos.detach().float().clone()
        rope = getattr(net, "rope", None) if self.eva else None
        self.rope32 = None if rope is None else copy.deepcopy(rope).float()
        self.grid = tuple(net.patch_embed.grid_size)
        self.npt_pos = 0 if net.no_embed_class else net.num_prefix_tokens  # prefix rows of the position table
        self._tables: OrderedDict[tuple, tuple[torch.Tensor | None, Any]] = OrderedDict()  # least recent first
        self._lock = threading.Lock()  # the runner is called from several threads

    def tables(self, h: int, w: int, dtype: torch.dtype, device: torch.device) -> tuple[Any, Any]:
        """(position table `[1, rows, C]` in `dtype`, RoPE embedding (float32)) of an `h x w` patch grid.

        Cached for the `_TABLES` most recently used grids. The cache alone owns a table: whoever keeps its address
        beyond the call (a captured CUDA graph) must keep the returned tuple.
        """
        cache = not torch.compiler.is_compiling() and torch.is_inference_mode_enabled()  # not in traces / autograd
        key = (h, w, dtype, device)
        if cache:
            with self._lock:
                hit = self._tables.get(key)
                if hit is not None:
                    self._tables.move_to_end(key)
                    return hit
        pos = None
        if self.pos32 is not None:
            p = self.pos32.to(device)
            if (h, w) != self.grid:
                p = self.resample(p, h, w)
            pos = p.to(dtype)
        rope = None
        if self.rope32 is not None:
            self.rope32.to(device)
            rope = self.rope32.get_embed(shape=(h, w))
        out = (pos, rope)
        if cache:
            with self._lock:
                out = self._tables.setdefault(key, out)  # another thread's table of the same grid wins
                self._tables.move_to_end(key)
                while len(self._tables) > _TABLES:
                    self._tables.popitem(last=False)
        return out

    def resample(self, pos: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """The float32 position table `pos` (checkpoint grid) for an `h x w` patch grid, as timm resamples it."""
        return timm().layers.resample_abs_pos_embed(
            pos, new_size=(h, w), old_size=self.grid, num_prefix_tokens=self.npt_pos
        )

    def run(
        self,
        net: Any,  # noqa: ANN401 - timm module
        x: torch.Tensor,
        layers: tuple[int, ...],
        used: list[tuple[Any, Any]] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """(`last_hidden_state`, hidden states at `layers`) of the network input `x` (in the network's dtype).

        `used`: the tables the run read (`tables`' tuple) are appended, for a caller that must keep them alive.
        """
        p = net.patch_embed(x)  # dynamic_img_size: [B, h, w, C]
        b, h, w, c = p.shape
        p = p.reshape(b, h * w, c)
        pos, rope = tabs = self.tables(h, w, p.dtype, p.device)
        if used is not None:
            used.append(tabs)
        prefix = [t.expand(b, -1, -1) for t in (net.cls_token, net.reg_token) if t is not None]
        if net.no_embed_class:
            t = torch.cat([*prefix, p + pos], 1) if pos is not None else torch.cat([*prefix, p], 1)
        else:
            t = torch.cat([*prefix, p], 1)
            t = t + pos if pos is not None else t
        t = net.norm_pre(t)
        hs = [t] if 0 in layers else []
        for i, blk in enumerate(net.blocks, 1):
            t = blk(t, rope=rope) if self.eva else blk(t)
            if i in layers:
                hs.append(t)
        return net.norm(t), hs


class _TimmFeatures(Runner):
    """Shared part of `Portable` and `Kernels`: build the timm network, optimise it, walk it."""

    batch = None  # any batch size
    Params = Params
    optimize_mode: str | None = None  # the `optimize` mode if not `ctx.mode`

    def __init__(self, ctx: Context) -> None:
        """Build the network (CPU, float32), keep its float32 tables, optimise it for `ctx.mode` on `ctx.device`."""
        super().__init__(ctx)
        net = self.network(ctx)
        self.walk = self.walker(net)
        self.hidden_size, self.num_layers = net.embed_dim, len(net.blocks)
        self.num_prefix_tokens = net.num_prefix_tokens
        self.num_register_tokens = getattr(net, "num_reg_tokens", 0)
        mode = self.optimize_mode or ctx.mode
        self.fast: FastModel = optimize(net, mode=mode, device=ctx.device, fp16=ctx.dtype == "float16")
        self.net: Any = self.fast.net
        self.gemm: Callable[[], contextlib.AbstractContextManager] = contextlib.nullcontext  # GEMM precision
        if ctx.device.type == "cuda":  # the fp32 weights stay cached otherwise
            torch.cuda.empty_cache()

    def network(self, ctx: Context) -> Any:  # noqa: ANN401 - a timm VisionTransformer / Eva
        """The network with the variant's weights (CPU, float32): timm's, built by `libnet.build`."""
        return build(ctx, dynamic_img_size=True)

    def walker(self, net: Any) -> _Walk:  # noqa: ANN401 - the network of `network`
        """The walk of `net` (made before `optimize` casts it)."""
        return _Walk(net)

    def pool(self, net: Any, last: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - the optimised network
        """`pooler_output` `[B, P]` of `last_hidden_state`: timm's `forward_head(pre_logits=True)`."""
        return net.forward_head(last, pre_logits=True)

    def _forward(
        self, image: torch.Tensor, layers: tuple[int, ...], used: list[tuple[Any, Any]] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, list]:
        """(`last_hidden_state`, `pooler_output`, hidden states) in the network's dtype; `used`: see `_Walk.run`."""
        net = self.net
        x = self.fast._input(net, image)  # noqa: SLF001 - the optimizer's input conversion (dtype, layout)
        with self.gemm():
            last, hs = self.walk.run(net, x, layers, used)
            return last, self.pool(net, last), hs

    def _run(self, image: torch.Tensor, layers: tuple[int, ...]) -> dict[str, torch.Tensor]:
        last, pooled, hs = self._forward(image, layers)
        out = {"last_hidden_state": last.float(), "pooler_output": pooled.float()}
        if layers:
            out["hidden_states"] = torch.stack([h.float() for h in hs], 1)
        return out

    def module(self) -> torch.nn.Module:
        """The network: normalised `[B, 3, h, w]` (sides multiples of the patch) -> `last_hidden_state` `[B, N, C]`.

        Without the fp16-accumulate switch (a process-global cuBLAS flag that tracing cannot capture).
        """

        def fn(_net: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
            net = self.net
            return self.walk.run(net, self.fast._input(net, x), ())[0].float()  # noqa: SLF001 - see `_forward`

        return Forward(self.net, fn)


class Portable(_TimmFeatures):
    """Plain PyTorch through `fastforks._optimize` (SDPA attention, folded LayerScale): float32, or fp16 weights."""

    dtypes = ("float32", "float16")

    def __init__(self, ctx: Context) -> None:
        """Build and optimise the network."""
        super().__init__(ctx)
        if ctx.dtype == "float16" and ctx.device.type == "cuda" and ctx.params.fp16_accumulate:
            self.gemm = fp16_accumulate

    def __call__(self, image: torch.Tensor, layers: tuple[int, ...]) -> dict[str, torch.Tensor]:
        """Features of a resized, normalised batch."""
        return self._run(image, layers)


class Kernels(_TimmFeatures):
    """Our kernels through `fastforks._optimize` (fused fp16 GEMMs with bias / GELU / residual epilogues, RoPE kernel).

    Small calls without hidden states (`_GRAPH_WORK`) replay a CUDA graph per input shape, captured on its second
    call (launch bound there); `release_graphs()` frees them. Every small call, graph or eager, with or without hidden
    states, runs its GEMMs of at most `conv_ops.LIN_GRAPH_ROWS` rows on cuBLAS (`graph_sized`; faster in a replay,
    slower eagerly than the fused kernels), so an image's features do not depend on the graph cache or on `layers`.
    """

    def __init__(self, ctx: Context) -> None:
        """Build and optimise the network; set up the graph cache."""
        super().__init__(ctx)
        # fused fp16-accumulate GEMMs and cuBLAS with fp16 accumulation (GeForce: 2x the fp32-accumulate rate), or
        # every GEMM through cuBLAS with fp32 accumulation (the portable precision inside the CUDA graphs)
        self.gemm = cublas_gemms if ctx.params.fp32_accumulate else fp16_accumulate
        self.patch: int = self.net.patch_embed.patch_size[0]
        self.graphs: GraphCache | None = GraphCache(self._captured, graphs=_GRAPHS)  # None: no graphs (tests)

    def _small(self, image: torch.Tensor) -> bool:
        """Whether the call is small enough for a graph: `_GRAPH_WORK` activations over the patch tokens."""
        b, _, h, w = image.shape
        return image.is_cuda and b * h * w * self.hidden_size <= _GRAPH_WORK * self.patch**2

    def _forward(
        self, image: torch.Tensor, layers: tuple[int, ...], used: list[tuple[Any, Any]] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, list]:
        """As the base class; a small call dispatches its GEMMs as its graph does (`graph_sized`)."""
        with graph_sized() if self._small(image) else contextlib.nullcontext():
            return super()._forward(image, layers, used)

    def _joined(self, image: torch.Tensor, used: list[tuple[Any, Any]] | None = None) -> torch.Tensor:
        """`[B, N * C + P]` float32: `last_hidden_state`, then `pooler_output`, flattened (one tensor for the graph)."""
        last, pooled, _ = self._forward(image, (), used)
        return torch.cat([last.flatten(1), pooled], 1).float()

    def _captured(self, image: torch.Tensor) -> tuple[torch.Tensor, list[tuple[Any, Any]]]:
        """The graph's forward: `_joined` and the position / RoPE tables it read.

        The graph replays with the tables' addresses, and `_Walk` drops a table when `_TABLES` other grids were used
        since: the cache entry (this output) keeps them alive as long as the graph.
        """
        used: list[tuple[Any, Any]] = []
        return self._joined(image, used), used

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        if self.graphs is not None:
            self.graphs.release()

    def __call__(self, image: torch.Tensor, layers: tuple[int, ...]) -> dict[str, torch.Tensor]:
        """Features of a resized, normalised batch."""
        if layers:
            return self._run(image, layers)
        j = None
        if self.graphs is not None and self._small(image):
            key = (image.device, image.dtype, tuple(image.shape))
            j = self.graphs.run(key, image, post=lambda out: out[0].clone())
        if j is None:
            j = self._joined(image)
        p, c = self.patch, self.hidden_size
        n = self.num_prefix_tokens + (image.shape[-2] // p) * (image.shape[-1] // p)
        return {"last_hidden_state": j[:, : n * c].view(-1, n, c), "pooler_output": j[:, n * c :]}


class MetalKernels(Portable):
    """Apple MPS: the portable network with Perception Encoder's interleaved RoPE of q / k as one Metal kernel.

    The kernel (`vit_metal`) replaces the RoPE's dozen elementwise ops per block (float32 math, one rounding to the
    network's dtype, as portable); the GEMMs and the attention stay PyTorch's.
    """

    optimize_mode = "portable"
    module = Runner.module  # the Metal kernel is not exportable

    def __init__(self, ctx: Context) -> None:
        """Build and optimise the network (portable), then route its RoPE blocks through the Metal kernel.

        Raises:
            ValueError: The network has no Perception Encoder RoPE block (a card listing `vit_metal` for it is wrong).
        """
        super().__init__(ctx)
        from fastforks.kernels import vit_metal  # noqa: PLC0415 - needs the kernel library

        lib = vit_metal.load(ctx.device)

        def qk_rope(qkv: torch.Tensor, rope: torch.Tensor, npt: int) -> tuple[torch.Tensor, torch.Tensor]:
            return vit_metal.pe_rope(lib, qkv, rope, npt)

        if not transformer.pe_rope(self.net, qk_rope):
            msg = f"{ctx.variant.ref}: no Perception Encoder RoPE block for the vit_metal kernel"
            raise ValueError(msg)
