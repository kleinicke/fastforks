"""TAPNext runners: weights -> network and the online forward pass.

TAPNext runs online as upstream, one frame at a time, carrying the RG-LRU and temporal-conv state (faster and
smaller than several frames per pass on an RTX 5060 Ti: 24 frames, 256 queries, fp16 portable 517 ms one frame per
pass, 717 ms 16 frames per pass). Frames are resized to
256 x 256 (bilinear, antialiased; the TAP-Vid protocol resizes before, so the eval feeds 256 x 256 frames) and scaled
to `[-1, 1]`. A query `(t, x, y)` in input pixels becomes a point at `(y, x)` in the continuous 256 x 256 frame; the
tracks come back the same way and the visibility is the sigmoid of upstream's visible logit (upstream: visible where
the logit is > 0).

On CUDA the validated dtype is float16 as upstream's evaluation (`torch.autocast("cuda", torch.float16)` around the
model): the linear / conv / attention GEMMs in fp16, the residual stream, the norms, the temporal conv and the RG-LRU
scan in float32. The GEMM weights are stored in fp16 (autocast would cast them on every call).
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.checkpoint import check_fits
from fastforks._core.half import fp16_accumulate
from fastforks._core.model import eager_in_compile
from fastforks._core.upload import device_tensor
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import GraphCache
from fastforks.tracking._runner import Forward, Runner

from .net.tapnext import (
    CONV_WIDTH,
    DEPTH,
    PATCH,
    SIZE,
    WIDTH,
    BlockDiagonalLinear,
    EncoderBlock,
    MLPBlock,
    RecurrentBlock,
    TAPNext,
    from_jax,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks.tracking._runner import Context

    Step = Callable[..., tuple[torch.Tensor, torch.Tensor, list]]


def load_weights(ctx: Context) -> dict[str, torch.Tensor]:
    """The variant's state dict: upstream's JAX `.npz` (BootsTAPNext) or the Lightning checkpoint (TAPNext++)."""
    if ctx.variant.weights[0].name.endswith(".npz"):
        return from_jax(ctx.state_dict(strip=""))
    return ctx.state_dict(key="state_dict", strip="tapnext.")


def cast_gemm_weights(net: TAPNext, dtype: torch.dtype) -> None:
    """GEMM weights in `dtype` (autocast casts them to it on every call anyway).

    Norms, tokens, embeddings, the temporal conv, the RG-LRU decay and the biases added after an einsum / matmul stay
    float32 (as upstream's sums).
    """
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.to(dtype)
        elif isinstance(m, BlockDiagonalLinear):
            m.w.data = m.w.data.to(dtype)
        elif isinstance(m, MLPBlock):
            w = _t(m.ffw_up.w)
            w.data = w.data.to(dtype)
    for m in net.modules():
        for name in ("in_proj_weight", "in_proj_bias"):
            if hasattr(m, name):
                p = _t(getattr(m, name))
                p.data = p.data.to(dtype)


@dataclass(frozen=True)
class Params:
    """Card `[params]`."""

    cuda_graphs: bool  # kernels: replay every frame after the first from a CUDA graph (option)


@eager_in_compile
def _read_frame(frames: Callable[[int], torch.Tensor], index: int) -> torch.Tensor:
    """Read and validate a public video frame outside Dynamo's tensor-only network graph."""
    return frames(index)


def track(  # noqa: PLR0913, PLR0917 - the runner call's arguments plus the autocast dtype
    net: TAPNext,
    frames: Callable[[int], torch.Tensor],
    n: int,
    size: tuple[int, int],
    queries: torch.Tensor,
    amp: torch.dtype | None,
    *,
    eager_frames: bool = False,
    step: Step | None = None,
) -> dict[str, torch.Tensor]:
    """`{"tracks": [T, Q, 2], "visibility": [T, Q]}` of frames `frames(t)` `[1, 3, H, W]` in `[0, 1]`.

    `step(video, qt, point, t, state) -> (yx, visible logit, state)` replaces the network call of one frame.
    """
    h, w = size
    scale = device_tensor((SIZE / w, SIZE / h), device=queries.device)  # (x, y)
    qt = queries[:, 0]
    tracks, vis = [], []
    region = torch.autocast(queries.device.type, amp) if amp is not None else contextlib.nullcontext()
    with region:
        point = net.query_embedding(((queries[:, 1:] + 0.5) * scale).flip(-1))
        state = None
        for t in range(n):
            video = _read_frame(frames, t) if eager_frames else frames(t)
            if video.shape[-2:] != (SIZE, SIZE):
                video = F.interpolate(video, (SIZE, SIZE), mode="bilinear", align_corners=False, antialias=True)
            if step is None:
                yx, logit, _, state = net(video * 2 - 1, qt, point, t, state)
            else:
                yx, logit, state = step(video * 2 - 1, qt, point, t, state)
            tracks.append(yx.flip(-1) / scale - 0.5)
            vis.append(logit.sigmoid())
    return {"tracks": torch.cat(tracks), "visibility": torch.cat(vis)}


class Portable(Runner):
    """Plain PyTorch; fp16 autocast on CUDA as upstream's evaluation (see the module docstring)."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        net = TAPNext()
        sd = load_weights(ctx)
        check_fits(net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
        net.load_state_dict(sd)
        net.eval().requires_grad_(requires_grad=False)
        net.to_inference_device(ctx.device)
        self.amp = torch.float16 if ctx.dtype == "float16" else None
        if self.amp is not None:
            cast_gemm_weights(net, self.amp)
        self.net = net

    def __call__(
        self, frames: Callable[[int], torch.Tensor], n: int, size: tuple[int, int], queries: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Tracks and visibility of the queries."""
        # A public video can hold numpy or PIL frames: keep conversion outside Dynamo while compiling the network.
        return track(self.net, frames, n, size, queries, self.amp, eager_frames=torch.compiler.is_compiling())

    def module(self) -> torch.nn.Module:
        """The network with the resizing and the frame loop; `forward(video [T, 3, H, W], queries [Q, 3])`."""
        amp = self.amp

        def fn(net: TAPNext, video: torch.Tensor, queries: torch.Tensor) -> dict[str, torch.Tensor]:
            size = (video.shape[-2], video.shape[-1])
            return track(net, lambda t: video[t : t + 1], video.shape[0], size, queries, amp)

        return Forward(self.net, fn)


GELU, GELU_TANH = 5, 6  # fused-epilogue activation codes (fastforks._optimize.conv_ops)


def _w4(w: torch.Tensor) -> torch.Tensor:
    """A Linear weight `[out, in]` as the fused GEMM's fp16 `[out, in, 1, 1]` channels-last filter."""
    return w.detach().half().reshape(*w.shape, 1, 1).contiguous(memory_format=torch.channels_last)


def _t(x: object) -> torch.Tensor:
    """A submodule attribute that holds a tensor (`nn.Module.__getattr__` is typed `Tensor | Module`)."""
    return cast("torch.Tensor", x)


class _RingState(NamedTuple):
    """A block's hidden state and convolution history in the graph's fixed buffers."""

    hidden: torch.Tensor
    history: torch.Tensor
    order: torch.Tensor
    first: torch.Tensor


def _temporal_graph(m: RecurrentBlock, x: torch.Tensor, y: torch.Tensor, state: _RingState) -> torch.Tensor:
    """Advance the fixed-address state for one captured recurrent frame."""
    h, history, order, first = state
    previous = history.index_select(0, order)
    history.index_copy_(0, order[:1], x)
    w, b = _t(m.conv_1d.w), _t(m.conv_1d.b)
    # Match the eager convolution's addition order and keep the state and gate arithmetic in float32.
    conv = b + (sum(w[i] * previous[i : i + 1] for i in range(CONV_WIDTH - 1)) + w[-1] * x)
    cell = m.rg_lru
    gate = torch.sigmoid(cell.input_gate(conv))
    log_a = -8.0 * torch.sigmoid(cell.a_gate(conv)) * F.softplus(cell.a_param.float())
    a = log_a.exp()
    gain = torch.where(first, 1.0, torch.sqrt(1 - (2 * log_a).exp()))
    h.mul_(a[0]).add_((conv * gate * gain)[0])
    return (h[None] * y).to(y.dtype)


def _reset_state(inputs: tuple[torch.Tensor, ...]) -> None:
    """Reset cached recurrent state in place when starting a new clip."""
    torch._foreach_zero_(inputs[5:])  # batched GPU fill of the recurrent buffers


def _recurrent(m: RecurrentBlock) -> Callable[..., Any]:
    """`RecurrentBlock.forward` with fused GEMMs (GELU-tanh epilogue on the gate branch)."""
    from fastforks._optimize.conv_ops import lin  # noqa: PLC0415 - loads the fvm_conv kernels (kernels mode only)

    wy, by, wx, bx = _w4(m.linear_y.weight), _t(m.linear_y.bias), _w4(m.linear_x.weight), _t(m.linear_x.bias)
    wo, bo = _w4(m.linear_out.weight), _t(m.linear_out.bias)

    def forward(
        x: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor] | _RingState | None, first: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | _RingState]:
        x16 = x.half()
        y = lin(x16, wy, by, act=GELU_TANH)
        x = lin(x16, wx, bx).float()
        if isinstance(cache, _RingState):
            return lin(_temporal_graph(m, x, y, cache).half(), wo, bo), cache
        return m.temporal(x, cache, y, lambda v: lin(v.half(), wo, bo), first)

    return forward


def _mlp(m: MLPBlock) -> Callable[[torch.Tensor], torch.Tensor]:
    """`MLPBlock.forward`: both up-projections in one fused GEMM, the gate's GELU-tanh before the down-projection."""
    from fastforks._optimize.conv_ops import lin  # noqa: PLC0415 - loads the fvm_conv kernels (kernels mode only)

    w = _t(m.ffw_up.w).detach()  # [2, C, 4C]
    wu, bu = _w4(torch.cat([w[0], w[1]], 1).T), _t(m.ffw_up.b).detach().flatten().half()
    wd, bd = _w4(m.ffw_down.weight), _t(m.ffw_down.bias)
    cast("torch.nn.Parameter", m.ffw_up.w).data = w.new_empty(0)  # the merged filter replaces it

    def forward(x: torch.Tensor) -> torch.Tensor:
        gate, up = lin(x.half(), wu, bu).chunk(2, -1)
        return lin(F.gelu(gate, approximate="tanh") * up, wd, bd)

    return forward


def _encoder(m: EncoderBlock) -> Callable[[torch.Tensor], torch.Tensor]:
    """`EncoderBlock.forward` with fused GEMMs (exact GELU epilogue on the MLP's first layer) and SDPA."""
    from fastforks._optimize.conv_ops import lin  # noqa: PLC0415 - loads the fvm_conv kernels (kernels mode only)

    a = m.self_attention
    fc1, fc2 = cast("nn.Linear", m.mlp[0]), cast("nn.Linear", m.mlp[3])
    wqkv, wo, w1, w2 = _w4(a.in_proj_weight), _w4(a.out_proj.weight), _w4(fc1.weight), _w4(fc2.weight)
    bqkv, bo, b1, b2 = _t(a.in_proj_bias), _t(a.out_proj.bias), _t(fc1.bias), _t(fc2.bias)

    def forward(x: torch.Tensor) -> torch.Tensor:
        t, n, c = x.shape
        q, k, v = lin(m.ln_1(x).half(), wqkv, bqkv).view(t, n, 3, a.heads, -1).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(t, n, c)
        x = x + lin(o, wo, bo)
        return x + lin(lin(m.ln_2(x).half(), w1, b1, act=GELU), w2, b2)

    return forward


class Kernels(Portable):
    """The fp16 portable pass with fused GEMMs (bias / GELU / GELU-tanh epilogues) and fp16 accumulation."""

    def __init__(self, ctx: Context) -> None:
        """The portable fp16 network with the fused-GEMM forwards (fp16 filters captured in closures)."""
        super().__init__(ctx)
        for m in self.net.modules():
            if isinstance(m, RecurrentBlock):
                m.forward = _recurrent(m)  # ty: ignore[invalid-assignment]  # per-instance override
            elif isinstance(m, MLPBlock):
                m.forward = _mlp(m)  # ty: ignore[invalid-assignment]  # per-instance override
            elif isinstance(m, EncoderBlock):
                m.forward = _encoder(m)  # ty: ignore[invalid-assignment]  # per-instance override
        # One graph per query count updates the state in place. When it exists, the first frame resets those same
        # buffers, skipping the eager first frame's separate state allocation. `_lock` isolates each clip.
        self.steps = GraphCache(self._graph_step, graphs=2) if ctx.params.cuda_graphs else None
        self._lock = threading.Lock()
        self._ring_origin = 0  # the first replayed frame in the current clip

    def _graph_step(
        self,
        video: torch.Tensor,
        qt: torch.Tensor,
        point: torch.Tensor,
        t0: torch.Tensor,
        order: torch.Tensor,
        *state: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """One frame, with the same ring order and first-frame flag for every block."""
        first = t0 == 0
        cache = [_RingState(h, history, order, first) for h, history in zip(state[::2], state[1::2], strict=True)]
        yx, logit, _, _ = self.net(video, qt, point, t0, cache)
        return yx, logit

    def _step(
        self, video: torch.Tensor, qt: torch.Tensor, point: torch.Tensor, t: int, state: list | None
    ) -> tuple[torch.Tensor, torch.Tensor, list]:
        """`track`'s frame step: from the frame's graph once it is captured, else eager."""
        if self.steps is not None:
            key = (video.device, tuple(video.shape), tuple(qt.shape), torch.is_autocast_enabled(video.device.type))
            static = self.steps.static_inputs(key)
            if static is not None or (state is not None and not self.steps.defer_first_request(key)):
                if state is None or static is None or state[0][0] is not static[5]:
                    # Start rotating when entering a graph from eager state, or on a fresh clip.
                    self._ring_origin = t
                history = CONV_WIDTH - 1
                phase = t - self._ring_origin
                order = device_tensor(tuple((phase + i) % history for i in range(history)), device=video.device)
                t0 = torch.full((), t, dtype=qt.dtype, device=qt.device)
                if state is None:
                    assert static is not None  # noqa: S101 - the first frame can only replay an existing graph
                    inputs = static[5:]
                else:
                    inputs = tuple(v for pair in state for v in pair)
                out = self.steps.run(
                    key, video, qt, point, t0, order, *inputs, before_replay=_reset_state if state is None else None
                )
                static = self.steps.static_inputs(key)
                if out is not None and static is not None:
                    return out[0], out[1], list(zip(static[5::2], static[6::2], strict=True))
        yx, logit, _, state = self.net(video, qt, point, t, state)
        return yx, logit, state

    def release_graphs(self) -> None:
        """Free the CUDA graphs, their pool and the state buffers."""
        if self.steps is not None:
            with self._lock:
                self.steps.release()

    def __call__(
        self, frames: Callable[[int], torch.Tensor], n: int, size: tuple[int, int], queries: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Tracks and visibility of the queries."""
        with self._lock, fp16_accumulate():
            return track(self.net, frames, n, size, queries, self.amp, step=self._step)

    def module(self) -> torch.nn.Module:
        """No exportable module: the GEMMs are kernel launches."""
        return Runner.module(self)


class Steps(Portable):
    """The MLX pass: one exported step graph (a frame in, its tracks and the new state out) run once per frame.

    One graph serves every video length and is exported once (about 2 s; the whole 24-frame pass as one graph was
    32 s of export, again for every other length). Frame 0 runs the same graph from a zero state with `first` set,
    which is upstream's reset (`RGLRU.forward`). The state stays on MLX between frames (`MlxModule.raw`; copying it
    out and back, 189 MB per frame, was 0.7 s of a 5 s call). The query tokens are computed once, in PyTorch.
    """

    def _step(  # noqa: PLR0913, PLR0917 - the step graph's tensors
        self,
        frame: torch.Tensor,
        t: torch.Tensor,
        qt: torch.Tensor,
        point: torch.Tensor,
        first: torch.Tensor,
        state: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if frame.shape[-2:] != (SIZE, SIZE):
            frame = F.interpolate(frame, (SIZE, SIZE), mode="bilinear", align_corners=False, antialias=True)
        yx, logit, _, state = self.net(frame * 2 - 1, qt, point, t, state, first)
        return yx, logit.sigmoid(), state

    def __call__(
        self, frames: Callable[[int], torch.Tensor], n: int, size: tuple[int, int], queries: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Tracks and visibility of the queries."""
        h, w = size
        scale = torch.tensor((SIZE / w, SIZE / h))  # (x, y)
        qt = queries[:, 0]
        point = self.net.query_embedding(((queries[:, 1:] + 0.5) * scale).flip(-1))
        tokens = (SIZE // PATCH) ** 2 + queries.shape[0]
        state = [(torch.zeros(tokens, WIDTH), torch.zeros(CONV_WIDTH - 1, tokens, WIDTH))] * DEPTH
        from fastforks._core.mlx import to_torch  # noqa: PLC0415 - MLX only

        step = cast("Any", self._step).raw  # the `MlxModule` of `mlx_wrap.port(methods=("_step",))`
        tracks, vis = [], []
        for t in range(n):
            yx, v, state = step(frames(t), torch.tensor(float(t)), qt, point, torch.tensor(t == 0), state)
            tracks.append(to_torch(yx).flip(-1) / scale - 0.5)
            vis.append(to_torch(v))
        return {"tracks": torch.cat(tracks), "visibility": torch.cat(vis)}


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(
        Steps, attrs=(), methods=("_step",), release=True, pipeline=True, fp32_inputs=("point", "state")
    ),
}
