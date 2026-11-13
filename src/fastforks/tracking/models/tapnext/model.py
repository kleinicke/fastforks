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
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.checkpoint import check_fits
from fastforks._core.half import fp16_accumulate
from fastforks.tracking._runner import Forward, NoParams, Runner

from .net.tapnext import SIZE, BlockDiagonalLinear, EncoderBlock, MLPBlock, RecurrentBlock, TAPNext, from_jax

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks.tracking._runner import Context


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


def track(  # noqa: PLR0913, PLR0917 - the runner call's arguments plus the autocast dtype
    net: TAPNext,
    frames: Callable[[int], torch.Tensor],
    n: int,
    size: tuple[int, int],
    queries: torch.Tensor,
    amp: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    """`{"tracks": [T, Q, 2], "visibility": [T, Q]}` of frames `frames(t)` `[1, 3, H, W]` in `[0, 1]`."""
    h, w = size
    scale = torch.tensor((SIZE / w, SIZE / h), device=queries.device)  # (x, y)
    qt = queries[:, 0]
    tracks, vis = [], []
    region = torch.autocast(queries.device.type, amp) if amp is not None else contextlib.nullcontext()
    with region:
        point = net.query_embedding(((queries[:, 1:] + 0.5) * scale).flip(-1))
        state = None
        for t in range(n):
            video = frames(t)
            if video.shape[-2:] != (SIZE, SIZE):
                video = F.interpolate(video, (SIZE, SIZE), mode="bilinear", align_corners=False, antialias=True)
            yx, logit, _, state = net(video * 2 - 1, qt, point, t, state)
            tracks.append(yx.flip(-1) / scale - 0.5)
            vis.append(logit.sigmoid())
    return {"tracks": torch.cat(tracks), "visibility": torch.cat(vis)}


class Portable(Runner):
    """Plain PyTorch; fp16 autocast on CUDA as upstream's evaluation (see the module docstring)."""

    dtypes = ("float32", "float16")
    Params = NoParams

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        net = TAPNext()
        sd = load_weights(ctx)
        check_fits(net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
        net.load_state_dict(sd)
        net.eval().requires_grad_(requires_grad=False).to(ctx.device)
        self.amp = torch.float16 if ctx.dtype == "float16" else None
        if self.amp is not None:
            cast_gemm_weights(net, self.amp)
        self.net = net

    def __call__(
        self, frames: Callable[[int], torch.Tensor], n: int, size: tuple[int, int], queries: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Tracks and visibility of the queries."""
        return track(self.net, frames, n, size, queries, self.amp)

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


def _recurrent(m: RecurrentBlock) -> Callable[..., Any]:
    """`RecurrentBlock.forward` with fused GEMMs (GELU-tanh epilogue on the gate branch)."""
    from fastforks._optimize.conv_ops import lin  # noqa: PLC0415 - loads the fvm_conv kernels (kernels mode only)

    wy, by, wx, bx = _w4(m.linear_y.weight), _t(m.linear_y.bias), _w4(m.linear_x.weight), _t(m.linear_x.bias)
    wo, bo = _w4(m.linear_out.weight), _t(m.linear_out.bias)

    def forward(x: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor] | None) -> Any:  # noqa: ANN401 - upstream's
        x16 = x.half()
        y = lin(x16, wy, by, act=GELU_TANH)
        return m.temporal(lin(x16, wx, bx).float(), cache, y, lambda v: lin(v.half(), wo, bo))

    return forward


def _mlp(m: MLPBlock) -> Callable[[torch.Tensor], torch.Tensor]:
    """`MLPBlock.forward`: both up-projections in one fused GEMM, the gate's GELU-tanh before the down-projection."""
    from fastforks._optimize.conv_ops import lin  # noqa: PLC0415 - loads the fvm_conv kernels (kernels mode only)

    w = _t(m.ffw_up.w).detach()  # [2, C, 4C]
    wu, bu = _w4(torch.cat([w[0], w[1]], 1).T), _t(m.ffw_up.b).detach().flatten().half()
    wd, bd = _w4(m.ffw_down.weight), _t(m.ffw_down.bias)

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

    def __call__(
        self, frames: Callable[[int], torch.Tensor], n: int, size: tuple[int, int], queries: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Tracks and visibility of the queries."""
        with fp16_accumulate():
            return super().__call__(frames, n, size, queries)

    def module(self) -> torch.nn.Module:
        """No exportable module: the GEMMs are kernel launches."""
        return Runner.module(self)


RUNNERS = {"portable": Portable, "kernels": Kernels}
