"""LocoTrack runners: weights -> network and the offline forward pass.

LocoTrack sees the whole video at once. Frames are resized to 256 x 256 (bilinear, antialiased; the TAP-Vid protocol
resizes before, so the eval feeds 256 x 256 frames), scaled to `[-1, 1]` and encoded `FRAME_CHUNK` at a time; the
queries run `QUERY_CHUNK` at a time (as upstream's inference; each query is computed on its own). A query `(t, x, y)`
in input pixels becomes the point `(t, y, x)` in the continuous 256 x 256 frame; the tracks come back the same way.
The visibility is upstream's `(1 - sigmoid(occlusion)) * (1 - sigmoid(expected_dist))` (visible where it is > 0.5).
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import check_fits
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import override
from fastforks._core.tracing import EXPORTING_FOR_MLX
from fastforks._core.upload import device_tensor
from fastforks.kernels.locotrack_sample import neighborhood
from fastforks.tracking._mlx import port as mlx_port
from fastforks.tracking._runner import Forward, NoParams, Runner

from .net.locotrack import (
    SIZE,
    SUPPORT,
    CMDTop,
    Conv2dSamePadding,
    Features,
    LocoTrack,
    MultiHeadAttention,
    PIPSTransformer,
    alibi_bias,
    posenc,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks.tracking._runner import Context

FRAME_CHUNK = 16  # fixed: memory chunking of the feature pass, not an architecture value
MPS_FRAME_CHUNK = 4  # fixed: the same on MPS, where smaller chunks cost no time but save memory
QUERY_CHUNK = 64  # fixed: upstream's inference chunk
# the MLX export unrolls every chunk (256 bench queries: 4 x 64); 128 halves the refinement's trace, export 5.4 -> 3.4 s
# at 4 % more warm time (256: 2.3 s, 8 %). Each query is computed on its own, so the chunk does not change the tracks.
MLX_QUERY_CHUNK = 128  # fixed: export trace size of the MLX port, not an architecture value
GELU = 5  # fixed: fused-epilogue activation code of exact GELU (fastforks._optimize.conv_ops)


def _join(chunks: list[torch.Tensor]) -> torch.Tensor:
    """The chunks as one tensor, without keeping the chunks alive next to it (the per-frame features: peak memory)."""
    out = chunks[0] if len(chunks) == 1 else torch.cat(chunks)
    chunks.clear()
    return out


def build(ctx: Context) -> LocoTrack:
    """The variant's network (base or small, from the mixer width of the weights)."""
    sd = ctx.state_dict(key="state_dict", strip="model.")
    size = "base" if sd["torch_pips_mixer.input_proj.weight"].shape[0] == 384 else "small"  # noqa: PLR2004
    net = LocoTrack(size)
    check_fits(net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
    net.load_state_dict(sd)
    return net


def track(  # noqa: PLR0913, PLR0917 - the runner call's arguments plus the autocast dtype
    net: LocoTrack,
    frames: Callable[[int], torch.Tensor],
    n: int,
    size: tuple[int, int],
    queries: torch.Tensor,
    amp: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    """`{"tracks": [T, Q, 2], "visibility": [T, Q]}` of frames `frames(t)` `[1, 3, H, W]` in `[0, 1]`."""
    h, w = size
    scale = device_tensor((SIZE / w, SIZE / h), device=queries.device)  # (x, y)
    region = torch.autocast(queries.device.type, amp) if amp is not None else contextlib.nullcontext()
    grids: list[list[torch.Tensor]] = [[], [], []]
    with region:
        step = MPS_FRAME_CHUNK if queries.device.type == "mps" else FRAME_CHUNK
        for s in range(0, n, step):
            video = torch.cat([frames(t) for t in range(s, min(n, s + step))])
            if video.shape[-2:] != (SIZE, SIZE):
                video = F.interpolate(video, (SIZE, SIZE), mode="bilinear", align_corners=False, antialias=True)
            for g, f in zip(grids, net.frame_features(video * 2 - 1), strict=True):
                g.append(f)
            del video
        q = torch.cat([queries[:, :1], ((queries[:, 1:] + 0.5) * scale).flip(-1)], 1)[None]  # (t, y, x)
        f = net.features(tuple(_join(g) for g in grids), q)
        points, vis = [], []
        chunk = MLX_QUERY_CHUNK if EXPORTING_FOR_MLX.get() else QUERY_CHUNK
        for s in range(0, q.shape[1], chunk):
            p, occ, expd = net.track(f, q, slice(s, s + chunk))
            points.append(p[0])
            vis.append(((1 - occ.float().sigmoid()) * (1 - expd.float().sigmoid()))[0])
    tracks = torch.cat(points).float().transpose(0, 1) / scale - 0.5
    return {"tracks": tracks, "visibility": torch.cat(vis).T}


class Portable(Runner):
    """Plain PyTorch."""

    dtypes = ("float32", "float16")
    Params = NoParams

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = build(ctx).eval().requires_grad_(requires_grad=False).to(ctx.device)
        self.amp = torch.float16 if ctx.dtype == "float16" else None

    def __call__(
        self, frames: Callable[[int], torch.Tensor], n: int, size: tuple[int, int], queries: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Tracks and visibility of the queries."""
        return track(self.net, frames, n, size, queries, self.amp)

    def module(self) -> torch.nn.Module:
        """The network with the resizing; `forward(video [T, 3, H, W], queries [Q, 3])`."""
        amp = self.amp

        def fn(net: LocoTrack, video: torch.Tensor, queries: torch.Tensor) -> dict[str, torch.Tensor]:
            size = (video.shape[-2], video.shape[-1])
            return track(net, lambda t: video[t : t + 1], video.shape[0], size, queries, amp)

        return Forward(self.net, fn)


def _w4(w: torch.Tensor) -> torch.Tensor:
    """A Linear weight `[out, in]` as the fused GEMM's fp16 `[out, in, 1, 1]` channels-last filter."""
    return w.detach().half().reshape(*w.shape, 1, 1).contiguous(memory_format=torch.channels_last)


def _b(*layers: torch.nn.Linear) -> torch.Tensor:
    """The layers' biases, concatenated, in fp16."""
    return torch.cat([cast("torch.Tensor", m.bias).detach() for m in layers]).half()


def _mixer(m: PIPSTransformer) -> Callable[[torch.Tensor], torch.Tensor]:
    """`PIPSTransformer.forward` on fused GEMMs in fp16.

    One q/k/v GEMM, the MLP with an exact-GELU epilogue, the ALiBi bias cached per length; the residual stream stays
    float32. The input projection (854 features, which the fused GEMM does not take) stays on cuBLAS.
    """
    from fastforks._optimize.conv_ops import lin  # noqa: PLC0415 - loads the fvm_conv kernels (kernels mode only)

    wi, bi = m.input_proj.weight.detach().half(), _b(m.input_proj)  # 854 inputs: not a multiple of 8, cuBLAS
    wo, bo = m.output_proj.weight.detach()[:4].half(), m.output_proj.bias.detach()[:4].half()
    layers = []
    for layer in cast("list[torch.nn.ModuleDict]", list(m.transformer.layers)):
        a = cast("MultiHeadAttention", layer["attn"])
        up, down = (cast("torch.nn.Linear", layer["dense"][i]) for i in (0, 2))  # ty: ignore[not-subscriptable]
        qkv = (a.query_proj, a.key_proj, a.value_proj)
        w = (_w4(torch.cat([p.weight for p in qkv])), _b(*qkv), _w4(a.final_proj.weight), _b(a.final_proj))
        layers.append(
            (layer["layer_norm1"], *w, layer["layer_norm2"], _w4(up.weight), _b(up), _w4(down.weight), _b(down))
        )
    heads, ln_out, biases = m.heads, m.transformer.ln_out, {}

    def forward(x: torch.Tensor) -> torch.Tensor:
        n, t, _ = x.shape
        if t not in biases:
            biases[t] = alibi_bias(heads, t, x.device).half()
        h = F.linear(x.half(), wi, bi).float()
        for ln1, wqkv, bqkv, wf, bf, ln2, w1, b1, w2, b2 in layers:
            q, k, v = lin(ln1(h).half(), wqkv, bqkv).view(n, t, 3, heads, -1).permute(2, 0, 3, 1, 4)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=biases[t])
            h = h + lin(y.transpose(1, 2).reshape(n, t, -1), wf, bf)
            h = h + lin(lin(ln2(h).half(), w1, b1, act=GELU), w2, b2)
        return F.linear(ln_out(h).half(), wo, bo)

    return forward


def _refine(  # noqa: PLR0913, PLR0917 - same signature as LocoTrack.refine
    net: LocoTrack,
    supports: list[torch.Tensor],
    points: torch.Tensor,
    occ: torch.Tensor,
    expd: torch.Tensor,
    f: Features,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample the local grids directly into the fp16 layouts consumed by the three batched GEMMs."""
    b, n, t, _ = points.shape
    corrs = []
    for support, grid, cmdtop in zip(supports, f.grids, net.cmdtop, strict=True):
        samples = neighborhood(grid, points).view(b * n, t * SUPPORT**2, -1)
        query = support.reshape(b * n, SUPPORT**2, -1).transpose(1, 2)
        corr = torch.bmm(samples, query)
        corrs.append(cmdtop(corr.reshape(b * n * t, SUPPORT, SUPPORT, SUPPORT, SUPPORT)).view(b, n, t, -1))
    delta = points[..., :-1, :] - points[..., 1:, :]
    rel = torch.cat([F.pad(delta, (0, 0, 0, 1)), F.pad(-delta, (0, 0, 1, 0))], -1) / SIZE
    x = torch.cat([occ[..., None], expd[..., None], *corrs, posenc(rel)], -1)
    update = net.torch_pips_mixer(x.reshape(b * n, t, -1)).reshape(b, n, t, -1).float()
    return points + update[..., :2], occ + update[..., 2], expd + update[..., 3]


def _same_conv(conv: Conv2dSamePadding, x: torch.Tensor) -> torch.Tensor:
    """Use the convolution's implicit symmetric halo and explicitly pad only a trailing odd pixel."""
    h, w = x.shape[-2:]
    sh, sw = conv.stride
    kh, kw = conv.kernel_size
    ph, pw = max(kh - (h - 1) % sh - 1, 0), max(kw - (w - 1) % sw - 1, 0)
    if ph % 2 or pw % 2:
        x = F.pad(x, (0, pw % 2, 0, ph % 2))
    return F.conv2d(x, conv.weight, conv.bias, conv.stride, (ph // 2, pw // 2))


def _last_corr_conv(conv: Conv2dSamePadding, x: torch.Tensor) -> torch.Tensor:
    """The last 2x2 CMDTop filter covers the entire 2x2 input and produces one feature vector."""
    if x.shape[-2:] != conv.kernel_size:
        msg = "LocoTrack CMDTop final convolution must span the entire correlation feature map"
        raise ValueError(msg)
    return F.linear(x.flatten(1), conv.weight.flatten(1), conv.bias)[:, :, None, None]


class Kernels(Portable):
    """The fp16 portable pass with the refinement transformer on fused-GEMM kernels and fp16 accumulation."""

    def __init__(self, ctx: Context) -> None:
        """The portable network with the kernel forward of the mixer (fp16 weights captured in a closure)."""
        super().__init__(ctx)
        for layer in self.net.modules():
            if isinstance(layer, Conv2dSamePadding):
                override(layer, _same_conv)
        for cmd in self.net.cmdtop:
            final = cast("torch.nn.Sequential", cast("CMDTop", cmd).conv[-1])
            override(final[0], _last_corr_conv)
        override(self.net, _refine, "refine")
        mixer = self.net.torch_pips_mixer
        mixer.forward = _mixer(mixer)  # ty: ignore[invalid-assignment]  # per-instance override

    def __call__(
        self, frames: Callable[[int], torch.Tensor], n: int, size: tuple[int, int], queries: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Tracks and visibility of the queries."""
        with fp16_accumulate():
            return super().__call__(frames, n, size, queries)

    def module(self) -> torch.nn.Module:
        """No exportable module: the mixer runs kernel launches."""
        return Runner.module(self)


RUNNERS = {"portable": Portable, "kernels": Kernels, "portable_mlx": mlx_port(Portable)}
