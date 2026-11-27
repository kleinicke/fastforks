"""MEMFOF runners: weights -> network and the forward pass of a pair."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import check_fits
from fastforks._core.half import half_weights
from fastforks._core.patch import override
from fastforks._core.volume_ops import corr_windows
from fastforks.flow._runner import Forward, Runner
from fastforks.kernels.flow_upsample import convex
from fastforks.shared import mlx_wrap

from .net.gma import Aggregate, Attention, rearrange
from .net.layer import ConvNextBlock
from .net.model import MEMFOF

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks._core.kernels import CubinModule
    from fastforks.flow._runner import Context

_REFINE_ROWS = 24  # fixed: rows per pointwise ConvNeXt chunk for large updates (memory bound, not numerics)
_REFINE_MIN_PIXELS = 16_384  # fixed: update size from which the pointwise work is chunked


@dataclass(frozen=True)
class Params:
    """Card `[params]` and `[options]`."""

    iters: int  # flow updates after the initial regression (upstream `iters`)
    scale: Literal[0, 1]  # upstream `get_val_scale`: 1 = the frames are upsampled 2x (its KITTI / Sintel evaluation)


def _net(ctx: Context) -> MEMFOF:
    net = MEMFOF()
    sd = ctx.state_dict()
    check_fits(net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
    net.load_state_dict(sd)
    return net.eval().to(ctx.device)


def _refine_in_rows(block: ConvNextBlock, image: torch.Tensor) -> torch.Tensor:
    """Keep the spatial convolutions whole and bound pointwise work for large CUDA updates."""
    if (
        torch.compiler.is_compiling()
        or torch.jit.is_tracing()
        or torch.is_grad_enabled()
        or image.shape[-2] * image.shape[-1] < _REFINE_MIN_PIXELS
    ):
        return ConvNextBlock.forward(block, image)
    features = block.dwconv(image).permute(0, 2, 3, 1)
    result = None
    layout = torch.channels_last if image.is_contiguous(memory_format=torch.channels_last) else torch.contiguous_format
    for start in range(0, features.shape[1], _REFINE_ROWS):
        part = features[:, start : start + _REFINE_ROWS]
        value = block.pwconv2(block.act(block.pwconv1(block.norm(part))))
        if block.gamma is not None:
            value = block.gamma * value
        stop = start + value.shape[1]
        value = image[..., start:stop, :] + value.permute(0, 3, 1, 2)
        value = block.final(value.contiguous(memory_format=layout))
        if result is None:
            result = torch.empty(
                (value.shape[0], value.shape[1], features.shape[1], features.shape[2]),
                device=value.device,
                dtype=value.dtype,
                memory_format=layout,
            )
        result[..., start:stop, :] = value
        del part, value
    assert result is not None  # noqa: S101 - the nonempty feature map yields at least one group
    return result


class _Scaled(Runner):
    """Upstream `scale_and_forward_flow`: the network sees the pair resized by `2^scale`, the flow is resized back."""

    def __init__(self, ctx: Context) -> None:
        """Check the card's `scale`."""
        super().__init__(ctx)
        if ctx.params.scale not in (0, 1):
            msg = f"{ctx.variant.ref}: card scale must be 0 or 1, not {ctx.params.scale!r}"
            raise ValueError(msg)

    def resize(self, x: torch.Tensor) -> torch.Tensor:
        """Double the pair (bilinear) when the card says `scale = 1`."""
        if self.ctx.params.scale == 0:
            return x
        return F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)

    def restore(self, flow: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:  # noqa: ARG002 - 2x then 0.5x: same size
        """Halve the flow field (bilinear) and its values when the card says `scale = 1`."""
        if self.ctx.params.scale == 0:
            return flow
        return F.interpolate(flow.float(), scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5


class Portable(_Scaled):
    """Plain PyTorch; on CUDA in fp16 the encoders and the update block run under autocast, the correlation in fp32."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        self.fp16 = ctx.dtype == "float16"
        if ctx.device.type == "cuda" and self.fp16:
            for block in self.net.update_block.refine:
                override(block, _refine_in_rows)
            # channels-last weights: cuDNN's fp16 kernels skip the NCHW<->NHWC conversion around each conv
            # (375x1242 99.0 -> 84.0 ms, peak 695 -> 488 MB RTX 5060 Ti, Windows); float32 stays NCHW
            self.net.to(memory_format=torch.channels_last)

    def _flow(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        return self.net.forward_pair(image1, image2, self.ctx.params.iters, fp16=self.fp16)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a padded batch."""
        return {"flow": self._flow(image1, image2)}

    def module(self) -> torch.nn.Module:
        """The network; returns the flow."""
        iters, fp16 = self.ctx.params.iters, self.fp16
        return Forward(self.net, lambda net, image1, image2: net.forward_pair(image1, image2, iters, fp16=fp16))


def _kernel_lookup(k: CubinModule, f1: torch.Tensor, f2: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """The 4-level 9 x 9 windows of the correlation computed on the fly by `raft_flow`: O(HW) memory, no volume.

    Upstream's pyramid halves fmap2 bilinearly (align_corners=False, scale 0.5), which is 2 x 2 average pooling, and
    pooling commutes with the dot product, so only the pooled fmap2 pyramid is kept (fp16). Operands are the fp16
    features with fp32 sums and bilinear weights; the windows are written in fp16, which the update block casts them to.
    """
    b, c, h, w = f1.shape
    a = f1.to(torch.float16).permute(0, 2, 3, 1).contiguous()
    x = f2.to(torch.float16).contiguous(memory_format=torch.channels_last)
    pyr = [x.permute(0, 2, 3, 1)]
    for _ in range(3):
        x = F.avg_pool2d(x, 2)
        pyr.append(x.permute(0, 2, 3, 1))
    side = 4 * 81

    def lookup(coords: torch.Tensor) -> torch.Tensor:
        xy = coords.permute(0, 2, 3, 1).contiguous()
        out = torch.empty(b, h, w, side, dtype=torch.float16, device=coords.device)
        for i in range(b):  # one warp per pixel, 8 warps per block
            k.launch(
                "corr_lookup_c1024_r4",
                (-(-h * w // 8), 1, 1),
                (256, 1, 1),
                a[i],
                *(p[i] for p in pyr),
                h,
                w,
                xy[i],
                out[i],
                side,
                1 / math.sqrt(c),
            )
        return out.permute(0, 3, 1, 2)

    return lookup


class Kernels(Portable):
    """The portable fp16 pass with the correlation windows computed on the fly (`raft_flow`): no all-pairs volume."""

    def __init__(self, ctx: Context) -> None:
        """Build the network and bind the lookup kernel."""
        super().__init__(ctx)
        half_weights(self.net)  # the encoders and each recurrent update reuse the autocast weights
        from fastforks.kernels import raft_flow  # noqa: PLC0415 - needs the kernel library

        self.lookup = functools.partial(_kernel_lookup, raft_flow.load(ctx.device))
        head = self.net.upsample_weight[-1]
        # Only the second 9 x 16² logits produce the forward flow returned by this task.
        forward_channels = 9 * 16**2
        if not isinstance(head, torch.nn.Conv2d) or head.bias is None or head.out_channels != 2 * forward_channels:
            msg = "Unexpected MEMFOF upsampling head"
            raise ValueError(msg)
        head.out_channels = forward_channels
        head.weight = torch.nn.Parameter(head.weight[forward_channels:].detach().clone(), requires_grad=False)
        head.bias = torch.nn.Parameter(head.bias[forward_channels:].detach().clone(), requires_grad=False)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a padded batch."""
        iters = self.ctx.params.iters
        return {
            "flow": self.net.forward_pair(image1, image2, iters, corr_fn=self.lookup, fp16=True, upsample_flow=convex)
        }

    def module(self) -> torch.nn.Module:
        """No exportable module: the lookup is a kernel launch."""
        return Runner.module(self)


_GMA_MATRIX_BYTES = 1 << 30  # fixed: float32 attention matrix from which the MLX pass recomputes it (memory bound)


def _windows(net: MEMFOF, f1: torch.Tensor, f2: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """`CorrBlock`'s windows without its all-pairs volumes: the pooled f2 pyramid and `corr_windows` (one kernel).

    Upstream halves f2 bilinearly (align_corners=False, scale 0.5), which is 2 x 2 average pooling, and pooling commutes
    with the dot product. At `scale = 1` (2160 x 3840 frames) each direction's volume alone is 4.2 GB in float32.
    """
    levels = [f2]
    for _ in range(net.corr_levels - 1):
        levels.append(F.avg_pool2d(levels[-1], 2))
    radius, scale = net.corr_radius, f1.shape[1] ** -0.5
    return lambda coords: corr_windows(f1, levels, coords, radius, scale)


def _attention_qk(att: Attention, fmap: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Upstream's attention matrix, or (scaled queries, keys) when it is large: `_aggregate_sdpa` recomputes it."""
    b, _, h, w = fmap.shape
    if b * att.heads * (h * w) ** 2 * 4 <= _GMA_MATRIX_BYTES:
        return Attention.forward(att, fmap)
    q, k = att.to_qk(fmap).chunk(2, dim=1)
    q, k = (rearrange(t, "b (h d) x y -> b h (x y) d", h=att.heads) for t in (q, k))
    return q * (att.scale * math.log(h * w, 3)), k


def _aggregate_sdpa(
    agg: Aggregate, attn: torch.Tensor | tuple[torch.Tensor, torch.Tensor], fmap: torch.Tensor
) -> torch.Tensor:
    """Upstream's aggregation; with (queries, keys) one attention op (MLX runs it in query blocks: no N^2 matrix)."""
    if not isinstance(attn, tuple):
        return Aggregate.forward(agg, attn, fmap)
    _, _, h, w = fmap.shape
    v = rearrange(agg.to_v(fmap), "b (h d) x y -> b h (x y) d", h=agg.heads)
    q, k = attn
    out = F.scaled_dot_product_attention(q, k, v.to(q.dtype), scale=1.0)
    out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
    if agg.project is not None:
        out = agg.project(out)
    return fmap + agg.gamma * out


class _Mlx(Portable):
    """The portable pass for the MLX replay: correlation windows without volumes, the large GMA matrix recomputed."""

    def __init__(self, ctx: Context) -> None:
        """Build the network and swap the volume and the attention matrix for their memory-bound forms."""
        super().__init__(ctx)
        override(self.net.att, _attention_qk)
        override(self.net.update_block.aggregator, _aggregate_sdpa)

    def _flow(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        corr_fn = functools.partial(_windows, self.net)
        return self.net.forward_pair(image1, image2, self.ctx.params.iters, corr_fn=corr_fn, fp16=self.fp16)


class _MlxPortable(_Scaled):
    """Apple MLX: the network written in MLX operations (`mlx_net`) from the checkpoint's tensors, one pair per call."""

    batch = 1  # one pair per call (`mlx_net`)
    dtypes = ("float32", "float16")
    Params = Params
    module = Runner.module  # the MLX network is not exportable
    CACHE_MB: ClassVar[dict[str, int]] = {"*": 1024}  # the buffer cache this mode asks for, by variant ('*': all)

    def __init__(self, ctx: Context) -> None:
        """The MLX network from the conversion cache (the first load converts the checkpoint's weights)."""
        super().__init__(ctx)
        from fastforks.shared import mlx_port  # noqa: PLC0415 - imports mlx

        from . import mlx_net  # noqa: PLC0415 - imports mlx

        path = ctx.weights()[0]
        self.mlx = mlx_port.recorded(
            ctx,
            mlx_net.LAYOUT,
            lambda sd: mlx_net.MlxMEMFOF(sd, ctx.dtype, ctx.params.iters),
            real=ctx.state_dict,
            skeleton=lambda: mlx_port.header_skeleton(path),
            sources=mlx_net.SOURCES,
        )
        if mb := self.CACHE_MB.get(ctx.variant.id, self.CACHE_MB.get("*", 0)):  # held while this runner lives
            from fastforks._core.mlx import request_cache  # noqa: PLC0415 - MLX only

            request_cache(self, mb)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a padded pair."""
        from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - imports mlx

        return {"flow": to_torch(self.mlx(to_mx(image1.float()), to_mx(image2.float())))}


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": _MlxPortable,
}
# FASTFORKS_MLX_GRAPH=1 (diagnostics, tools/mlx_core_ab.py): the exported-graph port the hand-written network replaced
GRAPH_RUNNERS = {"portable_mlx": mlx_wrap.port(_Mlx, attrs=(), methods=("_flow",), half_wino_f32=True)}
