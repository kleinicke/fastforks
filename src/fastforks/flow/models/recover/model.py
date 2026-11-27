"""ReCoVEr runners: weights -> network and the portable / kernels forward passes.

Both fold the encoders' batch norms into their convolutions and the ConvNeXt layer scales into the Linear before.
Portable on CUDA in float16 runs the whole network in fp16 channels-last (the flow state, the softmax of the
upsampling weights and the convex upsampling in float32); kernels adds fp16-accumulate GEMMs (the ConvNeXt MLPs)
and cuDNN fp16-accumulate convolutions.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.checkpoint import check_fits
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import is_base, override
from fastforks.flow._runner import Forward, Runner
from fastforks.kernels import cudnn_hacc, flow_upsample, recover_motion
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import SmallShapeGraphs

from .net import BasicBlock, CNBlock, ConvNextBlock, ReCoVEr, ResNetFPN

if TYPE_CHECKING:
    from fastforks.flow._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` and `[options]`."""

    arch: Literal["rn", "mn", "cx"]  # context encoder: ResNet34, MobileNetV3-Large, ConvNeXt-T stages
    iters: int  # flow updates after the initial regression (upstream `iters`, 4)
    cuda_graphs: bool  # option: kernels replays CUDA graphs for small repeated shapes


@torch.no_grad()
def _fold(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
    """`bn(conv(x))` -> `conv'(x)` (inference statistics)."""
    mean, var = cast("torch.Tensor", bn.running_mean), cast("torch.Tensor", bn.running_var)  # eval statistics
    s = bn.weight / torch.sqrt(var + bn.eps)
    conv.weight.data = conv.weight.data * s.view(-1, 1, 1, 1)
    b = conv.bias.data if conv.bias is not None else torch.zeros_like(mean)
    conv.bias = nn.Parameter((b - mean) * s + bn.bias)


def fold_batchnorms(net: nn.Module) -> int:
    """Fold every conv -> batch norm pair of the encoders; returns how many."""
    n = 0
    for m in list(net.modules()):
        if isinstance(m, nn.Sequential):
            for i in range(len(m) - 1):
                conv, bn = m[i], m[i + 1]
                if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                    _fold(conv, bn)
                    m[i + 1] = nn.Identity()
                    n += 1
        if isinstance(m, (BasicBlock, ResNetFPN)):
            pairs = [("conv1", "bn1")] + ([("conv2", "bn2")] if isinstance(m, BasicBlock) else [])
            for c, b in pairs:
                _fold(getattr(m, c), getattr(m, b))
                setattr(m, b, nn.Identity())
                n += 1
            if isinstance(m, BasicBlock) and m.downsample is not None:
                m.bn3 = nn.Identity()  # ty: ignore[invalid-assignment]  # folded with downsample.0 by the Sequential pass
    return n


def _cn_block(self: CNBlock, x: torch.Tensor) -> torch.Tensor:
    return x + self.block(x)  # layer scale folded into block[5] (`fold_layer_scales`)


def _convnext_block(self: ConvNextBlock, x: torch.Tensor) -> torch.Tensor:
    y = self.dwconv(x).permute(0, 2, 3, 1)
    y = self.pwconv2(self.act(self.pwconv1(self.norm(y))))  # gamma folded into pwconv2 (`fold_layer_scales`)
    return self.final(x + y.permute(0, 3, 1, 2))


@torch.no_grad()
def fold_layer_scales(net: nn.Module) -> int:
    """`gamma * linear(x)` -> `linear'(x)` in the ConvNeXt blocks (one multiply kernel less each); returns how many."""
    n = 0
    for m in net.modules():
        if isinstance(m, (CNBlock, ConvNextBlock)):
            scale, linear = (m.layer_scale, m.block[5]) if isinstance(m, CNBlock) else (m.gamma, m.pwconv2)
            linear = cast("nn.Linear", linear)
            g = scale.reshape(-1, 1)
            linear.weight.copy_(linear.weight * g)
            cast("torch.Tensor", linear.bias).copy_(cast("torch.Tensor", linear.bias) * g[:, 0])
            override(m, _cn_block if isinstance(m, CNBlock) else _convnext_block)
            n += 1
    return n


def _hacc_forward(self: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    from fastforks.kernels.cudnn_hacc import conv_half_acc  # noqa: PLC0415 - kernels only

    return conv_half_acc(x.half(), self.weight, self.bias, list(self.padding), list(self.stride), fuse_bias=True)  # ty: ignore


def _hacc_convs(module: nn.Module) -> int:
    """Route every fp16 Conv2d (groups 1, dilation 1, zero padding) through cuDNN fp16-accumulate convs."""
    n = 0
    for m in module.modules():
        if (
            is_base(m, nn.Conv2d)
            and m.groups == 1
            and m.dilation == (1, 1)
            and m.padding_mode == "zeros"
            and not isinstance(m.padding, str)
            and m.weight.dtype == torch.float16
        ):
            m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
            override(m, _hacc_forward)
            n += 1
    return n


class Portable(Runner):
    """Plain PyTorch: float32, or on CUDA the fp16 network (see the module docstring)."""

    dtypes = ("float32", "float16")
    Params = Params
    batch = None
    # pixels (pairs x H x W) per network call: the eager call is launch bound at small sizes (RTX 5060 Ti, Windows,
    # fp16, 8 pairs 480 x 640: mn 12.9 -> 6.6 ms per pair, kernels 14.1 -> 5.5; rn and cx gain 5-35 %), not at 720p
    batch_pixels = 1_300_000
    gacc = False  # fp16-accumulate GEMMs

    def __init__(self, ctx: Context) -> None:
        """Build the network, fold its batch norms, convert it."""
        super().__init__(ctx)
        self.net = ReCoVEr(ctx.params.arch)
        sd = {k: v for k, v in ctx.state_dict().items() if not k.startswith("fnet.")}  # unused by upstream
        check_fits(self.net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
        self.net.load_state_dict(sd)
        self.net.eval()
        fold_batchnorms(self.net)
        fold_layer_scales(self.net)
        dt = torch.float16 if ctx.dtype == "float16" else torch.float32
        self.layout = torch.channels_last if ctx.device.type == "cuda" else torch.contiguous_format
        self.net.to(ctx.device, dt).to(memory_format=self.layout)

    def accumulate(self) -> contextlib.AbstractContextManager[None]:
        """fp16-accumulate GEMMs for kernels."""
        return fp16_accumulate() if self.gacc else contextlib.nullcontext()

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a padded batch."""
        if not self.gacc:
            image1, image2 = (x.contiguous(memory_format=self.layout) for x in (image1, image2))
        with torch.autocast(image1.device.type, enabled=False), self.accumulate():
            return {"flow": self.net(image1, image2, self.ctx.params.iters)}

    def module(self) -> torch.nn.Module:
        """The network; returns the flow."""
        iters = self.ctx.params.iters
        return Forward(self.net, lambda net, image1, image2: net(image1, image2, iters))


def _kernel_pair(_net: nn.Module, image1: torch.Tensor, image2: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    b, c, h, w = image1.shape
    out = torch.empty((b, 2 * c, h, w), dtype=dtype, device=image1.device, memory_format=torch.channels_last)
    return torch.cat((image1, image2), 1, out=out)


def _tile(self: nn.Module, _h: int, _w: int) -> torch.Tensor:
    return cast("torch.Tensor", self.correlation_tile)


def _motion(self: Any, flow: torch.Tensor, tile: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - prepared motion encoder
    flo = F.relu(self.convf2(F.relu(self.convf1(flow))))
    motion = F.conv2d(flo, self.flow_weight, padding=1)
    return recover_motion.pack(motion, tile, flow)


_GRAPH_PIXELS = 1_100_000  # fixed: pairs x padded H x W; the pass's few hundred launches bound smaller calls
_GRAPHS = 2  # fixed: graph cache policy


class Kernels(Portable):
    """Portable's fp16 network with fp16-accumulate GEMMs and cuDNN fp16-accumulate convolutions.

    A repeated call shape up to `_GRAPH_PIXELS` replays a CUDA graph (`fastforks.shared.cuda_graphs`): the eager pass
    is launch-bound there.
    """

    gacc = True

    def __init__(self, ctx: Context) -> None:
        """Build the network and route its dense convolutions through `cudnn_hacc`."""
        super().__init__(ctx)
        if _hacc_convs(self.net) == 0:
            msg = "ReCoVEr kernels: no conv routed through cudnn_hacc"
            raise RuntimeError(msg)

        encoder = self.net.update_block.encoder
        with torch.inference_mode(), self.accumulate():
            tile = encoder.corr_term(5, 5).contiguous(memory_format=torch.channels_last)
            encoder.register_buffer("correlation_tile", tile, persistent=False)
            channels = encoder.convf2.out_channels
            weight = encoder.conv.weight[:, -channels:].contiguous(memory_format=torch.channels_last)
            encoder.register_buffer("flow_weight", weight, persistent=False)
        override(encoder, _tile, "corr_term")
        override(encoder, _motion)
        override(self.net, _kernel_pair, "pair")
        self.net.upsample = flow_upsample.convex
        self._graphs = SmallShapeGraphs(
            self._eager,
            pixels=_GRAPH_PIXELS,
            graphs=_GRAPHS,
            enabled=ctx.params.cuda_graphs,
            capture_after=cudnn_hacc.capture_after(),
        )

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a padded batch."""
        return {"flow": self._graphs(image1, image2)}

    def _eager(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        with torch.autocast(image1.device.type, enabled=False), self.accumulate():
            return self.net(image1, image2, self.ctx.params.iters)

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        self._graphs.release()

    def module(self) -> torch.nn.Module:
        """No exportable module: kernel launches."""
        return Runner.module(self)


class _OnePair(Portable):
    """Portable for the MLX graph port: one pair per call, as before batching (not measured on Apple GPUs)."""

    batch = 1


class _MlxPortable(Runner):
    """Apple MLX: the network written in MLX operations (`mlx_net`) from the checkpoint's tensors, one pair per call."""

    batch = 1  # one pair per call (`mlx_net`)
    dtypes = ("float32", "float16")
    Params = Params
    module = Runner.module  # the MLX network is not exportable
    # the buffer cache this mode asks for, by `variant:dtype`, else by variant (none: the base limit). The e2e footprint
    # carries the cache: these are the largest that keep it below the graph port's rows at 1080 x 1920 (M1 Pro);
    # 1024 MB was up to 7 % faster but 26-59 % heavier
    CACHE_MB: ClassVar[dict[str, int]] = {"recover-cx": 256, "recover-rn:float16": 256}

    def __init__(self, ctx: Context) -> None:
        """The MLX network from the conversion cache (the first load converts the checkpoint's weights)."""
        super().__init__(ctx)
        from fastforks.shared import mlx_port  # noqa: PLC0415 - imports mlx

        from . import mlx_net  # noqa: PLC0415 - imports mlx

        arch, iters = ctx.params.arch, ctx.params.iters
        self.mlx = mlx_port.recorded(
            ctx,
            mlx_net.LAYOUT,
            lambda sd: mlx_net.MlxReCoVEr(sd, arch, ctx.dtype, iters),
            real=ctx.state_dict,
            skeleton=lambda: ReCoVEr(arch).state_dict(),
            sources=mlx_net.SOURCES,
        )
        v = ctx.variant.id
        if mb := self.CACHE_MB.get(f"{v}:{ctx.dtype}", self.CACHE_MB.get(v, 0)):  # held while this runner lives
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
GRAPH_RUNNERS = {"portable_mlx": mlx_wrap.port(_OnePair)}
