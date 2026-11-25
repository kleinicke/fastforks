"""MemFlow runners: weights -> network and the forward pass of a pair."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import check_fits
from fastforks._core.patch import override
from fastforks.flow._runner import Forward, Runner
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import SmallShapeGraphs

from .net import LEVELS, MemFlowNet, PCBlock, convert

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks._core.kernels import CubinModule
    from fastforks.flow._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` and `[options]`."""

    iters: int  # flow updates (upstream `val_decoder_depth`)
    train_avg_length: float  # upstream config value of the attention scale (tokens of its training crop at 1/8, x1.5)
    cuda_graphs: bool


def _net(ctx: Context) -> MemFlowNet:
    net = MemFlowNet(ctx.params.train_avg_length)
    sd = convert(ctx.state_dict(key="model"))
    check_fits(net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
    net.load_state_dict(sd)
    return net.eval().to(ctx.device)


class Portable(Runner):
    """Plain PyTorch; on CUDA in fp16 the encoders and the update block run under autocast, the correlation in fp32."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        self.fp16 = ctx.dtype == "float16"

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

    Upstream's pyramid average-pools the all-pairs volume over fmap2's pixels, which commutes with the dot product,
    so only the pooled fmap2 pyramid is kept (fp16). Operands are the fp16 features with fp32 sums and bilinear
    weights; the windows are written in fp16, which the update block casts them to.
    """
    b, c, h, w = f1.shape
    a = f1.to(torch.float16).permute(0, 2, 3, 1).contiguous()
    pyr, x = [], f2.to(torch.float16)
    for _ in range(LEVELS):
        pyr.append(x.permute(0, 2, 3, 1).contiguous())
        x = F.avg_pool2d(x, 2)
    side = LEVELS * 81

    def lookup(coords: torch.Tensor) -> torch.Tensor:
        xy = coords.permute(0, 2, 3, 1).contiguous()
        out = torch.empty(b, h, w, side, dtype=torch.float16, device=coords.device)
        for i in range(b):  # one warp per pixel, 8 warps per block
            k.launch(
                "corr_lookup_c256_r4",
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
        return out.permute(0, 3, 1, 2).contiguous()  # NCHW: a channels-last view slows the update block's convs

    return lookup


_FFT_MIN_KERNEL = 15  # fixed: the motion encoder's and flow head's depthwise kernels (the 7 x 7 ones gain nothing)
_FFT_CHANNELS = 64  # fixed: channels per cuFFT batch (1080p, 256 x 15 x 15: 0.96 ms vs 1.16 at 128, 1.87 at 32)
# fixed: float16 readout softmax matrix rows kept across the iterations (the rest are fused attention in each one)
ATTENTION_MATRIX_BYTES = 1 << 30


def _fft_size(n: int) -> int:
    """The smallest size >= `n` with no prime factor above 5 (cuFFT's fast sizes: 149 -> 150 halves the time)."""
    while True:
        m = n
        for f in (2, 3, 5):
            while m % f == 0:
                m //= f
        if m == 1:
            return n
        n += 1


def _fft_fits(m: torch.nn.Conv2d) -> bool:
    """A square depthwise conv of at least `_FFT_MIN_KERNEL` taps, stride 1, zeros padding `k // 2`, with a bias."""
    k = m.kernel_size[0]
    return (
        m.groups == m.in_channels == m.out_channels
        and m.kernel_size == (k, k)
        and k >= _FFT_MIN_KERNEL
        and m.padding == (k // 2, k // 2)
        and m.stride == m.dilation == (1, 1)
        and m.padding_mode == "zeros"
        and m.bias is not None
    )


def _fft_depthwise(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """The `k x k` depthwise conv (zeros padding `k // 2`) as float32 FFT products, `_FFT_CHANNELS` maps at a time.

    PyTorch runs a fp16 NCHW depthwise conv this large with its own direct kernel (cuDNN's channels-last one is 5-7x
    slower still): 2.8 ms for 256 x 15 x 15 at 135 x 240 (1080p at 1/8), 0.96 ms as FFTs. The spectra of the kernels
    are kept for the last size. The output is rounded once to the conv dtype, as the fp32-accumulating conv's.
    """
    b, c, h, w = x.shape
    k, p = conv.kernel_size[0], conv.kernel_size[0] // 2
    size = (_fft_size(h + k - 1), _fft_size(w + k - 1))  # the full linear correlation: no wrap-around
    real = torch.promote_types(x.dtype, torch.float32)  # float64 (tests) stays float64
    cached = conv.__dict__.get("_fft")
    if cached is None or cached[0] != (size, real):
        taps = torch.zeros((c, *size), dtype=real, device=x.device)
        # a correlation is the convolution with the flipped kernel
        taps[:, :k, :k] = conv.weight.detach()[:, 0].to(real).flip(-1, -2)
        cached = ((size, real), torch.fft.rfft2(taps))
        conv.__dict__["_fft"] = cached
    spectra = cached[1]
    dev = x.device.type
    dtype = torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else x.dtype
    out = torch.empty((b, c, h, w), dtype=dtype, device=x.device)
    bias = cast("torch.Tensor", conv.bias).detach().to(real)[:, None, None]
    for i in range(b):
        for s in range(0, c, _FFT_CHANNELS):
            e = s + _FFT_CHANNELS
            y = torch.fft.irfft2(torch.fft.rfft2(x[i, s:e].to(real), s=size).mul_(spectra[s:e]), s=size)
            torch.add(y[:, p : p + h, p : p + w], bias[s:e], out=out[i, s:e])
    return out


def _pointwise_fits(m: torch.nn.Conv2d) -> bool:
    """A plain 1 x 1 conv (stride 1, no padding, one group) with a bias."""
    return (
        m.kernel_size == (1, 1)
        and m.groups == 1
        and m.stride == (1, 1)
        and m.padding == (0, 0)
        and m.dilation == (1, 1)
        and m.bias is not None
    )


def _pointwise(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """The 1 x 1 conv as one GEMM per image on the NCHW layout: `bias + W @ x[C, H * W]` in the weights' dtype.

    cuDNN runs a fp16 NCHW conv in NHWC: it transposes the input and the output of each of the ~35 pointwise convs of
    every update (1080p: 92 of 744 ms). cuBLAS takes the `[C, H * W]` view as it is, with the same fp32 accumulation
    and one rounding of `W @ x + b` (the bias is the GEMM's accumulator input).
    """
    b, c, h, w = x.shape
    weight = conv.weight.detach().view(conv.out_channels, c)
    out = torch.empty((b, conv.out_channels, h, w), dtype=weight.dtype, device=x.device)
    with torch.autocast(x.device.type, enabled=False):  # the weights are already in the autocast dtype
        for i in range(b):
            x2d = x[i].reshape(c, h * w).to(weight.dtype)
            torch.addmm(cast("torch.Tensor", conv.bias).detach()[:, None], weight, x2d, out=out[i].view(-1, h * w))
    return out


def _gelu_residual(conv: torch.nn.Conv2d, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """`gelu(x + conv(h))` for a 1 x 1 conv: `x + bias` is the GEMMs' accumulator input, the GELU runs in place."""
    b, c, hh, ww = h.shape
    weight = conv.weight.detach().view(conv.out_channels, c)
    out = torch.add(x, cast("torch.Tensor", conv.bias).detach()[:, None, None]).to(weight.dtype)
    with torch.autocast(x.device.type, enabled=False):  # the weights are already in the autocast dtype
        for i in range(b):
            out[i].view(-1, hh * ww).addmm_(weight, h[i].reshape(c, hh * ww).to(weight.dtype))
    return torch.ops.aten.gelu_(out)


def _pcblock(m: PCBlock, x: torch.Tensor) -> torch.Tensor:
    """`PCBlock.forward` with its residual GELUs fused into fewer passes over the maps (kernels, fp16 weights).

    The residual 1 x 1 convs (`ffn1`'s second, `pw`) add `x + bias` as their GEMMs' accumulator input; the 1 x 1
    depthwise conv and its residual are one `bias + x * (1 + w)`; every GELU after a sum runs in place. Upstream's
    sums round the conv output to fp16 before adding `x`; here `x + bias` is rounded instead (the same error size).
    """
    conv1, conv2, conv3, conv4 = (cast("torch.nn.Conv2d", c) for c in (m.ffn1[0], m.ffn1[2], m.ffn2[0], m.ffn2[2]))
    x = x.to(m.pw.weight.dtype)
    x = _gelu_residual(conv2, x, torch.ops.aten.gelu_(conv1(x)))
    for module in m.conv_list:
        conv = cast("torch.nn.Conv2d", module)
        if conv.kernel_size == (1, 1):  # depthwise: a per-channel scale
            scale = conv.__dict__.get("_scale")
            if scale is None:
                scale = conv.__dict__["_scale"] = (1 + conv.weight.detach().view(-1, 1, 1).float()).to(x.dtype)
            x = torch.addcmul(cast("torch.Tensor", conv.bias).detach()[:, None, None], x, scale)
        else:
            x = conv(x).add_(x)
        x = torch.ops.aten.gelu_(x)
    x = _gelu_residual(m.pw, x, x)
    return conv4(torch.ops.aten.gelu_(conv3(x)))


class Kernels(Portable):
    """The portable fp16 pass with the correlation windows computed on the fly (`raft_flow`): no all-pairs volume.

    The conv weights are stored in fp16 (the values autocast casts them to on each call); the 15 x 15 depthwise convs
    run as FFTs (`_fft_depthwise`; 1080p, RTX 5060 Ti: 856 -> 740 ms) and the 1 x 1 convs as GEMMs on the NCHW maps
    (`_pointwise`; 744 -> 644 ms).
    """

    _GRAPH_PIXELS = 384 * 640  # fixed: graph work limit (batch x padded image pixels)

    def __init__(self, ctx: Context) -> None:
        """Build the network and bind the lookup kernel."""
        super().__init__(ctx)
        from fastforks.kernels import raft_flow  # noqa: PLC0415 - needs the kernel library

        self.lookup = functools.partial(_kernel_lookup, raft_flow.load(ctx.device))
        # fp16 conv weights: autocast would cast each one on every call (15 updates x ~90 convs); the norms and
        # the aggregator's gain (a product outside a conv) keep fp32 parameters, as under autocast.
        for m in self.net.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.half()
                if _fft_fits(m):
                    override(m, _fft_depthwise)
                elif _pointwise_fits(m):
                    override(m, _pointwise)
            elif isinstance(m, PCBlock):
                override(m, _pcblock)
        # The depthwise convs replace their cached FFT spectra when their geometry changes.
        self._graphs = SmallShapeGraphs(
            self._eager,
            pixels=self._GRAPH_PIXELS,
            enabled=ctx.params.cuda_graphs,
            capture_after=2,
            one_shape=True,
            capture_warmup=False,
        )

    def _eager(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        iters = self.ctx.params.iters
        return self.net.forward_pair(
            image1, image2, iters, corr_fn=self.lookup, fp16=True, matrix_bytes=ATTENTION_MATRIX_BYTES
        )

    def _flow(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        return self._graphs(image1, image2)

    def release_graphs(self) -> None:
        """Free the CUDA graph and its memory pool."""
        self._graphs.release()

    def module(self) -> torch.nn.Module:
        """No exportable module: the lookup is a kernel launch."""
        return Runner.module(self)


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(Portable, attrs=(), methods=("_flow",), cache_mb=1024),
}
