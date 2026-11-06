"""DINOv3 ConvNeXt T/S/B/L in plain PyTorch, with the parameter names of the Hugging Face checkpoints.

Same maths as `transformers` `DINOv3ConvNextModel` in eval: 4 stages (stem conv 4x4 + LayerNorm, then LayerNorm +
conv 2x2 downsampling), blocks = depthwise conv 7x7, LayerNorm, Linear 4x, GELU, Linear, LayerScale, residual.
Outputs as HF: `last_hidden_state` = final LayerNorm of `[mean-pooled token, stage-4 grid tokens]` (stride 32), so the
"cls" token is the pooled token and there are no registers; `pooler_output` = its first token. No `hidden_states`:
HF's per-stage maps have different sizes (not one token grid).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.checkpoint import load_strict
from fastforks.features._runner import Forward, Runner
from fastforks.shared.cuda_graphs import SmallShapeGraphs

if TYPE_CHECKING:
    from fastforks.features._runner import Context

EPS = 1e-6
SIZES = {  # depths, hidden_sizes (config.json)
    "tiny": ((3, 3, 9, 3), (96, 192, 384, 768)),
    "small": ((3, 3, 27, 3), (96, 192, 384, 768)),
    "base": ((3, 3, 27, 3), (128, 256, 512, 1024)),
    "large": ((3, 3, 27, 3), (192, 384, 768, 1536)),
}


@dataclass(frozen=True)
class Params:
    """Variant `params`."""

    arch: str  # "tiny", "small", "base" or "large"


class _LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channels of `[B, C, H, W]` (HF `data_format="channels_first"`)."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002 - nn.LayerNorm's parameter name
        return super().forward(input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _Layer(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.depthwise_conv = nn.Conv2d(c, c, 7, padding=3, groups=c)
        self.layer_norm = nn.LayerNorm(c, eps=EPS)
        self.pointwise_conv1, self.pointwise_conv2 = nn.Linear(c, 4 * c), nn.Linear(4 * c, c)
        self.gamma = nn.Parameter(torch.empty(c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layer_norm(self.depthwise_conv(x).permute(0, 2, 3, 1))
        y = self.pointwise_conv2(F.gelu(self.pointwise_conv1(y))) * self.gamma
        return x + y.permute(0, 3, 1, 2)


class _Stage(nn.Module):
    def __init__(self, cin: int, cout: int, depth: int, *, stem: bool) -> None:
        super().__init__()
        self.downsample_layers = nn.ModuleList(
            [nn.Conv2d(cin, cout, 4, 4), _LayerNorm2d(cout, eps=EPS)]
            if stem
            else [_LayerNorm2d(cin, eps=EPS), nn.Conv2d(cin, cout, 2, 2)]
        )
        self.layers = nn.Sequential(*(_Layer(cout) for _ in range(depth)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.downsample_layers:
            x = m(x)
        return self.layers(x)


class ConvNext(nn.Module):
    """`forward(x)`: normalised `[B, 3, H, W]` (multiples of 32) -> `last_hidden_state` `[B, 1 + h * w, C]`."""

    def __init__(self, arch: str) -> None:
        """Uninitialised network (assign a checkpoint with `load_strict(..., assign=True)`)."""
        super().__init__()
        depths, widths = SIZES[arch]
        cins = (3, *widths[:-1])
        self.stages = nn.ModuleList(
            _Stage(ci, co, d, stem=i == 0) for i, (ci, co, d) in enumerate(zip(cins, widths, depths, strict=True))
        )
        self.layer_norm = nn.LayerNorm(widths[-1], eps=EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """HF `last_hidden_state`."""
        for s in self.stages:
            x = s(x)
        t = x.flatten(2).transpose(1, 2)
        return self.layer_norm(torch.cat([t.mean(1, keepdim=True), t], 1))


def _forward(net: ConvNext, x: torch.Tensor) -> torch.Tensor:
    """`ConvNext.forward` with the blocks as functional ops, same maths.

    Small calls are CPU launch bound, and the module calls of a block cost more than its GPU work; the LayerScale +
    residual is one `addcmul` (one fp16 rounding instead of two).
    """
    for st in cast("list[_Stage]", net.stages):
        for m in st.downsample_layers:
            x = m(x)
        for blk in cast("list[_Layer]", st.layers):
            dw, ln, fc1, fc2 = blk.depthwise_conv, blk.layer_norm, blk.pointwise_conv1, blk.pointwise_conv2
            y = F.conv2d(x, dw.weight, dw.bias, padding=3, groups=x.shape[1])
            y = F.layer_norm(y.permute(0, 2, 3, 1), ln.normalized_shape, ln.weight, ln.bias, ln.eps)
            y = F.linear(F.gelu(F.linear(y, fc1.weight, fc1.bias)), fc2.weight, fc2.bias)
            x = torch.addcmul(x, y.permute(0, 3, 1, 2), blk.gamma[:, None, None])
    t = x.flatten(2).transpose(1, 2)
    return net.layer_norm(torch.cat([t.mean(1, keepdim=True), t], 1))


class Portable(Runner):
    """Plain PyTorch (functional blocks): float32, or fp16 weights / activations."""

    dtypes = ("float32", "float16")
    batch = None
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        net = ConvNext(ctx.params.arch)
        load_strict(net, ctx.state_dict(), f"{ctx.variant.ref} checkpoint", assign=True)
        self.half = ctx.dtype == "float16"
        self.net = (net.half() if self.half else net).eval().to(ctx.device)
        self.hidden_size, self.num_layers = SIZES[ctx.params.arch][1][-1], 4
        self.num_prefix_tokens, self.num_register_tokens = 1, 0

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        return _forward(self.net, x.half() if self.half else x).float()

    def __call__(self, image: torch.Tensor, layers: tuple[int, ...]) -> dict[str, torch.Tensor]:
        """Features of a resized, normalised batch (`layers` is always empty: the card has no hidden_states)."""
        del layers
        last = self._run(image)
        return {"last_hidden_state": last, "pooler_output": last[:, 0]}

    def module(self) -> torch.nn.Module:
        """The network: normalised `[B, 3, h, w]` (sides multiples of 32) -> `last_hidden_state` `[B, N, C]`."""
        return Forward(self.net, lambda _net, x: self._run(x))


# CUDA graphs while the call is CPU launch bound: up to this many input pixels (batch x H x W), measured
_GRAPH_PIXELS = 1 << 20  # fixed: launch-bound threshold, measured
_GRAPHS = 4  # fixed: graph cache policy, not architecture
_GELU = 5  # fixed: fvm activation code
# second MLP GEMMs with at least this reduction length (ConvNeXt-L stage 4: 6144) run cuBLAS (fp32 accumulate) + add:
# golden p99 0.032 -> 0.027 (tolerance 0.04) at no measured cost; from 3072 on: 0.024 but 1088x1920 61 -> 71 ms
_FP32_ACC_K = 6144  # fixed: measured accuracy / speed trade-off
_LN_ROWS = 8  # fixed: rows (warps) per block of `ln_gather_h`, measured (4 / 8 / 16 within 5 %)
_LN_MAX_BLOCKS = 65535 * 8  # fixed: grid cap, the kernel loops over the rest


def _h(t: torch.Tensor | None, dev: torch.device) -> torch.Tensor:
    assert t is not None  # noqa: S101 - every conv / norm here has a bias
    return t.detach().half().to(dev)


def _w4(w: torch.Tensor, dev: torch.device) -> torch.Tensor:
    """Linear `[N, K]` or conv weight as a channels-last fp16 conv weight `[N, K, R, S]`."""
    w = w.detach().half()
    return (w[:, :, None, None] if w.dim() == 2 else w).contiguous(memory_format=torch.channels_last).to(dev)  # noqa: PLR2004


def _kernels_layer(blk: _Layer, dev: torch.device) -> tuple[torch.Tensor, ...]:
    """Depthwise `[7, 7, C]` + bias, LayerNorm, fc1, fc2 with the LayerScale folded in (fp32 product, then fp16)."""
    g = blk.gamma.detach().float()
    fc2 = blk.pointwise_conv2
    return (
        _h(blk.depthwise_conv.weight[:, 0].permute(1, 2, 0).contiguous(), dev),
        _h(blk.depthwise_conv.bias, dev),
        _h(blk.layer_norm.weight, dev),
        _h(blk.layer_norm.bias, dev),
        _w4(blk.pointwise_conv1.weight, dev),
        _h(blk.pointwise_conv1.bias, dev),
        _w4(fc2.weight.detach().float() * g[:, None], dev),
        _h(fc2.bias.detach().float() * g, dev),
    )


class Kernels(Runner):
    """fvm_conv kernels on fp16 channels-last activations.

    Depthwise 7x7 in fp32 accumulation; the stem / downsampling convs and the MLP GEMMs as fused CUTLASS convs (fp16
    accumulate) with bias + GELU, and LayerScale (folded) + residual in the epilogue (from `_FP32_ACC_K`: cuBLAS + add).
    LayerNorms as fvm_seg `ln_gather_h` (one warp per row, identity row index; 4.3x faster than torch's at C = 96) on
    the channels-last view. Small calls (`_GRAPH_PIXELS`) replay a CUDA graph per input shape, captured on its second
    call. Outputs float32.
    """

    batch = None
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Load the checkpoint, fold LayerScale into the second MLP GEMM, keep fp16 channels-last weights."""
        super().__init__(ctx)
        from fastforks._optimize import conv_ops  # noqa: PLC0415 - fvm_conv only (not `ops`: all fvm_* libraries)
        from fastforks.kernels import fvm_seg  # noqa: PLC0415

        self.ops, self.lin = conv_ops, conv_ops.lin
        self.seg = fvm_seg.load(ctx.device)
        self._idx = torch.arange(0, dtype=torch.int32, device=ctx.device)  # identity row index of `ln_gather_h`
        self._old: list[torch.Tensor] = []  # outgrown index buffers a captured graph may still read
        net = ConvNext(ctx.params.arch)
        load_strict(net, ctx.state_dict(), f"{ctx.variant.ref} checkpoint", assign=True)
        dev = ctx.device
        self.stages = []
        with torch.no_grad():
            for i, st in enumerate(net.stages):
                dl = list(cast("nn.ModuleList", st.downsample_layers))
                conv, norm = dl if i == 0 else dl[::-1]
                assert isinstance(conv, nn.Conv2d), conv  # noqa: S101 - fixed architecture
                assert isinstance(norm, nn.LayerNorm), norm  # noqa: S101
                w = conv.weight if i else F.pad(conv.weight, (0, 0, 0, 0, 0, 8 - conv.in_channels))  # `input_nhwc8`
                down = (_w4(w, dev), _h(conv.bias, dev), list(conv.stride))
                layers = [_kernels_layer(cast("_Layer", blk), dev) for blk in cast("nn.Sequential", st.layers)]
                self.stages.append((down, (_h(norm.weight, dev), _h(norm.bias, dev)), layers))
            self.final = (_h(net.layer_norm.weight, dev), _h(net.layer_norm.bias, dev))
        self.hidden_size, self.num_layers = SIZES[ctx.params.arch][1][-1], 4
        self.num_prefix_tokens, self.num_register_tokens = 1, 0
        self.graphs = SmallShapeGraphs(self._forward, pixels=_GRAPH_PIXELS, graphs=_GRAPHS)

    def _ln(self, x: torch.Tensor, wb: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """LayerNorm over the last dim of a contiguous fp16 tensor (fp32 statistics)."""
        c = x.shape[-1]
        m = x.numel() // c
        if self._idx.numel() < m:  # grows on an eager first call of a larger shape, never while capturing
            self._old = [*self._old, self._idx]  # rebind, not append: atomic vs `release_graphs` from another thread
            self._idx = torch.arange(max(m, 2 * self._idx.numel()), dtype=torch.int32, device=x.device)
        y = torch.empty_like(x)
        grid = (min(-(-m // _LN_ROWS), _LN_MAX_BLOCKS), 1, 1)
        self.seg.launch("ln_gather_h", grid, (32 * _LN_ROWS, 1, 1), x, self._idx, wb[0], wb[1], y, m, c, EPS)
        return y

    def _forward(self, image: torch.Tensor) -> torch.Tensor:
        """fp16 channels-last forward: `[B, 3, H, W]` float32 -> `last_hidden_state` float32."""
        ops, lin = self.ops, self.lin
        x = ops.input_nhwc8(image)
        for i, ((wd, bd, sd), (lw, lb), layers) in enumerate(self.stages):
            if i == 0:
                x = ops.conv_fused(x, wd, bd, None, [0, 0], sd, 0)
                x = self._ln(x.permute(0, 2, 3, 1), (lw, lb)).permute(0, 3, 1, 2)
            else:
                x = self._ln(x.permute(0, 2, 3, 1), (lw, lb)).permute(0, 3, 1, 2)
                x = ops.conv_fused(x, wd, bd, None, [0, 0], sd, 0)
            c = x.shape[1]
            for dw, dwb, nw, nb, w1, b1, w2, b2 in layers:
                h = ops.dwconv(x, dw, dwb, [1, 1], [3, 3], 0).permute(0, 2, 3, 1)
                h = lin(self._ln(h, (nw, nb)), w1, b1, act=_GELU)
                r = x.permute(0, 2, 3, 1)
                if w2.shape[1] >= _FP32_ACC_K:
                    x = (F.linear(h, w2.reshape(w2.shape[0], -1), b2) + r).permute(0, 3, 1, 2)
                else:
                    x = lin(h, w2, b2, res=r).permute(0, 3, 1, 2)
        b, c = x.shape[:2]
        t = x.permute(0, 2, 3, 1).reshape(b, -1, c)
        t = torch.cat([t.float().mean(1, keepdim=True).half(), t], 1)
        return self._ln(t, self.final).float()

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        self.graphs.release()
        self._old = []

    def __call__(self, image: torch.Tensor, layers: tuple[int, ...]) -> dict[str, torch.Tensor]:
        """Features of a resized, normalised batch (`layers` is always empty: the card has no hidden_states)."""
        del layers
        with torch.autocast("cuda", enabled=False):
            last = self.graphs(image)
        return {"last_hidden_state": last, "pooler_output": last[:, 0]}


RUNNERS = {"portable": Portable, "kernels": Kernels}
