"""NAFNet runners: the network of megvii-research/NAFNet and its test call (`net(lq)`, clamped to `[0, 1]`).

The network pads the input with zeros to a multiple of 16 itself and crops its output (`check_image_size`). The
deblurring models are `NAFNetLocal` (test-time local converter, TLC): every channel attention averages over a window of
1.5 x the training size at its level (upstream: `AvgPool2d` with the window found by one forward pass at the training
size on construction, sums from a cumulative sum, replicate-padded); here the windows are computed from the level, the
box sums are float32 as upstream's. Each block's channel LayerNorm runs as `F.layer_norm` over the channel dimension
of a channels-last view (upstream: per-pixel mean / variance as an autograd function); its affine and the block's
residual scales (`beta`, `gamma`) are folded into the 1 x 1 convs next to them at load. The residual stream runs
scaled by 1 / 1024 and the channel attention by 1 / 64 (exact: weights and norm epsilons scaled at load), as float16
overflows them on small inputs (upstream's `.half()` too). On CUDA the network runs channels-last.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks._core.errors import WeightsError
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import is_base, override
from fastforks.kernels import cudnn_hacc, dpt_fused, nafnet_gate
from fastforks.restoration._runner import Forward, Runner
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import SmallShapeGraphs

from .net import NAFNet
from .net.NAFNet_arch import NAFBlock

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks.restoration._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params` (the test options' `network_g`)."""

    scale: int  # 1
    width: int  # channels at full resolution
    enc_blocks: tuple[int, ...]  # NAFBlocks per encoder level
    middle_blocks: int
    dec_blocks: tuple[int, ...]  # NAFBlocks per decoder level (deepest first)
    key: str  # checkpoint key ("params")
    tlc: int  # training patch size of `NAFNetLocal` (local channel attention, window 1.5 x); 0: `NAFNet` (global)
    cuda_graphs: bool


class _LocalMean(torch.nn.Module):
    """`local_arch.AvgPool2d` (TLC, `fast_imp=False`) with its window given: box means, replicate-padded to the size."""

    def __init__(self, window: int) -> None:
        super().__init__()
        self.window = window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if self.window >= h and self.window >= w:
            return x.mean((2, 3), keepdim=True)
        k1, k2 = min(h, self.window), min(w, self.window)
        out = _box_mean(x, k1, k2).to(x.dtype)
        return F.pad(out, _replicate_pad(k1, k2), mode="replicate")


def _replicate_pad(kh: int, kw: int) -> tuple[int, int, int, int]:
    return ((kw - 1) // 2, kw // 2, (kh - 1) // 2, kh // 2)


_BOX_BUFFER_BYTES = 16 * 1024**2
_FFN_TILE_BYTES = 128 << 20  # fixed: bound the CUDA gate projection of large feature maps
_FFN_TILE_ROWS = 256  # fixed: cap the number of rows in a CUDA gate tile


def _box_mean(x: torch.Tensor, k1: int, k2: int) -> torch.Tensor:
    """Limit the temporary integral image by processing independent CUDA channels in groups."""
    if x.is_cuda and not (torch.is_grad_enabled() or torch.compiler.is_compiling() or torch.jit.is_tracing()):
        batch, channels, h, w = x.shape
        integral_bytes = 4 * batch * (h + 1) * (w + 1)
        if integral_bytes * channels >= 6 * _BOX_BUFFER_BYTES:
            group = max(1, _BOX_BUFFER_BYTES // integral_bytes)
            out = x.new_empty((batch, channels, h - k1 + 1, w - k2 + 1), dtype=torch.float32)
            for src, dst in zip(x.split(group, dim=1), out.split(group, dim=1), strict=True):
                dst.copy_(_integral_mean(src, k1, k2))
            return out
    return _integral_mean(x, k1, k2)


def _integral_mean(x: torch.Tensor, k1: int, k2: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    s = x.new_zeros((*x.shape[:2], h + 1, w + 1), dtype=torch.float32)  # one float32 buffer, sums in place
    s[:, :, 1:, 1:] = x
    s = s.cumsum_(-1).cumsum_(-2)
    out = s[:, :, k1:, k2:] + s[:, :, :-k1, :-k2]
    out = out.sub_(s[:, :, :-k1, k2:]).sub_(s[:, :, k1:, :-k2]).div_(k1 * k2)
    del s
    return out


def _norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Channel LayerNorm without affine (folded into the next conv), over a channels-last view.

    On the CPU the result is made contiguous: the channels-last view sends the next (depthwise) convs down
    `_slow_conv2d` (5.5x slower).
    """
    y = F.layer_norm(x.permute(0, 2, 3, 1), x.shape[1:2], eps=eps).permute(0, 3, 1, 2)
    return y.contiguous() if x.device.type == "cpu" else y


def _block(m: NAFBlock, inp: torch.Tensor) -> torch.Tensor:
    """`NAFBlock.forward` (eval: no dropout) with the norms' affine and `beta` / `gamma` folded into the convs."""
    x = m.conv2(m.conv1(_norm(inp, m.norm1.eps)))
    a, b = x.chunk(2, 1)
    x = a * b
    return _finish_block(m, inp, x)


def _finish_block(m: NAFBlock, inp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    y = inp + m.conv3(x * m.sca(x))
    a, b = m.conv4(_norm(y, m.norm2.eps)).chunk(2, 1)
    return y + m.conv5(a * b)


def _fold(m: NAFBlock) -> None:
    """Fold norm1 / norm2's affine into conv1 / conv4 and `beta` / `gamma` into conv3 / conv5."""
    with torch.no_grad():
        for norm, conv in ((m.norm1, m.conv1), (m.norm2, m.conv4)):
            bias: torch.Tensor = conv.bias  # ty: ignore[invalid-assignment]
            weight, shift = norm.weight, norm.bias  # LayerNorm2d parameters
            bias.add_(conv.weight[:, :, 0, 0] @ shift)  # ty: ignore[unsupported-operator]
            conv.weight.mul_(weight[None, :, None, None])  # ty: ignore[not-subscriptable]
        for scale, conv in ((m.beta, m.conv3), (m.gamma, m.conv5)):
            bias: torch.Tensor = conv.bias  # ty: ignore[invalid-assignment]
            conv.weight.mul_(scale.view(-1, 1, 1, 1))
            bias.mul_(scale.view(-1))
    eps1, eps2 = m.norm1.eps, m.norm2.eps
    m.norm1, m.norm2 = _Eps(eps1), _Eps(eps2)  # ty: ignore[invalid-assignment]
    del m.beta, m.gamma


_STREAM, _GATE = 1 / 1024, 1 / 64  # powers of two: exact in the weights


def _scale_stream(net: NAFNet) -> None:
    """Carry the residual stream times `_STREAM` and the channel attention times `_GATE`: float16 overflows both.

    Upstream's GoPro model reaches ~2e5 in the stream of its deepest levels and ~6e5 in `x * sca(x)` on a 101 x 151
    image (~3e3 on GoPro frames). The stream is linear in `intro`, `downs` (their bias), each block's conv3 / conv5
    and the skips; the blocks' norms see it scale-free once their eps is scaled by `_STREAM^2`; `ending` takes the
    factor back. `sca`'s 1 x 1 conv carries `_GATE`, conv3 takes it back. 1 / 16 was not enough: low-contrast noise
    (uniform noise at 10 % contrast, the smoke test's synthetic image) took the deepest blocks' `conv3` to ~5e5 (true
    scale), so `inp + conv3` overflowed to inf and the next norm gave NaN. 1 / 1024 keeps those inputs finite with the
    same accuracy as 1 / 256 .. 1 / 4096 (float16 keeps relative precision; the norms are scale-free).
    """
    with torch.no_grad():
        net.intro.weight.mul_(_STREAM)
        net.intro.bias.mul_(_STREAM)  # ty: ignore[unresolved-attribute]
        net.ending.weight.div_(_STREAM)
        for down in net.downs:
            down.bias.mul_(_STREAM)  # ty: ignore[call-non-callable]
        for m in net.modules():
            if isinstance(m, _Eps):
                m.eps *= _STREAM**2
            elif isinstance(m, NAFBlock):
                for conv in (m.conv3, m.conv5):
                    conv.weight.mul_(_STREAM)
                    conv.bias.mul_(_STREAM)  # ty: ignore[unresolved-attribute]
                gate: torch.nn.Conv2d = m.sca[1]  # ty: ignore[invalid-assignment]
                gate.weight.mul_(_GATE)
                gate.bias.mul_(_GATE)  # ty: ignore[unresolved-attribute]
                m.conv3.weight.div_(_GATE)


class _Eps(torch.nn.Module):
    """A folded LayerNorm2d: only its epsilon is left."""

    def __init__(self, eps: float) -> None:
        super().__init__()
        self.eps = eps


def _prepare(net: NAFNet, tlc: int) -> None:
    """Fold the blocks' norms and scales, set the TLC windows (`tlc`: 0 for global attention), scale the stream."""
    n = len(net.encoders)
    levels = [*((i, s) for i, s in enumerate(net.encoders)), (n, net.middle_blks)]
    levels += [(n - 1 - i, s) for i, s in enumerate(net.decoders)]
    for level, stage in levels:
        for m in stage:  # ty: ignore[not-iterable]
            _fold(m)
            override(m, _block)
            if tlc:  # upstream's window: (tlc >> level) * (1.5 tlc) // tlc
                window = (tlc >> level) * (3 * tlc // 2) // tlc
                m.sca[0] = _LocalMean(window)
    _scale_stream(net)


class Portable(Runner):
    """Plain PyTorch in the card's dtype, channels-last on CUDA."""

    dtypes = ("float32", "float16", "bfloat16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network, fold the norms, set the TLC windows."""
        super().__init__(ctx)
        p = ctx.params
        self.net = NAFNet(
            img_channel=3, width=p.width, middle_blk_num=p.middle_blocks,
            enc_blk_nums=list(p.enc_blocks), dec_blk_nums=list(p.dec_blocks),
        )  # fmt: skip
        load_strict(self.net, ctx.state_dict(key=p.key), f"{ctx.variant.ref} checkpoint")
        _prepare(self.net, p.tlc)
        self.fmt = torch.channels_last if torch.device(ctx.device).type == "cuda" else torch.contiguous_format
        self.net.eval().to(ctx.device, getattr(torch, ctx.dtype)).to(memory_format=self.fmt)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The restored batch."""
        x = image.to(getattr(torch, self.ctx.dtype)).contiguous(memory_format=self.fmt)
        if x.is_cuda and not (torch.is_grad_enabled() or torch.compiler.is_compiling() or torch.jit.is_tracing()):
            return {"image": _restore(self.net, x)}
        return {"image": self.net(x)}

    def module(self) -> torch.nn.Module:
        """The network in the card's dtype."""
        dtype, fmt = getattr(torch, self.ctx.dtype), self.fmt
        return Forward(self.net, lambda net, image: net(image.to(dtype).contiguous(memory_format=fmt)))


def _hacc_conv(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    return cudnn_hacc.conv_half_acc(x, conv.weight, conv.bias, conv.padding, conv.stride, conv.dilation)  # ty: ignore[invalid-argument-type]


def _gated_block(m: NAFBlock, inp: torch.Tensor) -> torch.Tensor:
    x = m.conv1(_norm(inp, m.norm1.eps))
    x = nafnet_gate.depthwise_gate(x, m.gate_weight, m.conv2.bias)  # ty: ignore[invalid-argument-type]
    return _finish_block(m, inp, x)


def _pointwise(
    conv: torch.nn.Conv2d,
    x: torch.Tensor,
    fused: dpt_fused.DptFused,
    residual: torch.Tensor | None = None,
    *,
    overwrite_residual: bool = False,
) -> torch.Tensor:
    """Pointwise NHWC convolution with its bias and optional residual in the GEMM epilogue."""
    x = x.permute(0, 2, 3, 1).contiguous()
    res = None if residual is None else residual.permute(0, 2, 3, 1).contiguous()
    y = fused.gemm(x, conv.weight.flatten(1), conv.bias, res=res, out=res if overwrite_residual else None)
    return y.permute(0, 3, 1, 2)


def _scale_features(m: NAFBlock, x: torch.Tensor) -> torch.Tensor:
    pool = m.sca[0]
    h, w = x.shape[-2:]
    if isinstance(pool, _LocalMean) and pool.window < max(h, w):
        # The same 1x1 projection is applied at every pixel. Project each valid
        # window once before replicating its boundary scores.
        kh, kw = min(h, pool.window), min(w, pool.window)
        means = _box_mean(x, kh, kw).to(dtype=x.dtype, memory_format=torch.channels_last)
        scores = m.sca[1](means).contiguous(memory_format=torch.channels_last)
        return nafnet_gate.scale_replicate_(x, scores)
    return x.mul_(m.sca(x))


def _kernel_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    if x.shape[1] in nafnet_gate.NORM_CHANNELS:
        return nafnet_gate.channel_norm(x, eps)
    return _norm(x, eps)


def _fused_block(m: NAFBlock, inp: torch.Tensor) -> torch.Tensor:
    fused = dpt_fused.load(inp.device)
    x = _pointwise(m.conv1, _kernel_norm(inp, m.norm1.eps), fused)
    x = nafnet_gate.depthwise_gate(x, m.gate_weight, m.conv2.bias)  # ty: ignore[invalid-argument-type]
    y = _pointwise(m.conv3, _scale_features(m, x), fused, inp)
    del x
    height, width = y.shape[-2:]
    row_bytes = y.shape[0] * width * m.conv4.out_channels * y.element_size()
    rows = max(1, _FFN_TILE_BYTES // row_bytes)
    if rows >= height:
        return _finish_fused_gate(m, y, fused)
    rows = min(rows, _FFN_TILE_ROWS)
    for start in range(0, height, rows):
        dest = y[:, :, start : start + rows]
        result = _finish_fused_gate(m, dest, fused)
        if result.data_ptr() != dest.data_ptr():
            dest.copy_(result)  # interleaved batches need a packed tile for the GEMMs
    return y


def _finish_fused_gate(m: NAFBlock, y: torch.Tensor, fused: dpt_fused.DptFused) -> torch.Tensor:
    """Finish the pointwise gate and reuse its freshly computed residual as the destination."""
    a, b = _pointwise(m.conv4, _kernel_norm(y, m.norm2.eps), fused).chunk(2, 1)
    gated = a * b
    del a, b
    return _pointwise(m.conv5, gated, fused, y, overwrite_residual=True)


class _Shuffle(torch.nn.Module):
    """Write a 1x1 expansion directly into the 2x pixel-shuffled NHWC output."""

    weight: torch.Tensor

    def __init__(self, conv: torch.nn.Conv2d) -> None:
        super().__init__()
        matrix = conv.weight.detach().flatten(1)
        # PixelShuffle stores (channel, dy, dx). The GEMM writes (dy, dx, channel).
        packed = matrix.reshape(-1, 4, matrix.shape[1]).transpose(0, 1).reshape_as(matrix).contiguous()
        self.register_buffer("weight", packed, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.permute(0, 2, 3, 1).contiguous()
        y = dpt_fused.load(x.device).gemm(rows, self.weight, shuffle=(rows.shape[1], rows.shape[2], 2, 2))
        return y.permute(0, 3, 1, 2)


def _restore(net: NAFNet, image: torch.Tensor) -> torch.Tensor:
    """Consume each skip after its last use, adding it into the upsampler's new output."""
    h, w = image.shape[-2:]
    padded = net.check_image_size(image)
    x = net.intro(padded)
    skips = []
    for encoder, down in zip(net.encoders, net.downs, strict=True):
        x = encoder(x)
        skips.append(x)
        x = down(x)
    x = net.middle_blks(x)
    for decoder, up in zip(net.decoders, net.ups, strict=True):
        x = decoder(up(x).add_(skips.pop()))
    return net.ending(x).add_(padded)[:, :, :h, :w]


class Kernels(Portable):
    """Channels-last fp16 convolutions with a fused depthwise convolution and SimpleGate."""

    module = Runner.module  # fp16-accumulate convolutions are cuDNN plans, not exportable operators
    _GRAPH_PIXELS = 256 * 384  # fixed: small restoration patches where replay reduces launch overhead

    def __init__(self, ctx: Context) -> None:
        """Build the network and route its dense convolutions through `cudnn_hacc`."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        fused = dpt_fused.load(ctx.device) if ctx.params.width % 8 == 0 else None
        for m in self.net.modules():
            if isinstance(m, NAFBlock):
                # Pack the 3 x 3 filters once in the order neighboring CUDA lanes read.
                weight = m.conv2.weight.detach().flatten(1).T.contiguous()
                m.register_buffer("gate_weight", weight, persistent=False)
                if fused is None:  # custom checkpoints can have channels outside tensor-core alignment
                    override(m, _gated_block)
                else:
                    override(m, _fused_block)
            elif is_base(m, torch.nn.Conv2d) and m.groups == 1:
                override(m, _hacc_conv)

        if fused is not None:
            for i, stage in enumerate(self.net.ups):
                self.net.ups[i] = _Shuffle(stage[0])  # ty: ignore[not-subscriptable] - NAFNet builds Sequential upsamplers
            override(self.net, _restore)
        self._graphs = SmallShapeGraphs(
            self._restore_image,
            pixels=self._GRAPH_PIXELS,
            enabled=ctx.params.cuda_graphs,
            capture_after=cudnn_hacc.capture_after(),
            one_shape=True,
            capture_warmup=cudnn_hacc.BUILD_AFTER > 0,
        )

    def _restore_image(self, image: torch.Tensor) -> torch.Tensor:
        with fp16_accumulate():
            return self.net(image)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The restored batch."""
        x = image.to(torch.float16).contiguous(memory_format=self.fmt)
        return {"image": self._graphs(x)}

    def release_graphs(self) -> None:
        """Free the CUDA graph and its memory pool."""
        self._graphs.release()


class _MlxPortable(Portable):
    """Apple MLX: the prepared network written in `mlx.core` operations (`mlx_net`)."""

    net: Any  # mlx_net.MlxNAFNet
    module = Runner.module  # not a PyTorch module

    def __init__(self, ctx: Context) -> None:
        """Build and prepare the network on the CPU in float32 (`Portable`), then move it to MLX."""
        super().__init__(ctx)
        from . import mlx_net  # noqa: PLC0415 - imports mlx

        self.net = mlx_net.MlxNAFNet(self.net, mlx_wrap.run_dtype(ctx))

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The restored batch."""
        from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - imports mlx

        return {"image": to_torch(self.net(to_mx(image.float())))}


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(_MlxPortable, attrs=(), cache_mb=1024),
}

_BLOCK = re.compile(r"(encoders|decoders)\.(\d+)\.(\d+)\.beta|middle_blks\.(\d+)\.beta")


def from_checkpoint(sd: Mapping[str, torch.Tensor]) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """`load(path)`: a NAFNet state dict as a variant and its params; keys unchanged.

    The deblurring structure (encoder `[1, 1, 1, 28]`, one middle block: GoPro, REDS) runs as `NAFNetLocal` with
    training size 256 (upstream's test options), any other as `NAFNet`.
    """
    if "intro.weight" not in sd or "ending.weight" not in sd or "encoders.0.0.conv1.weight" not in sd:
        return None
    if sd["intro.weight"].shape[1] != 3:  # noqa: PLR2004 - RGB
        msg = f"NAFNet checkpoint with {sd['intro.weight'].shape[1]} input channels: only RGB models are supported"
        raise WeightsError(msg)
    enc: dict[int, int] = {}
    dec: dict[int, int] = {}
    middle = 0
    for k in sd:
        if m := _BLOCK.fullmatch(k):
            if m.group(4) is not None:
                middle = max(middle, int(m.group(4)) + 1)
            else:
                d = enc if m.group(1) == "encoders" else dec
                d[int(m.group(2))] = max(d.get(int(m.group(2)), 0), int(m.group(3)) + 1)
    enc_blocks = tuple(enc[i] for i in range(len(enc)))
    deblur = middle == 1 and enc_blocks[-1:] == (28,)
    params = {
        "scale": 1,
        "width": sd["intro.weight"].shape[0],
        "enc_blocks": enc_blocks,
        "middle_blocks": middle,
        "dec_blocks": tuple(dec[i] for i in range(len(dec))),
        "tlc": 256 if deblur else 0,
    }
    return "nafnet-gopro" if deblur else "nafnet-sidd", params, dict(sd)
