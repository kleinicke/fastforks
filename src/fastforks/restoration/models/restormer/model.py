"""Restormer runners: the network of swz30/Restormer and its test call (reflect padding to a multiple of 8, `model`).

The test scripts (`Motion_Deblurring/test.py`, `Deraining/test.py`) reflect-pad bottom / right to a multiple of 8, run
the network, clamp and crop; `pad` does the same (`test_real_denoising_sidd.py` runs 256 x 256 blocks without padding,
the same call). Each transformer block's LayerNorm runs over the channel dimension without its affine, which is folded
into the 1 x 1 conv after it (`qkv`, `project_in`; WithBias: `F.layer_norm` over a channels-last view, BiasFree: the
input divided by its channel standard deviation, no mean subtraction, as upstream). The transposed (channel) attention
normalises q and k in float32 (upstream: `F.normalize` in the input dtype, which can overflow in float16 on large
images), multiplies q by the head's temperature and runs the `[c, c]` softmax in float32. On CUDA the network runs
channels-last.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks._core.errors import WeightsError
from fastforks._core.patch import is_base, override
from fastforks.kernels import cudnn_hacc, fastbb
from fastforks.restoration._runner import Crop, Forward, Runner

from .net import Restormer
from .net.restormer_arch import Attention, FeedForward, TransformerBlock, WithBias_LayerNorm

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks.restoration._runner import Context

_EPS = 1e-5  # fixed: upstream's LayerNorm epsilon (hard-coded in restormer_arch.py)


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params` (the test options' `network_g`)."""

    scale: int  # 1
    dim: int  # 48: channels at full resolution
    blocks: tuple[int, ...]  # transformer blocks per level (the last: the latent level)
    refinement: int  # refinement blocks at full resolution
    heads: tuple[int, ...]  # attention heads per level
    ffn: float  # 2.66: GDFN expansion factor
    norm: str  # "WithBias" or "BiasFree" (real denoising)
    key: str  # checkpoint key ("params")
    multiple: int  # 8: the test scripts' reflect padding


def _gram(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """`q^T k` of `[B, N, C]` token views as `[B, C, C]` float32 (half inputs: float32 GEMM output, no overflow)."""
    if q.dtype == torch.float32:
        return torch.bmm(q.mT, k)
    if torch.compiler.is_exporting() or q.device.type == "mps":  # ONNX and MPS have no float32-output half GEMM
        return torch.bmm(q.mT.float(), k.float())
    return torch.bmm(q.mT, k, out_dtype=torch.float32)


def _attention(m: Attention, x: torch.Tensor) -> torch.Tensor:
    """`Attention.forward` (MDTA) on the token view of the `qkv` output, with `project_out` folded into the product.

    Upstream reshapes q / k / v to `[B, head, c, H W]`, normalises q and k over the pixels, multiplies the per-head
    `[c, c]` products by the temperature, takes the softmax and multiplies v, then runs `project_out`. Here the full
    `[C, C]` product `q^T k` is one float32-output GEMM over the `[B, H W, 3 C]` token view (free on channels-last),
    divided by the float32 norms, scaled and masked to the head blocks (`-inf` elsewhere); after the softmax the
    `project_out` weight is multiplied in, so the output is one GEMM `v (W A)^T` straight into channels-last tokens.
    """
    b, c, h, w = x.shape
    t = m.qkv_dwconv(m.qkv(x)).permute(0, 2, 3, 1).reshape(b, h * w, 3 * c)
    q, k, v = t.split(c, -1)
    f32 = torch.float32
    nq = torch.linalg.vector_norm(q, dim=1, dtype=f32).clamp_min(1e-12)
    nk = torch.linalg.vector_norm(k, dim=1, dtype=f32).clamp_min(1e-12)
    scale: torch.Tensor = m.scale  # ty: ignore[invalid-assignment]
    block: torch.Tensor = m.block  # ty: ignore[invalid-assignment]
    proj: torch.Tensor = m.proj  # ty: ignore[invalid-assignment]
    g = _gram(q, k) * (scale / nq)[:, :, None] / nk[:, None, :] + block
    mat = (proj @ g.softmax(-1)).to(v.dtype)  # [B, C out, C in]
    return (v @ mat.mT).view(b, h, w, c).permute(0, 3, 1, 2)


def _fold_attention(m: Attention, device: torch.device | str) -> None:
    """Per-channel temperature, head-block mask and `project_out` weight for `_attention`, float32 on `device`.

    Plain attributes, not buffers: the network's cast to float16 must not round them.
    """
    c = m.project_out.in_channels
    head = c // m.num_heads
    temperature: torch.Tensor = m.temperature
    ids = torch.arange(c) // head
    m.scale = temperature.detach().view(-1).repeat_interleave(head).float().to(device)
    m.block = torch.zeros(c, c).masked_fill_(ids[:, None] != ids[None, :], float("-inf")).to(device)
    m.proj = m.project_out.weight.detach()[:, :, 0, 0].float().to(device)
    del m.temperature, m.project_out
    override(m, _attention)


def _head_attention(m: Attention, x: torch.Tensor) -> torch.Tensor:
    """Form only the per-head channel products, then fold the head projections into one value GEMM.

    Iterate over images so the token views of interleaved Q/K/V remain strided views even for B > 1.
    """
    b, c, h, w = x.shape
    heads, width = m.num_heads, c // m.num_heads
    tokens = m.qkv_dwconv(m.qkv(x)).permute(0, 2, 3, 1).reshape(b, h * w, 3 * c)
    norms = torch.linalg.vector_norm(tokens[..., : 2 * c], dim=1, dtype=torch.float32).clamp_min_(1e-12)
    temperature: torch.Tensor = m.scale  # ty: ignore[invalid-assignment]
    proj: torch.Tensor = m.proj  # ty: ignore[invalid-assignment]
    matrices = []
    for values, norm in zip(tokens.unbind(), norms.unbind(), strict=True):
        q, k, _ = values.view(h * w, 3, heads, width).permute(1, 2, 0, 3).unbind()
        nq, nk = norm.view(2, heads, width).unbind()
        gram = _gram(q, k) * (temperature / nq).unsqueeze(-1) / nk.unsqueeze(-2)
        matrices.append(torch.bmm(gram.softmax(-1).mT, proj).reshape(c, c))
    mat = (matrices[0].unsqueeze(0) if b == 1 else torch.stack(matrices)).to(tokens.dtype)
    return torch.bmm(tokens[..., 2 * c :], mat).view(b, h, w, c).permute(0, 3, 1, 2)


def _fold_head_attention(m: Attention, device: torch.device | str) -> None:
    heads = m.num_heads
    c = m.project_out.in_channels
    m.scale = m.temperature.detach().view(heads, 1).float().to(device)
    m.proj = m.project_out.weight.detach()[:, :, 0, 0].T.reshape(heads, c // heads, c).float().contiguous().to(device)
    del m.temperature, m.project_out
    override(m, _head_attention)


def _block(m: TransformerBlock, x: torch.Tensor) -> torch.Tensor:
    """`TransformerBlock.forward` with the norms' affine folded into `qkv` / `project_in`."""
    if x.is_cuda:  # PixelShuffle gives NCHW strides; a mixed-stride add is 8x slower
        x = x.contiguous(memory_format=torch.channels_last)
    x = x + m.attn(m.norm1(x))
    return x + m.ffn(m.norm2(x))


def _pad_ffn(m: FeedForward) -> None:
    """Zero-pad the GDFN's hidden channels (int(2.66 C): 127, 255, 510, 1021) to a multiple of 8.

    Odd channel counts make cuDNN pad every channels-last tensor of the block; the padded channels are zero after
    `project_in` and the depthwise conv (zero weights and bias), stay zero through `gelu(x1) * x2` and have zero
    `project_out` columns.
    """
    hidden = m.project_out.in_channels
    padded = -(-hidden // 8) * 8
    if padded == hidden:
        return
    rows = torch.cat([torch.arange(hidden), padded + torch.arange(hidden)])  # x1 / x2 halves at 0 / padded

    def conv(old: torch.nn.Conv2d, cin: int, cout: int, groups: int = 1) -> torch.nn.Conv2d:
        k = old.kernel_size[0]
        new = torch.nn.Conv2d(cin, cout, k, padding=k // 2, groups=groups, bias=old.bias is not None)
        with torch.no_grad():
            new.weight.zero_()
            if new.bias is not None:
                new.bias.zero_()
        return new.requires_grad_(requires_grad=False)

    pin, dw, pout = m.project_in, m.dwconv, m.project_out
    m.project_in = conv(pin, pin.in_channels, 2 * padded)
    m.dwconv = conv(dw, 2 * padded, 2 * padded, groups=2 * padded)
    m.project_out = conv(pout, padded, pout.out_channels)
    with torch.no_grad():
        m.project_in.weight[rows] = pin.weight
        if pin.bias is not None and m.project_in.bias is not None:
            m.project_in.bias[rows] = pin.bias
        m.dwconv.weight[rows] = dw.weight
        if dw.bias is not None and m.dwconv.bias is not None:
            m.dwconv.bias[rows] = dw.bias
        m.project_out.weight[:, :hidden] = pout.weight
        if pout.bias is not None and m.project_out.bias is not None:
            m.project_out.bias.copy_(pout.bias)


def _split_ffn(m: FeedForward, _device: torch.device | str) -> None:
    """Split `project_in` and the depthwise conv into their x1 / x2 halves (`_ffn`)."""
    pin, dw = m.project_in, m.dwconv
    half = dw.out_channels // 2

    def part(old: torch.nn.Conv2d, cin: int, sl: slice, groups: int = 1) -> torch.nn.Conv2d:
        k = old.kernel_size[0]
        new = torch.nn.Conv2d(cin, half, k, padding=k // 2, groups=groups, bias=old.bias is not None)
        with torch.no_grad():
            new.weight.copy_(old.weight[sl])
            if old.bias is not None and new.bias is not None:
                new.bias.copy_(old.bias[sl])
        return new.requires_grad_(requires_grad=False)

    for i, sl in enumerate((slice(0, half), slice(half, 2 * half)), 1):
        setattr(m, f"in{i}", part(pin, pin.in_channels, sl))
        setattr(m, f"dw{i}", part(dw, half, sl, groups=half))
    del m.project_in, m.dwconv
    override(m, _ffn)


def _ffn(m: FeedForward, x: torch.Tensor) -> torch.Tensor:
    """`FeedForward.forward` (GDFN) one half at a time: `gelu(x1)` is ready before x2's `project_in` output exists.

    Upstream holds the whole `project_in` and depthwise outputs at once (4 hidden-width tensors at the peak), this 3.
    """
    g = F.gelu(m.dw1(m.in1(x)))  # ty: ignore[call-non-callable]
    return m.project_out(g.mul_(m.dw2(m.in2(x))))  # ty: ignore[call-non-callable]


def _gate_ffn(m: FeedForward, _device: torch.device | str) -> None:
    """Pack the depthwise filters before casting the network. dwgate reads fp32 weights in kernel-position order."""
    dw = m.dwconv
    m.gate_weight = dw.weight.detach().reshape(dw.out_channels, 9).T.contiguous().to(_device)
    m.gate_bias = None if dw.bias is None else dw.bias.detach().to(_device)  # ty: ignore[invalid-assignment]
    del m.dwconv
    override(m, _fused_ffn)


def _fused_ffn(m: FeedForward, x: torch.Tensor) -> torch.Tensor:
    projected = m.project_in(x).permute(0, 2, 3, 1)
    gated = fastbb.dwgate(projected, m.gate_weight, m.gate_bias)  # ty: ignore[invalid-argument-type]
    return m.project_out(gated.permute(0, 3, 1, 2))


def _fold_norm(norm: torch.nn.Module, conv: torch.nn.Conv2d) -> torch.nn.Module:
    """Fold the LayerNorm's affine into the bias-free 1 x 1 `conv` after it; returns the bare norm marker."""
    body: Any = norm.body
    with_bias = isinstance(body, WithBias_LayerNorm)
    with torch.no_grad():
        w = conv.weight[:, :, 0, 0]
        bias = w @ body.bias if with_bias else torch.zeros(w.shape[0], dtype=w.dtype)
        conv.weight.mul_(body.weight[None, :, None, None])
        conv.bias = torch.nn.Parameter(bias, requires_grad=False)
    return _Norm(with_bias=with_bias)


class _Norm(torch.nn.Module):
    """A folded LayerNorm: only its kind is left."""

    def __init__(self, *, with_bias: bool) -> None:
        super().__init__()
        self.with_bias = with_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the channels after folding the affine into the next convolution."""
        if self.with_bias:
            y = F.layer_norm(x.permute(0, 2, 3, 1), x.shape[1:2], eps=_EPS).permute(0, 3, 1, 2)
            return y.contiguous() if x.device.type == "cpu" else y  # channels-last slows the CPU convolutions
        var = x.float().var(1, keepdim=True, unbiased=False)
        return x * torch.rsqrt(var + _EPS).to(x.dtype)


class _CudaNorm(_Norm):
    """Read and write the channels-last activation in one CUDA pass."""

    def __init__(self, norm: _Norm, unit_weight: torch.Tensor) -> None:
        super().__init__(with_bias=norm.with_bias)
        self.register_buffer("unit_weight", unit_weight, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a half-precision activation with the affine already folded."""
        tokens = x.permute(0, 2, 3, 1).contiguous()
        weight = self.get_buffer("unit_weight")
        return fastbb.ln_lastdim(tokens, weight, None, _EPS, bias_free=not self.with_bias).permute(0, 3, 1, 2)


def _restore(net: Restormer, image: torch.Tensor) -> torch.Tensor:
    """Release each eager CUDA encoder feature after its last decoder use."""
    if not image.is_cuda or torch.is_grad_enabled() or torch.compiler.is_compiling() or torch.jit.is_tracing():
        return net(image)

    first = net.patch_embed(image)
    level1 = net.encoder_level1(first)
    if not net.dual_pixel_task:
        first = None
    level2 = net.encoder_level2(net.down1_2(level1))
    level3 = net.encoder_level3(net.down2_3(level2))
    x = net.latent(net.down3_4(level3))

    x = torch.cat((net.up4_3(x), level3), dim=1)
    del level3
    x = net.decoder_level3(net.reduce_chan_level3(x))
    x = torch.cat((net.up3_2(x), level2), dim=1)
    del level2
    x = net.decoder_level2(net.reduce_chan_level2(x))
    x = torch.cat((net.up2_1(x), level1), dim=1)
    del level1
    x = net.refinement(net.decoder_level1(x))
    if first is not None:
        return net.output(x + net.skip_conv(first))
    return net.output(x).add_(image)


class Portable(Runner):
    """Plain PyTorch in the card's dtype, channels-last on CUDA."""

    dtypes = ("float32", "float16", "bfloat16")
    Params = Params
    _prepare_attention = staticmethod(_fold_attention)
    _prepare_ffn = staticmethod(_split_ffn)

    def __init__(self, ctx: Context) -> None:
        """Build the network, fold the norms."""
        super().__init__(ctx)
        p = ctx.params
        self.net = Restormer(
            dim=p.dim, num_blocks=list(p.blocks), num_refinement_blocks=p.refinement, heads=list(p.heads),
            ffn_expansion_factor=p.ffn, bias=False, LayerNorm_type=p.norm,
        )  # fmt: skip
        load_strict(self.net, ctx.state_dict(key=p.key), f"{ctx.variant.ref} checkpoint")
        for m in self.net.modules():
            if isinstance(m, TransformerBlock):
                m.norm1 = _fold_norm(m.norm1, m.attn.qkv)  # ty: ignore[invalid-assignment]
                m.norm2 = _fold_norm(m.norm2, m.ffn.project_in)  # ty: ignore[invalid-assignment]
                override(m, _block)
                self._prepare_attention(m.attn, ctx.device)
                _pad_ffn(m.ffn)
                self._prepare_ffn(m.ffn, ctx.device)
        self.fmt = torch.channels_last if torch.device(ctx.device).type == "cuda" else torch.contiguous_format
        self.net.eval().to(ctx.device, getattr(torch, ctx.dtype)).to(memory_format=self.fmt)

    def pad(self, x: torch.Tensor) -> tuple[torch.Tensor, Crop]:
        """The test scripts: reflect-pad bottom / right to a multiple of 8."""
        h, w = x.shape[-2:]
        k = self.ctx.params.multiple
        x = F.pad(x, (0, -w % k, 0, -h % k), "reflect")
        return x, Crop(0, 0, h, w)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The restored batch."""
        x = image.to(getattr(torch, self.ctx.dtype)).contiguous(memory_format=self.fmt)
        return {"image": _restore(self.net, x)}

    def module(self) -> torch.nn.Module:
        """The network in the card's dtype."""
        dtype, fmt = getattr(torch, self.ctx.dtype), self.fmt
        return Forward(self.net, lambda net, image: net(image.to(dtype).contiguous(memory_format=fmt)))


def _hacc_conv(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    return cudnn_hacc.conv_half_acc(x, conv.weight, conv.bias, conv.padding, conv.stride, conv.dilation)  # ty: ignore[invalid-argument-type]


class Kernels(Portable):
    """float16, channels-last, fp16-accumulate convolutions (`cudnn_hacc`; the depthwise convs stay PyTorch's).

    The attention GEMMs keep float32 accumulation (the `q^T k` sums run over all pixels).
    """

    _prepare_attention = staticmethod(_fold_head_attention)
    _prepare_ffn = staticmethod(_gate_ffn)
    module = Runner.module  # fp16-accumulate convolutions are cuDNN plans, not exportable operators

    def __init__(self, ctx: Context) -> None:
        """Build the network and route its dense convolutions through `cudnn_hacc`."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        fastbb.load(ctx.device)
        unit_weights: dict[int, torch.Tensor] = {}
        for m in self.net.modules():
            if is_base(m, torch.nn.Conv2d) and m.groups == 1:
                override(m, _hacc_conv)
            elif isinstance(m, TransformerBlock):
                channels = m.attn.qkv.in_channels
                if channels % 8 or channels > fastbb.LN_MAX_C:
                    continue  # arbitrary-width checkpoint variants use the generic channel normalization
                if channels not in unit_weights:
                    unit_weights[channels] = torch.ones(channels, device=ctx.device, dtype=torch.float32)
                for name in ("norm1", "norm2"):
                    m.register_module(name, _CudaNorm(getattr(m, name), unit_weights[channels]))

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The restored batch."""
        x = image.to(torch.float16).contiguous(memory_format=self.fmt)
        return {"image": _restore(self.net, x)}


RUNNERS = {"portable": Portable, "kernels": Kernels}


def from_checkpoint(sd: Mapping[str, torch.Tensor]) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """`load(path)`: a Restormer state dict as a variant and its params; keys unchanged.

    BiasFree norms (no `norm1.body.bias`) are the real-denoising model, WithBias ones the deblurring one (the deraining
    model has the same structure; pick its variant by name).
    """
    if "patch_embed.proj.weight" not in sd or "encoder_level1.0.attn.temperature" not in sd:
        return None
    if "skip_conv.weight" in sd or sd["patch_embed.proj.weight"].shape[1] != 3:  # noqa: PLR2004 - RGB
        msg = "Restormer checkpoint with a dual-pixel or non-RGB input: only RGB models are supported"
        raise WeightsError(msg)

    def count(prefix: str) -> int:
        return len({k.split(".")[1] for k in sd if k.startswith(prefix + ".")})

    levels = ("encoder_level1", "encoder_level2", "encoder_level3", "latent")
    biased = "encoder_level1.0.norm1.body.bias" in sd
    params = {
        "scale": 1,
        "dim": sd["patch_embed.proj.weight"].shape[0],
        "blocks": tuple(count(n) for n in levels),
        "refinement": count("refinement"),
        "heads": tuple(sd[f"{n}.0.attn.temperature"].shape[0] for n in levels),
        "norm": "WithBias" if biased else "BiasFree",
    }
    hidden, dim = sd["encoder_level1.0.ffn.project_out.weight"].shape[1], params["dim"]
    params["ffn"] = 2.66 if int(dim * 2.66) == hidden else (hidden + 0.5) / dim  # upstream: int(dim * ffn) channels
    return "restormer-deblur" if biased else "restormer-sidd", params, dict(sd)
