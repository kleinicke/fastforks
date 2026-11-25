"""DRCT runners: the network of ming053l/DRCT and `drct/models/drct_model.py`'s test call (reflect pad, `net_g(img)`).

`DRCTModel.pre_process` reflect-pads the input bottom / right to a multiple of the window size; `pad` does the same.
Upstream builds the shifted-window mask on the CPU on every call at sizes other than the training one and gathers the
relative position bias on every call; here the mask is made on the device once per size, the bias once at load, and
the window attention is one `scaled_dot_product_attention` (`fastforks.shared.swin_sr`). The dense layers' head sizes
(30, 53, 122, 46, 77 for 180 channels and 6 heads) are zero-padded to multiples of 8 in the weights so the fused
kernel runs. Each residual dense group turns its tokens into a feature map and back around every 1 x 1 `adjust` conv
(`PatchUnEmbed`, conv, `PatchEmbed`: two transposes per layer); here the 1 x 1 convs are `F.linear` on the tokens
(the same products). The input mean is a buffer (upstream: a plain attribute, `type_as` per call).

float16 needs no rescaling: the largest group output on the golden image is 280 (DRCT) / 951 (DRCT-L).
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks._core.errors import WeightsError
from fastforks._core.half import fp16_accumulate
from fastforks._core.kernels import linear
from fastforks._core.mps import release
from fastforks._core.patch import is_base, override
from fastforks.kernels import cudnn_hacc, shift_windows, sr_norm
from fastforks.restoration._runner import Crop, Forward, Runner
from fastforks.shared import mlx_wrap
from fastforks.shared.sr_layer_norm import fast_layer_norm
from fastforks.shared.swin_sr import (
    attention,
    checkpoint_upsampler,
    pad_heads,
    patch_embed,
    relative_bias,
    shift_mask,
    window_attention,
)

from .net import DRCT
from .net.drct_arch import RDG, PatchEmbed, SwinTransformerBlock, WindowAttention

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks.restoration._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params` (the `network_g` section of the test options)."""

    scale: int  # 4
    embed_dim: int  # 180
    heads: tuple[int, ...]  # attention heads per residual dense group (6 groups: DRCT, 12: DRCT-L)
    img_size: int  # training patch size (the size of the stored attention masks)
    key: str  # checkpoint key ("params_ema")
    window: int  # 16: attention window (all released models)


class _NoMask:
    """An unshifted block's mask: upstream's is all zeros and gets `.to(device)`; here attention runs without one."""

    def to(self, *_: object) -> None:
        return None


_NO_MASK = _NoMask()
_LARGE_IMAGE_PIXELS = 256**2
_WORKSPACE_BYTES = 64 * 1024**2
_WINDOW_BYTES = 64 << 20
_WINDOW_STREAM_THRESHOLD = 256 << 20
_MLP_BYTES = 192 << 20


@dataclass(frozen=True)
class _BoundaryMask:
    """The geometry of a shifted mask, with materialization deferred to attention."""

    size: tuple[int, int]
    window: int
    shift: int

    def to(self, *_: object) -> _BoundaryMask:
        """Keep the geometry when the vendored block moves its mask to the device."""
        return self


def _eager_inference() -> bool:
    return not (
        torch.compiler.is_compiling()
        or torch.jit.is_tracing()
        or torch.is_grad_enabled()
        or torch.is_autocast_enabled("cuda")
    )


def _mask(m: SwinTransformerBlock, x_size: tuple[int, int]) -> torch.Tensor | _NoMask:
    """`SwinTransformerBlock.calculate_mask` made on the device, once per size (upstream: all zeros if unshifted)."""
    if not m.shift_size:
        return _NO_MASK
    return shift_mask(*x_size, m.window_size, m.shift_size, m.norm1.weight.device, dtype=m.norm1.weight.dtype)


def _bounded_mask(m: SwinTransformerBlock, x_size: tuple[int, int]) -> torch.Tensor | _NoMask | _BoundaryMask:
    """Describe the shifted mask on large FP16 CUDA and MPS inputs."""
    if m.shift_size and x_size[0] * x_size[1] >= _LARGE_IMAGE_PIXELS and _eager_inference():
        return _BoundaryMask(x_size, m.window_size, m.shift_size)
    return _mask(m, x_size)


def _attention(m: WindowAttention, x: torch.Tensor, mask: torch.Tensor | _BoundaryMask | None = None) -> torch.Tensor:
    """`WindowAttention.forward` with the bias gathered at load."""
    if isinstance(mask, _BoundaryMask):
        return _bounded_shift_attention(m, x, mask)
    return window_attention(x, m.qkv, m.proj, heads=m.num_heads, scale=m.scale, bias=m.bias, mask=mask)  # ty: ignore[invalid-argument-type]


def _bounded_shift_attention(m: WindowAttention, x: torch.Tensor, mask: _BoundaryMask) -> torch.Tensor:
    """Project all windows together and expand only a group of shifted biases at a time."""
    count, tokens, _ = x.shape
    heads = m.num_heads
    dim = m.qkv.out_features // (3 * heads)
    q, k, v = m.qkv(x).reshape(count, tokens, 3, heads, dim).permute(2, 0, 3, 1, 4).unbind(0)
    win = mask.window
    patterns = shift_mask(2 * win, 2 * win, win, mask.shift, x.device, dtype=q.dtype)
    biases = m.get_buffer("bias").to(q.dtype)[None] + patterns[:, None]
    rows, cols = (side // win for side in mask.size)
    row_kind = (torch.arange(rows, device=x.device) == rows - 1).long() * 2
    col_kind = (torch.arange(cols, device=x.device) == cols - 1).long()
    kinds = (row_kind[:, None] + col_kind).flatten().repeat(count // (rows * cols))
    step = max(1, _WORKSPACE_BYTES // (heads * tokens**2 * q.element_size()))
    out = q.new_empty((count, tokens, heads, dim))
    for start in range(0, count, step):
        section = slice(start, start + step)
        add = biases.index_select(0, kinds[section])
        out[section] = attention(q[section], k[section], v[section], add, m.scale).transpose(1, 2)
    return m.proj(out.reshape(count, tokens, -1))


def _strip_reconstruction(m: DRCT, x: torch.Tensor, rows: int) -> torch.Tensor:
    """Upscale each strip with the two low-resolution halo rows needed by the three convolutions."""
    batch, _, h, w = x.shape
    out = x.new_empty((batch, m.conv_last.out_channels, 4 * h, 4 * w))
    for start in range(0, h, rows):
        end = min(start + rows, h)
        lo, hi = max(0, start - 2), min(h, end + 2)
        tile = m.conv_last(m.upsample(x[:, :, lo:hi]))
        out[:, :, 4 * start : 4 * end].copy_(tile[:, :, 4 * (start - lo) : 4 * (end - lo)])
    return out


def _forward(m: DRCT, image: torch.Tensor) -> torch.Tensor:
    """Bound the 4x reconstruction workspace on large FP16 CUDA and MPS inputs."""
    if not _eager_inference():
        return DRCT.forward(m, image)
    batch, _, h, w = image.shape
    row_bytes = batch * m.conv_last.in_channels * w * 16 * image.element_size()
    rows = max(1, _WORKSPACE_BYTES // row_bytes)
    if rows >= h:
        return DRCT.forward(m, image)
    mean = m.mean.type_as(image)
    x = m.conv_first((image - mean) * m.img_range)
    x = m.conv_after_body(m.forward_features(x)) + x
    x = m.conv_before_upsample(x)
    return _strip_reconstruction(m, x, rows) / m.img_range + mean


def _rdg(m: RDG, x: torch.Tensor, xsize: tuple[int, int], *, reuse_outputs: bool = False) -> torch.Tensor:
    """`RDG.forward` with the 1 x 1 `adjust` convs as `F.linear` on the tokens (upstream: on an unembedded map)."""
    feats = [x]
    for k in range(1, 5):
        part = _adjust(m, k, torch.cat(feats, -1) if k > 1 else x, xsize)
        feats.append(F.leaky_relu(part, 0.2, inplace=reuse_outputs))
    dense = torch.cat(feats, -1)
    del feats, part
    result = _adjust(m, 5, dense, xsize)  # the last `adjust` has no LeakyReLU
    return result.mul_(0.2).add_(x) if reuse_outputs else result * 0.2 + x


def _cuda_rdg(m: RDG, x: torch.Tensor, xsize: tuple[int, int]) -> torch.Tensor:
    return _rdg(m, x, xsize, reuse_outputs=True)


def _adjust(m: RDG, k: int, x: torch.Tensor, xsize: tuple[int, int]) -> torch.Tensor:
    """`adjust<k>(swin<k>(x))` of a group, the 1 x 1 conv on the tokens.

    On MPS the buffers the previous layer left cached are released first: every layer of a group has its own width, so
    none of them is reused and they pile up to the memory limit (512 x 512 float32: 11.2 GiB allocated, 1 GiB live).
    """
    conv = m.get_submodule(f"adjust{k}")
    weight = conv.get_parameter("weight").flatten(1)
    return F.linear(m.get_submodule(f"swin{k}")(release(x), xsize), weight, conv.get_parameter("bias"))


class Portable(Runner):
    """Plain PyTorch in the card's dtype."""

    dtypes = ("float32", "float16", "bfloat16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network, gather the relative position biases, pad the heads, cache the masks per size."""
        super().__init__(ctx)
        p = ctx.params
        kind = torch.device(ctx.device).type
        # MPS, any dtype: a 512x512 input's shifted masks, score blocks and 4x reconstruction pass its 11.8 GiB limit
        bound_workspace = p.scale == 4 and (kind == "mps" or (ctx.dtype == "float16" and kind == "cuda"))  # noqa: PLR2004
        with warnings.catch_warnings():  # upstream's torch.meshgrid without `indexing` ("ij", the default) warns
            warnings.filterwarnings("ignore", "torch.meshgrid", UserWarning)
            self.net = DRCT(
                upscale=p.scale, in_chans=3, img_size=p.img_size, window_size=p.window, img_range=1.0,
                depths=[6] * len(p.heads), embed_dim=p.embed_dim, num_heads=list(p.heads), mlp_ratio=2,
                upsampler="pixelshuffle", resi_connection="1conv",
            )  # fmt: skip  # depths: only the stochastic depth rates (training); every group has 5 layers
        load_strict(self.net, ctx.state_dict(key=p.key), f"{ctx.variant.ref} checkpoint")
        for m in self.net.modules():
            if isinstance(m, WindowAttention):
                table, index = m.relative_position_bias_table, m.relative_position_index
                m.register_buffer("bias", relative_bias(table, index))  # ty: ignore[invalid-argument-type]
                del m.relative_position_bias_table, m.relative_position_index
                pad_heads(m.qkv, m.proj, m.num_heads)
                override(m, _attention)
            elif isinstance(m, SwinTransformerBlock):
                override(m, _bounded_mask if bound_workspace else _mask, "calculate_mask")
            elif isinstance(m, RDG):
                override(m, _rdg)
            elif isinstance(m, PatchEmbed):
                override(m, patch_embed)
        mean = self.net.mean
        del self.net.mean
        self.net.register_buffer("mean", mean, persistent=False)
        self.net.eval().to(ctx.device, getattr(torch, ctx.dtype))
        if ctx.device.type == "cuda" and ctx.dtype == "float16":
            # channels-last weights: cuDNN's fp16 convolutions skip the NCHW<->NHWC conversion around each conv
            # (256x256 x4 peak portable 620 -> 596 MB, kernels 495 -> 471 MB, same speed; RTX 5060 Ti, Windows)
            self.net.to(memory_format=torch.channels_last)
        if bound_workspace:
            override(self.net, _forward)

    def pad(self, x: torch.Tensor) -> tuple[torch.Tensor, Crop]:
        """`DRCTModel.pre_process`: reflect-pad bottom / right to a multiple of the window."""
        h, w = x.shape[-2:]
        win = self.ctx.params.window
        x = F.pad(x, (0, -w % win, 0, -h % win), "reflect")
        return x, Crop(0, 0, h, w)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The upscaled batch."""
        return {"image": self.net(image.to(getattr(torch, self.ctx.dtype)))}

    def module(self) -> torch.nn.Module:
        """The network in the card's dtype."""
        dtype = getattr(torch, self.ctx.dtype)
        return Forward(self.net, lambda net, image: net(image.to(dtype)))


def _hacc_conv(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    return cudnn_hacc.conv_half_acc(x, conv.weight, conv.bias, conv.padding, conv.stride, conv.dilation)  # ty: ignore[invalid-argument-type]


def _boundary_attention(
    attn: WindowAttention, windows: torch.Tensor, size: tuple[int, int], win: int, shift: int
) -> torch.Tensor:
    """Share one additive bias per boundary type, avoiding a [num_windows, heads, win², win²] shifted mask."""
    rows, cols = size[0] // win, size[1] // win
    grid = windows.reshape(-1, rows, cols, *windows.shape[1:])
    out = torch.empty_like(grid)
    # A 2-by-2 window grid contains all four mask patterns (interior, right, bottom, corner).
    masks = shift_mask(2 * win, 2 * win, win, shift, windows.device, dtype=windows.dtype)
    for row, r in enumerate((slice(0, -1), slice(-1, None))):
        for col, c in enumerate((slice(0, -1), slice(-1, None))):
            region = grid[:, r, c]
            if region.numel() == 0:
                continue
            kind = 2 * row + col
            bias: torch.Tensor = attn.bias  # ty: ignore[invalid-assignment]
            if kind:
                bias = bias + masks[kind]
            result = window_attention(
                region.reshape(-1, *windows.shape[1:]),
                attn.qkv,
                attn.proj,
                heads=attn.num_heads,
                scale=attn.scale,
                bias=bias,
                mask=None,
            )
            out[:, r, c] = result.reshape_as(region)
    return out.reshape_as(windows)


def _stream_window_residual(m: SwinTransformerBlock, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Pack, normalize and attend to each group of windows before writing its updated shortcut."""
    h, w = size
    width = x.shape[-1]
    win, shift = m.window_size, m.shift_size
    nr, nc = h // win, w // win
    regions = ((0, 0, nr, nc),)
    masks = None
    if shift:
        masks = shift_mask(2 * win, 2 * win, win, shift, x.device, dtype=x.dtype)
        regions = ((0, 0, nr - 1, nc - 1), (0, nc - 1, nr - 1, 1), (nr - 1, 0, 1, nc - 1), (nr - 1, nc - 1, 1, 1))
    k = shift_windows.load(x.device)
    out = torch.empty_like(x)
    step = max(1, _WINDOW_BYTES // (win * win * m.attn.qkv.out_features * x.element_size()))
    nw, nb = m.norm1.get_buffer("_sr_weight"), m.norm1.get_buffer("_sr_bias")
    for kind, (r0, c0, rows, cols) in enumerate(regions):
        bias: torch.Tensor = m.attn.bias  # ty: ignore[invalid-assignment]
        if masks is not None and kind:
            bias = bias + masks[kind]
        for b in range(x.shape[0]):
            for first in range(0, rows * cols, step):
                count = min(step, rows * cols - first)
                windows = x.new_empty((count, win * win, width))
                threads = windows.numel() // 2
                geometry = (h, w, width // 2, win, shift, r0, c0, cols, first, threads)
                k.launch("stream_window2", *linear(threads), x[b], None, windows, *geometry)
                windows = sr_norm.layer_norm(windows, nw, nb, m.norm1.eps)
                result = window_attention(
                    windows, m.attn.qkv, m.attn.proj, heads=m.attn.num_heads, scale=m.attn.scale, bias=bias, mask=None
                )
                del windows
                k.launch("stream_window2", *linear(threads), result, x[b], out[b], *geometry)
                del result
    return out


def _window_block(m: SwinTransformerBlock, x: torch.Tensor, xsize: tuple[int, int]) -> torch.Tensor:
    """Keep wide shifted-window temporaries bounded on large inputs."""
    large = x.numel() * x.element_size() >= _WINDOW_STREAM_THRESHOLD
    if (
        large
        and x.is_contiguous()
        and x.data_ptr() % 4 == 0
        and x.shape[1] == math.prod(xsize)
        and all(side % m.window_size == 0 for side in xsize)
        and hasattr(m.norm1, "_sr_weight")
    ):
        x = _stream_window_residual(m, x, xsize)
    else:
        windows = shift_windows.rearrange(m.norm1(x), xsize, m.window_size, m.shift_size)
        attended = (
            _boundary_attention(m.attn, windows, xsize, m.window_size, m.shift_size)
            if m.shift_size
            else m.attn(windows)
        )
        del windows
        x = shift_windows.reverse_residual(attended, x, None, xsize, window=m.window_size, shift=m.shift_size)
        del attended
    mm = m.mlp
    rows = max(1, _MLP_BYTES // (mm.fc1.out_features * x.element_size()))
    for part in x.flatten(0, 1).split(rows):
        z = torch.ops.aten.gelu_(mm.fc1(m.norm2(part)), approximate=mm.act.approximate)
        part.add_(m.drop_path(mm.drop(mm.fc2(mm.drop(z)))))
        del z
    return x


class Kernels(Portable):
    """float16, fp16-accumulate GEMMs and convolutions (`cudnn_hacc`)."""

    module = Runner.module  # fp16-accumulate convolutions are cuDNN plans, not exportable operators

    def __init__(self, ctx: Context) -> None:
        """Build the network and route its convolutions through `cudnn_hacc`."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        fast_layer_norm(self.net, ctx.device)
        shift_windows.load(ctx.device)
        for m in self.net.modules():
            if isinstance(m, RDG):
                override(m, _cuda_rdg)
            if isinstance(m, SwinTransformerBlock):
                del m.attn_mask
                override(m, _window_block)
            if is_base(m, torch.nn.Conv2d):
                override(m, _hacc_conv)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The upscaled batch."""
        with fp16_accumulate():
            return {"image": self.net(image.to(torch.float16))}


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(Portable),
}

_GROUP = re.compile(r"layers\.(\d+)\.swin1\.attn\.relative_position_bias_table")


def from_checkpoint(sd: Mapping[str, torch.Tensor]) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """`load(path)`: a DRCT state dict as a variant and its params (channels, groups, heads); keys unchanged."""
    groups = sorted(int(m.group(1)) for k in sd if (m := _GROUP.match(k)))
    if "conv_first.weight" not in sd or "layers.0.adjust1.weight" not in sd or not groups:
        return None
    embed_dim = sd["conv_first.weight"].shape[0]
    tables = [sd[f"layers.{n}.swin1.attn.relative_position_bias_table"] for n in groups]
    window = (math.isqrt(tables[0].shape[0]) + 1) // 2
    mask = sd.get("layers.0.swin2.attn_mask")
    up = checkpoint_upsampler(sd)
    problems = []
    if sd["conv_first.weight"].shape[1] != 3:  # noqa: PLR2004 - RGB
        problems.append("not 3 input channels")
    if up is None or up[0] != "pixelshuffle":
        problems.append("an upsampler other than pixelshuffle")
    if sd["layers.0.adjust1.weight"].shape[0] != 32:  # noqa: PLR2004 - upstream's fixed growth `gc`
        problems.append("a growth other than 32 channels")
    if sd["layers.0.swin1.mlp.fc1.weight"].shape[0] != 2 * embed_dim:
        problems.append("an MLP ratio other than 2")
    if "absolute_pos_embed" in sd or "conv_after_body.weight" not in sd:
        problems.append("absolute position embedding or an identity residual connection")
    if problems:
        msg = f"DRCT checkpoint with {', '.join(problems)}: not supported"
        raise WeightsError(msg)
    params = {
        "scale": up[1],  # ty: ignore[not-subscriptable]
        "embed_dim": embed_dim,
        "heads": tuple(t.shape[1] for t in tables),
        "img_size": 64 if mask is None else math.isqrt(mask.shape[0]) * window,
        "window": window,
    }
    return ("drct-l-x4" if len(groups) > 6 else "drct-x4"), params, dict(sd)  # noqa: PLR2004 - DRCT-L: 12 groups
