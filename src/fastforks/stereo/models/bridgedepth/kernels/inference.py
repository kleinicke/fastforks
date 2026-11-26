"""Kernels inference path for BridgeDepth (general: any input size, no per-size compilation / tuning).

The portable path (`net/inference.py`: float16 autocast, float16 LayerNorms) on our kernels:
  - every WindowAttention (MRF inference: 6 x 6 cells x 4 candidates = 144 tokens, refinement: 4 x 4 x 1) as one fused
    kernel (`fastbb` `win_rpe_attn`, from our NMRF / BridgeDepth forks' general path): logits with the
    relative-position embeddings on q / k / v, mask, online softmax and the output in one pass, fp32 math, reading
    the packed qkv projection and writing the [windows, T, C] input of the output projection (no permute copies)
  - the Depth Anything V2 ViT of the mono branch as the fp16 DINOv2 of `fastforks.shared.depth_anything_v2`
    (`use_fast_vit`: fp16-accumulate GEMMs, `fastbb` flash attention)
  - the stereo encoder's instance norms on the `fastbb` norm kernels, fp16 channels-last with the ReLUs and residual
    adds fused (`fastforks.shared.instance_norm.fuse_nmrf_encoder`; Scene Flow 994 x 1482 352 -> 335 ms)
"""

import sys
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import nn

from fastforks._core.errors import BackendUnavailableError
from fastforks._core.patch import override
from fastforks.kernels import fastbb
from fastforks.shared.depth_anything_v2 import use_fast_vit
from fastforks.shared.instance_norm import fuse_nmrf_encoder

from ..net.bridgedepth import BridgeDepth
from ..net.inference import LN16, forward, prepare
from ..net.nmp import WindowAttention

if TYPE_CHECKING:
    from fastforks._core.metal import MetalLibrary


class KernelsWindowAttention(nn.Module):
    """WindowAttention.forward(x, attn_mask) with the attention on `fastbb.win_rpe_attn` (head_dim 32)."""

    table: torch.Tensor

    def __init__(self, a: WindowAttention, attn: Callable[..., torch.Tensor] = fastbb.win_rpe_attn) -> None:
        """`attn`: `fastbb.win_rpe_attn` or a kernel with its signature (the Metal one)."""
        super().__init__()
        self.attn = attn
        if a.window_size[0] != a.window_size[1] or a.dim != 32 * a.num_heads:
            msg = f"BridgeDepth kernels: square windows with head_dim 32 only, got {a.window_size}, {a.dim} / {a.num_heads}"
            raise BackendUnavailableError(msg)
        self.ws, self.heads, self.scale = a.window_size[0], a.num_heads, a.scale
        self.qkv, self.proj = a.qkv, a.proj
        self.register_buffer("table", a.relative_position_enc_table.detach().half().contiguous(), persistent=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        """x [windows, T, C'] -> [windows, T, C]; attn_mask [nW, T, T] (window w: mask[w % nW]) or None."""
        _, t, _ = x.shape
        n = t // (self.ws * self.ws)
        mask = None if attn_mask is None else attn_mask.reshape(-1, t, t).float().contiguous()
        o = self.attn(
            self.qkv(x).contiguous(), self.table, mask, heads=self.heads, ws=self.ws, n=n, grid=(self.ws, self.ws),
            shift=0, scale=self.scale,
        )  # fmt: skip
        return self.proj(o)


def _swap(mod: nn.Module, make: Callable[[WindowAttention], nn.Module]) -> int:
    """Replace every WindowAttention `a` by `make(a)`; the count."""
    count = 0
    for name, c in mod.named_children():
        if isinstance(c, WindowAttention):
            setattr(mod, name, make(c))
            count += 1
        else:
            count += _swap(c, make)
    return count


_NCHW_DIMS = 4
_NCHW_DIMS = 4
_OUT_CONV_MIN_PIXELS = 307_200
_OUT_CONV_BAND_ROWS = 64


def _can_band_out_conv(conv: nn.Conv2d, x: torch.Tensor) -> bool:
    return (
        not (torch.compiler.is_compiling() or torch.jit.is_tracing() or torch.is_grad_enabled() or conv.training)
        and x.is_cuda
        and x.dtype == torch.float32
        and x.ndim == _NCHW_DIMS
        and x.shape[0] == 1
        and x.is_contiguous()
        and x.shape[2] > _OUT_CONV_BAND_ROWS
        and x.shape[2] * x.shape[3] >= _OUT_CONV_MIN_PIXELS
        and conv.weight.dtype == torch.float16
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.float16
    )


def _bands_out_conv(conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    b, _, h, w = x.shape
    out = x.new_empty((b, conv.out_channels, h, w), dtype=torch.float16)
    for start in range(0, h, _OUT_CONV_BAND_ROWS):
        stop = min(start + _OUT_CONV_BAND_ROWS, h)
        band = x[:, :, start:stop].to(dtype=torch.float16, memory_format=torch.contiguous_format)
        out[:, :, start:stop].copy_(nn.Conv2d.forward(conv, band))
        del band
    return out


def _out_conv_forward(conv: nn.Conv2d, input: torch.Tensor) -> torch.Tensor:  # noqa: A002 - Conv2d keyword  # noqa: A002 - Conv2d keyword
    """Bound the FP16 autocast copy and cuDNN workspace of the final mono 1x1 projection."""
    if _can_band_out_conv(conv, input):
        return _bands_out_conv(conv, input)
    return nn.Conv2d.forward(conv, input)


@torch.no_grad()
def prepare_kernels(model: BridgeDepth, device: torch.device, *, stage_on_cpu: bool = False) -> BridgeDepth:
    """The portable float16 preparation plus the fused window attention and the fp16 DINOv2.

    Raises:
        KernelUnavailableError: The `fastbb` kernels cannot be loaded for `device`.
        BackendUnavailableError: A window geometry does not fit the kernel on `device` (the reason says which).
        RuntimeError: No WindowAttention was replaced (the module tree changed).
    """
    fastbb.load(device)
    prepare(model, half=True)
    candidates = max(getattr(m, "num_proposals", 1) for m in model.modules())

    def make(c: WindowAttention) -> KernelsWindowAttention:
        a = KernelsWindowAttention(c)
        if reason := fastbb.win_rpe_attn_reason(device, ws=a.ws, n=candidates):
            msg = f"BridgeDepth kernels: {reason}"
            raise BackendUnavailableError(msg)
        return a

    if not _swap(model, make):
        msg = "BridgeDepth kernels: no WindowAttention found"
        raise RuntimeError(msg)
    if not stage_on_cpu:
        model.to(device)
    fast = use_fast_vit(model.mono.pretrained)
    # fp16 weights: autocast cast every conv / linear weight and `LN16` every LayerNorm weight to fp16 on each call
    # (about 1100 copies); the casts are the same values, so the disparity is bit-identical. `get_v` stays fp32:
    # `get_rpe` also multiplies by its weight outside the conv, in fp32 by promotion.
    keep = {id(m.get_v) for m in model.modules() if isinstance(getattr(m, "get_v", None), nn.Conv2d)}
    for m in model.modules():
        if isinstance(m, LN16):
            m.weight.data, m.bias.data = m.weight.data.half(), m.bias.data.half()
        elif isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)) and id(m) not in keep:
            m.half()
    fuse_nmrf_encoder(model.stereo_encoder)
    if stage_on_cpu:
        model.to(device)
        # The helper is not a registered submodule. Move its preserved float32 table explicitly.
        fast.pos_fp32 = fast.pos_fp32.to(device)
    if sys.platform == "linux" and device.type == "cuda":
        out_conv = model.mono.depth_head.scratch.refinenet1.out_conv
        override(out_conv, _out_conv_forward)
    return model


@torch.no_grad()
def use_metal_attention(model: nn.Module, lib: "MetalLibrary") -> None:
    """Swap every WindowAttention of the (float16-prepared) portable network for the Metal kernel; in place.

    Raises:
        RuntimeError: No WindowAttention was replaced (the module tree changed).
    """
    from fastforks.kernels.stereo_metal.modules import win_rpe_attn  # noqa: PLC0415 - Apple MPS only

    attn = partial(win_rpe_attn, lib)
    if not _swap(model, lambda c: KernelsWindowAttention(c, attn)):
        msg = "BridgeDepth kernels: no WindowAttention found"
        raise RuntimeError(msg)


def forward_kernels(model: BridgeDepth, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """The portable test-mode forward under float16 autocast; returns the disparity."""
    return forward(model, left, right, half=True)
