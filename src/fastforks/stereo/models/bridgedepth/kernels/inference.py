"""Kernels inference path for BridgeDepth (general: any input size, no per-size compilation / tuning).

The portable path (`net/inference.py`: float16 autocast, float16 LayerNorms) on our kernels:
  - every WindowAttention (MRF inference: 6 x 6 cells x 4 candidates = 144 tokens, refinement: 4 x 4 x 1) as one fused
    kernel (`fastbb` `win_rpe_attn`, from our NMRF / BridgeDepth forks' general path): logits with the
    relative-position embeddings on q / k / v, mask, online softmax and the output in one pass, fp32 math, reading
    the packed qkv projection and writing the [windows, T, C] input of the output projection (no permute copies)
  - the Depth Anything V2 ViT of the mono branch as the fp16 DINOv2 of `fastforks.shared.depth_anything_v2`
    (`use_fast_vit`: fp16-accumulate GEMMs, `fastbb` flash attention)
"""

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import nn

from fastforks._core.errors import BackendUnavailableError
from fastforks.kernels import fastbb
from fastforks.shared.depth_anything_v2 import use_fast_vit

from ..net.bridgedepth import BridgeDepth
from ..net.inference import forward, prepare
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


@torch.no_grad()
def prepare_kernels(model: BridgeDepth, device: torch.device) -> BridgeDepth:
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
    model.to(device)
    use_fast_vit(model.mono.pretrained)
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
