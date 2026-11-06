"""Kernels inference path for NMRF (general: any input size /8, no per-size compilation / tuning).

The portable path (`net/inference.py`: float16 autocast, float16 LayerNorms, SDPA stripe attention) with every
WindowAttention (MRF inference: 6 x 6 cells x 4 candidates = 144 tokens, refinement: 4 x 4 x 1) as one fused kernel
(`fastbb` `win_rpe_attn`, from our NMRF fork's general path `nmrf/models/fast.py`): logits with the relative-position
embeddings on q / k / v, mask, online softmax and the output in one pass, fp32 math, straight on the packed qkv
projection (no cyclic shift, window partition or reverse copies) instead of 3 einsums + 2 matmuls + adds + softmax over
materialised [windows, heads, T, T] logits.
"""

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import nn

from fastforks._core.errors import BackendUnavailableError
from fastforks.kernels import fastbb

from ..net.inference import forward, prepare
from ..net.nmp import WindowAttention

if TYPE_CHECKING:
    from fastforks._core.metal import MetalLibrary


class KernelsWindowAttention(nn.Module):
    """WindowAttention.forward(qkv, attn_mask) on `fastbb.win_rpe_attn` (fp16 qkv, head_dim 32)."""

    table: torch.Tensor

    def __init__(self, a: WindowAttention, attn: Callable[..., torch.Tensor] = fastbb.win_rpe_attn) -> None:
        """`attn`: `fastbb.win_rpe_attn` or a kernel with its signature (the Metal one)."""
        super().__init__()
        self.attn = attn
        if a.window_size[0] != a.window_size[1] or a.dim != 32 * a.num_heads:
            msg = (
                f"NMRF kernels: square windows with head_dim 32 only, got {a.window_size}, dim {a.dim} / {a.num_heads}"
            )
            raise BackendUnavailableError(msg)
        self.ws, self.shift, self.heads, self.scale = a.window_size[0], a.shift_size, a.num_heads, a.scale
        self.register_buffer("table", a.relative_position_enc_table.detach().half().contiguous(), persistent=False)

    def forward(self, qkv: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        """qkv [B, H, W, N, 3C] -> [B, H, W, N, C]; attn_mask [nW, T, T] (window w: mask[w % nW]) or None."""
        _, ht, wd, n, _ = qkv.shape
        t = self.ws * self.ws * n
        mask = None if attn_mask is None else attn_mask.reshape(-1, t, t).float().contiguous()
        return self.attn(
            qkv.contiguous(), self.table, mask, heads=self.heads, ws=self.ws, n=n, grid=(ht, wd), shift=self.shift,
            scale=self.scale,
        )  # fmt: skip


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
def prepare_kernels(model: nn.Module, device: torch.device) -> nn.Module:
    """The portable float16 preparation plus the fused window attention.

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
            msg = f"NMRF kernels: {reason}"
            raise BackendUnavailableError(msg)
        return a

    if not _swap(model, make):
        msg = "NMRF kernels: no WindowAttention found"
        raise RuntimeError(msg)
    return model.to(device)


@torch.no_grad()
def use_metal_attention(model: nn.Module, lib: "MetalLibrary") -> None:
    """Swap every WindowAttention of the (float16-prepared) portable network for the Metal kernel; in place.

    Raises:
        RuntimeError: No WindowAttention was replaced (the module tree changed).
    """
    from fastforks.kernels.stereo_metal.modules import win_rpe_attn  # noqa: PLC0415 - Apple MPS only

    attn = partial(win_rpe_attn, lib)
    if not _swap(model, lambda c: KernelsWindowAttention(c, attn)):
        msg = "NMRF kernels: no WindowAttention found"
        raise RuntimeError(msg)


def forward_kernels(model: nn.Module, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """The portable test-mode forward under float16 autocast; returns the disparity."""
    return forward(model, left, right, half=True)
