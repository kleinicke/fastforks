"""Kernels inference path for NMRF (general: any input size /8, no per-size compilation / tuning).

The portable path (`net/inference.py`: float16 autocast, float16 LayerNorms, SDPA stripe attention) with every
WindowAttention (MRF inference: 6 x 6 cells x 4 candidates = 144 tokens, refinement: 4 x 4 x 1) as one fused kernel
(`fastbb` `win_rpe_attn`, from our NMRF fork's general path `nmrf/models/fast.py`): logits with the relative-position
embeddings on q / k / v, mask, online softmax and the output in one pass, fp32 math, straight on the packed qkv
projection (no cyclic shift, window partition or reverse copies) instead of 3 einsums + 2 matmuls + adds + softmax over
materialised [windows, heads, T, T] logits. The feature encoder's instance norms (ResNet backbone, `concatconv` / `gw`
heads) run on the `fastbb` norm kernels in fp16 channels-last, with the ReLUs and residual adds in the apply pass
(`fuse_instance_norms`; Middlebury-H 195 -> 175 ms).
"""

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.errors import BackendUnavailableError
from fastforks._core.patch import base_type, is_base, override
from fastforks.kernels import fastbb
from fastforks.shared.cnn_kernels import CL
from fastforks.shared.instance_norm import RELU, instance_norm, is_in_resblock, resblock_forward

from ..net.inference import LN16, forward, prepare
from ..net.nmp import Mlp, WindowAttention

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


def _backbone_forward(self: nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
    """`Backbone.forward` in fp16 channels-last from the scaled input on, the stem ReLU inside its norm pass."""
    x = (2 * (x / 255.0) - 1.0).half().contiguous(memory_format=CL)
    x = instance_norm(self.norm1, self.conv1(x), RELU, inplace=True)
    x = self.conv2(self.layer3(self.layer2(self.layer1(x))))
    return [x, F.avg_pool2d(x, kernel_size=2, stride=2)]  # high to low res


def _norm_relu(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return instance_norm(self, x, RELU, inplace=True)


def fuse_instance_norms(encoder: nn.Module) -> int:
    """Run an NMRF encoder's instance norms on `fastbb` (fp16 weights, in place); returns how many modules changed.

    The ResNet `Backbone` (if it is one: BridgeDepth's ConvNeXt variant has none) and the `concatconv` / `gw` heads,
    `Sequential(conv, InstanceNorm2d, ReLU, conv)`, whose ReLU moves into the norm's apply pass.

    Raises:
        RuntimeError: A block or head does not have the layout the fused forwards assume.
    """
    count = 0
    bb = encoder.backbone
    if base_type(bb).__name__ == "Backbone" and is_base(getattr(bb, "norm1", None), nn.InstanceNorm2d):
        blocks = [m for m in bb.modules() if base_type(m).__name__ == "ResidualBlock"]
        if not blocks or not all(is_in_resblock(b) for b in blocks):
            msg = "NMRF kernels: unexpected ResidualBlock layout in the backbone"
            raise RuntimeError(msg)
        for b in blocks:
            override(b, resblock_forward)
        override(bb, _backbone_forward)
        bb.to(memory_format=CL)
        count += len(blocks) + 1
    for seq in (encoder.concatconv, encoder.gw):
        if not (len(seq) == 4 and is_base(seq[1], nn.InstanceNorm2d) and isinstance(seq[2], nn.ReLU)):
            msg = "NMRF kernels: unexpected concatconv / gw layout"
            raise RuntimeError(msg)
        override(seq[1], _norm_relu)
        seq[2] = nn.Identity()
        seq.to(memory_format=CL)
        count += 1
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
    for layer in model.modules():
        if isinstance(layer, Mlp) and isinstance(layer.act, nn.GELU):
            layer.bounded_activation = True
    # fp16 weights: autocast would cast every fp32 conv / linear weight (and LN16 its affine) on each call. A stripe
    # attention's `get_v` weight also scales a tensor outside its conv (fp32 product there) and stays fp32.
    keep = {id(m.get_v) for m in model.modules() if isinstance(getattr(m, "get_v", None), nn.Conv2d)}
    for m in model.modules():
        if isinstance(m, LN16):
            m.weight.data, m.bias.data = m.weight.data.half(), m.bias.data.half()
        elif isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)) and id(m) not in keep:
            m.half()
    model.to(device)
    fuse_instance_norms(model)
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
        msg = "NMRF kernels: no WindowAttention found"
        raise RuntimeError(msg)


def forward_kernels(model: nn.Module, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """The portable test-mode forward under float16 autocast; returns the disparity."""
    return forward(model, left, right, half=True)
