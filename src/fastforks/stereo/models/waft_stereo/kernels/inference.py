"""Kernels inference path for WAFT-Stereo (general: any input size, no per-size compilation / tuning).

Port of our WAFT-Stereo fork's `cuda-kernels` branch (model/waft_fast.py @ 8dec350) on the shared kernel libraries.

`prepare_kernels(model, device)`, then `forward_kernels(model, left, right)` (images `[B, 3, H, W]`, 0..255, any size):
  - LoRA adapters merged into the base weights (W + B A alpha / r, exact up to rounding), DINOv2 as the fastbb fp16 ViT (flash kernel, fp16-accumulate GEMMs)
  - pure fp16 weights and NHWC activations; every conv through `cnn_kernels.conv_ep` (fp16-accumulate cuDNN graphs, or CUTLASS
    with the epilogue fused for >= 128 output channels)
  - the pre-activation `resconv` blocks as InstanceNorm + SiLU in one epilogue each and conv + bias + residual in
    another, the 1x1-conv `Mlp`s as conv + bias + GELU
  - the test path only: no training-time outputs (bin upsampling of the proposal, per-iteration info / disparity
    upsampling), convex upsampling after the last iteration; disparity state, bin regression, warps and the upsampling
    in fp32
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.patch import is_base, override
from fastforks.kernels import dpt_fused
from fastforks.shared import cnn_kernels
from fastforks.shared.cnn_kernels import ACT, CL, conv_ep
from fastforks.shared.depth_anything_v2 import use_fast_vit

from ..net._lora import PeftModel
from ..net._timm import Mlp
from ..net.block import resconv
from ..net.utils import Padder, disp_warp

if TYPE_CHECKING:
    from ..net.waft import WAFT

_NO_AFFINE = "the kernels path implements InstanceNorm2d without affine parameters and SiLU only"


def _in_act(x: torch.Tensor, eps: float, act: int, *, inplace: bool) -> torch.Tensor:
    """InstanceNorm2d (no affine) + activation as one epilogue per sample; a channels-last copy unless `inplace`."""
    y = x if inplace else x.clone(memory_format=CL)
    for i in range(y.shape[0]):  # the epilogue's scale / shift are per channel, so one launch per sample
        cnn_kernels.instance_norm(y[i : i + 1], eps, act)
    return y


def _resconv_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """IN -> SiLU -> conv -> IN -> SiLU -> conv, + skip."""
    n0, _, c0, n1, _, c1 = self.conv
    skip = x if isinstance(self.skip_conv, nn.Identity) else conv_ep(self.skip_conv, x)
    y = conv_ep(c0, _in_act(x, n0.eps, ACT["silu"], inplace=False))
    y = _in_act(y, n1.eps, ACT["silu"], inplace=True)
    return conv_ep(c1, y, r=skip.contiguous(memory_format=CL))


def _mlp_conv_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """`Mlp(use_conv=True)` at eval (no norm / dropout): fc1 + bias + GELU, fc2 + bias."""
    return conv_ep(self.fc2, conv_ep(self.fc1, x, act=ACT["gelu"]))


def _merge_lora(model: WAFT) -> int:
    n = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, PeftModel):
                setattr(parent, name, child.merge_and_unload())
                n += 1
    return n


def prepare_kernels(model: WAFT, device: torch.device) -> None:
    """Switch `model` (fp32, eval, on `device`) to the kernels path in place.

    Raises:
        KernelUnavailableError: A kernel library is unusable on `device`.
    """
    model.eval()
    _merge_lora(model)
    use_fast_vit(
        model.encoder.encoder, fused=dpt_fused.load(device)
    )  # fp16 DINOv2 with the fastbb flash kernel; the iterative modules use SDPA
    for m in model.modules():
        if isinstance(m, resconv):
            if not (isinstance(m.conv[1], nn.SiLU) and is_base(m.conv[0], nn.InstanceNorm2d) and not m.conv[0].affine):
                raise ValueError(_NO_AFFINE)
            override(m, _resconv_forward)
        elif isinstance(m, Mlp) and isinstance(m.fc1, nn.Conv2d):
            override(m, _mlp_conv_forward)
    cnn_kernels.prepare_generic(model, device)


@torch.no_grad()
def forward_kernels(model: WAFT, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Disparity `[B, H, W]` of a pair in 0..255; the same math as `WAFT.forward` (test path)."""
    image1, image2 = model.normalize_image(left), model.normalize_image(right)
    padder = Padder(image1.shape, factor=model.factor)
    image1, image2 = padder.pad(image1), padder.pad(image2)
    fmap1, fmap2, hidden = model.encoder(torch.stack([image1, image2], dim=1).half())
    fmap1, fmap2, hidden = (x.contiguous(memory_format=CL) for x in (fmap1, fmap2, hidden))
    bins = torch.linspace(0, model.max_disp / 2, model.n_bins, device=fmap1.device).view(1, model.n_bins, 1, 1)
    prop = model.prop_decoder(model.prop_proj(torch.cat([fmap1, fmap2], dim=1)))
    prob = F.softmax(model.prop_bins_head(prop).float(), dim=1)  # (the 'init' output is a training target only)
    disp = torch.sum(prob * bins, dim=1, keepdim=True)
    del prob, prop
    fmap2_f32 = fmap2.float()
    for _ in range(model.iters):
        warped = disp_warp(fmap2_f32, disp, padding_mode="zeros").half()
        x = torch.cat([fmap1, warped, hidden, disp.half()], dim=1).contiguous(memory_format=CL)
        hidden = model.delta_decoder(model.delta_proj(x))
        disp = disp + model.delta_disp_head(hidden).float()
    mask = 0.25 * model.delta_mask_head(hidden).float()
    return padder.unpad(model.convex_upsample(disp * 2, mask)).squeeze(1)
