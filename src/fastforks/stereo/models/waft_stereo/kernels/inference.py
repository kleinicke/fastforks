"""Kernels inference path for WAFT-Stereo (general: any input size, no per-size compilation / tuning).

Port of our WAFT-Stereo fork's `cuda-kernels` branch (model/waft_fast.py @ 8dec350) on the shared kernel libraries.

`prepare_kernels(model, device)`, then `forward_kernels(model, left, right)` (images `[B, 3, H, W]`, 0..255, any size):
  - LoRA adapters merged into the base weights (W + B A alpha / r, exact up to rounding), DINOv2 as the fastbb fp16 ViT (flash kernel, fp16-accumulate GEMMs)
  - the iterative ViT's attention through the same fastbb flash kernel (`timm_blocks.flash`; SDPA's cuDNN kernel
    before)
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
from fastforks.kernels import dpt_fused, stereo_warp
from fastforks.shared import cnn_kernels
from fastforks.shared.cnn_kernels import ACT, CL, conv_ep
from fastforks.shared.depth_anything_v2 import use_fast_vit
from fastforks.shared.timm_blocks import Mlp
from fastforks.shared.timm_blocks.flash import use_flash_attention

from ..net._lora import PeftModel
from ..net.block import resconv
from ..net.utils import Padder

if TYPE_CHECKING:
    from ..net.dav2 import DAv2Encoder
    from ..net.dpt import ProjFeats, UpsampleFeats
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


def _finest(proj: ProjFeats, up: UpsampleFeats, features: list[torch.Tensor]) -> torch.Tensor:
    """Process the DPT scales coarse to fine, keeping only the preceding refined map."""
    prev = None
    blocks = zip(
        reversed(proj.projects),
        reversed(proj.resize_layers),
        reversed(up.scratch),
        reversed(up.refine),
        reversed(features),
        strict=True,
    )
    for project, resize, scratch, refine, tokens in blocks:
        x = scratch(resize(project(tokens)))
        if prev is not None:
            prev = F.interpolate(prev, scale_factor=2, mode="bilinear", align_corners=True)
            x.add_(refine.resConfUnit1(prev))
            del prev
        prev = refine.out_conv(refine.resConfUnit2(x))
        del x
    assert prev is not None
    return prev


def _stream_dpt(model: DAv2Encoder, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project and refine paired or joint DINOv2 features without retaining every scale."""
    b, n, _, h, w = imgs.shape
    h, w = h // 16, w // 16
    imgs = F.interpolate(imgs.flatten(0, 1), (h * 14, w * 14), mode="bilinear", align_corners=True)
    feats = model.encoder.get_intermediate_layers(imgs, model.idx, return_class_token=True)
    fmap_tokens = [x[0].reshape(b * n, h, w, -1).permute(0, 3, 1, 2) for x in feats]
    pair = _finest(model.fmap_proj, model.fmap_upsample, fmap_tokens).unflatten(0, (b, n))
    del fmap_tokens
    hidden_tokens = [x[0].reshape(b, n, h, w, -1).permute(0, 1, 4, 2, 3).reshape(b, -1, h, w) for x in feats]
    hidden = _finest(model.hidden_proj, model.hidden_upsample, hidden_tokens)
    return pair[:, 0], pair[:, 1], hidden


def prepare_kernels(model: WAFT, device: torch.device) -> None:
    """Switch `model` (fp32, eval, on `device`) to the kernels path in place.

    Raises:
        KernelUnavailableError: A kernel library is unusable on `device`.
    """
    model.eval()
    stereo_warp.load(device)
    _merge_lora(model)
    use_fast_vit(model.encoder.encoder, fused=dpt_fused.load(device))  # fp16 DINOv2 with the fastbb flash kernel
    override(model.encoder, _stream_dpt)
    for m in model.modules():
        if isinstance(m, resconv):
            if not (isinstance(m.conv[1], nn.SiLU) and is_base(m.conv[0], nn.InstanceNorm2d) and not m.conv[0].affine):
                raise ValueError(_NO_AFFINE)
            override(m, _resconv_forward)
        elif isinstance(m, Mlp) and isinstance(m.fc1, nn.Conv2d):
            override(m, _mlp_conv_forward)
    # The update's 2 E + D + 1 channel input (145) gets zero-weight channels up to a multiple of 8: 16-byte pixels, so
    # `forward_kernels` writes its parts straight into one channels-last buffer with vectorized copies.
    fc1 = model.delta_proj.fc1
    if pad := -fc1.in_channels % 8:
        wide = nn.Conv2d(fc1.in_channels + pad, fc1.out_channels, 1, device=fc1.weight.device)
        wide.weight.data = F.pad(fc1.weight.data, (0, 0, 0, 0, 0, pad))
        wide.bias.data = fc1.bias.data.clone()
        model.delta_proj.fc1 = wide
    cnn_kernels.prepare_generic(model, device)
    use_flash_attention(model)  # the iterative ViT (DINOv2's blocks are not timm `Attention`s)
    # `WAFT.normalize_image`'s constants, on the device once: its `new_tensor` is a host copy per call
    model._image_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    model._image_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


# Host copies (a CPU `new_tensor` / `arange`) cannot be captured in a CUDA graph: the kernels path makes the same
# tensors on the device instead of calling upstream's `normalize_image` / `disp_warp`
def _normalize(model: WAFT, img: torch.Tensor) -> torch.Tensor:
    """`WAFT.normalize_image` (`(img / 255 - mean) / std`) with the constants `prepare_kernels` put on the device."""
    mean = model._image_mean.to(img.dtype)
    std = model._image_std.to(img.dtype)
    return ((img / 255.0 - mean) / std).contiguous()


@torch.no_grad()
def forward_kernels(model: WAFT, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Disparity `[B, H, W]` of a pair in 0..255; the same math as `WAFT.forward` (test path)."""
    image1, image2 = _normalize(model, left), _normalize(model, right)
    padder = Padder(image1.shape, factor=model.factor)
    image1, image2 = padder.pad(image1), padder.pad(image2)
    fmap1, fmap2, hidden = model.encoder(torch.stack([image1, image2], dim=1).half())
    fmap1, fmap2, hidden = (x.contiguous(memory_format=CL) for x in (fmap1, fmap2, hidden))
    bins = torch.linspace(0, model.max_disp / 2, model.n_bins, device=fmap1.device).view(1, model.n_bins, 1, 1)
    prop = model.prop_decoder(model.prop_proj(torch.cat([fmap1, fmap2], dim=1)))
    prob = F.softmax(model.prop_bins_head(prop).float(), dim=1)  # (the 'init' output is a training target only)
    disp = torch.sum(prob * bins, dim=1, keepdim=True)
    del prob, prop
    # the update's input `[fmap1, warped fmap2, hidden, disp]`, written part by part (and the zero channels once)
    b, e, h, w = fmap1.shape
    shape = (b, model.delta_proj.fc1.in_channels, h, w)
    x = torch.empty(shape, dtype=fmap1.dtype, device=fmap1.device, memory_format=CL)
    x[:, :e] = fmap1
    d = hidden.shape[1]
    x[:, 2 * e + d + 1 :] = 0
    for _ in range(model.iters):
        stereo_warp.sample_into(fmap2, disp, x[:, e : 2 * e])
        x[:, 2 * e : 2 * e + d] = hidden
        x[:, 2 * e + d : 2 * e + d + 1] = disp
        hidden = model.delta_decoder(model.delta_proj(x))
        disp = disp + model.delta_disp_head(hidden).float()
    mask = 0.25 * model.delta_mask_head(hidden).float()
    return padder.unpad(model.convex_upsample(disp * 2, mask)).squeeze(1)
