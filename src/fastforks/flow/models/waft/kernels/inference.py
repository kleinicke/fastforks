"""Kernels inference path for WAFT (flow; general: any input size, no per-size compilation / tuning).

Same recipe as WAFT-Stereo's: `prepare_kernels(net, device)`, then `forward_kernels(net, image1, image2, iters)`
(images `[B, 3, H, W]`, 0..255):
  - Depth Anything's DINOv2 as the fp16 `fastbb` ViT (flash kernel, fp16-accumulate GEMMs, `use_fast_vit`)
  - the refinement ViT's attention through the same `fastbb` flash kernel (q, k, v read as views of the packed
    qkv; SDPA's cuDNN kernel took 2.7 ms per block at 1080p)
  - pure fp16 weights and NHWC activations; every conv through `cnn_kernels.conv_ep` (fp16-accumulate cuDNN graphs, or
    CUTLASS with the epilogue fused for >= 128 output channels)
  - the two DPT heads' bilinear resizes through `dpt_fused`'s NHWC kernel (bit-identical, 4x PyTorch's NHWC one)
  - the pre-activation `resconv` blocks as GELU + (conv + bias + GELU) + (conv + bias + residual)
  - flow state, warp coordinates, grid sampling and the softmax upsampling in fp32; only the last iteration is
    upsampled (the earlier ones are training outputs)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.patch import override
from fastforks.kernels import dpt_fused
from fastforks.shared import cnn_kernels
from fastforks.shared.cnn_kernels import ACT, CL, conv_ep
from fastforks.shared.depth_anything_v2 import use_fast_vit, use_fused_bilinear
from fastforks.shared.timm_blocks.flash import use_flash_attention

from ..net.utils import Padder, bilinear_sampler, coords_grid
from ..net.waft_a1 import resconv

if TYPE_CHECKING:
    from ..net import ViTWarpV8


def _resconv_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """GELU -> conv -> GELU -> conv, + skip (conv or identity)."""
    _, c0, _, c1 = self.conv
    skip = x if isinstance(self.skip_conv, nn.Identity) else conv_ep(self.skip_conv, x)
    y = conv_ep(c0, F.gelu(x), act=ACT["gelu"])
    return conv_ep(c1, y, r=skip.contiguous(memory_format=CL))


def prepare_kernels(net: ViTWarpV8, device: torch.device) -> None:
    """Switch `net` (fp32, eval, on `device`) to the kernels path in place.

    Raises:
        KernelUnavailableError: A kernel library is unusable on `device`.
    """
    net.eval()
    fused = dpt_fused.load(device)
    use_fast_vit(net.da_feature.depth_anything.pretrained, fused=fused)
    use_fused_bilinear(net, fused)  # both DPT heads' channels-last resizes (PyTorch's NHWC bilinear is 4x slower)
    for m in net.modules():
        if isinstance(m, resconv):
            override(m, _resconv_forward)
    use_flash_attention(net.refine_net)
    cnn_kernels.prepare_generic(net, device)


def _upsample(net: ViTWarpV8, flow: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Convex 2x upsampling of the flow (the info channels are not needed at inference)."""
    n, _, h, w = flow.shape
    mask = torch.softmax(mask.float().view(n, 1, 9, 2, 2, h, w), dim=2)
    up = F.unfold(2 * flow, [3, 3], padding=1).view(n, 2, 9, 1, 1, h, w)
    up = torch.sum(mask * up, dim=2).permute(0, 1, 4, 2, 5, 3)
    return up.reshape(n, 2, 2 * h, 2 * w)


@torch.no_grad()
def forward_kernels(net: ViTWarpV8, image1: torch.Tensor, image2: torch.Tensor, iters: int) -> torch.Tensor:
    """Flow `[B, 2, H, W]` of a pair in 0..255; the same math as `ViTWarpV8.forward` (last iteration)."""
    image1, image2 = net.normalize_image(image1), net.normalize_image(image2)
    padder = Padder(image1.shape, factor=112)
    image1, image2 = padder.pad(image1), padder.pad(image2)
    n, _, h, w = image1.shape
    both = torch.cat([image1, image2], dim=0).half().contiguous(memory_format=CL)
    da = net.da_feature(both)["out"]
    feats = net.fnet(torch.cat([da, both], dim=1).contiguous(memory_format=CL))
    da_2x = F.interpolate(da, scale_factor=0.5, mode="bilinear", align_corners=True)
    fmap = net.fmap_conv(torch.cat([feats[0], da_2x], dim=1).contiguous(memory_format=CL))
    fmap1, fmap2 = fmap[:n], fmap[n:]
    hidden = net.hidden_conv(torch.cat([fmap1, fmap2], dim=1).contiguous(memory_format=CL))
    fmap2_f32 = fmap2.float()
    flow = torch.zeros(n, 2, h // 2, w // 2, device=image1.device)
    coords0 = coords_grid(n, h // 2, w // 2, device=image1.device)
    # The two 1 x 1 convs' channels-last inputs are written part by part (upstream: two cats, each copied again to
    # channels last). warp_in keeps fmap1 across iterations; refine_in is rebuilt each time so it is not held during
    # refine_net (kept, the peak grew 135 MiB at 1080p; as is, 31 MiB).
    c = fmap1.shape[1]
    warp_in = torch.empty((n, 3 * c + 2, h // 2, w // 2), dtype=fmap1.dtype, device=fmap1.device, memory_format=CL)
    warp_in[:, :c] = fmap1
    for _ in range(iters):
        warp_in[:, c : 2 * c] = bilinear_sampler(fmap2_f32, (coords0 + flow).permute(0, 2, 3, 1))
        warp_in[:, 2 * c : 3 * c] = hidden
        warp_in[:, 3 * c :] = flow
        out = net.refine_net(net.warp_linear(warp_in))["out"]
        refine_in = torch.empty(
            (n, out.shape[1] + c, *out.shape[2:]), dtype=out.dtype, device=out.device, memory_format=CL
        )
        refine_in[:, : out.shape[1]] = out
        refine_in[:, out.shape[1] :] = hidden
        del out
        hidden = net.refine_transform(refine_in)
        del refine_in
        flow = flow + net.flow_head(hidden)[:, :2].float()
    mask = 0.25 * net.upsample_weight(hidden).float()
    return padder.unpad(_upsample(net, flow, mask))
