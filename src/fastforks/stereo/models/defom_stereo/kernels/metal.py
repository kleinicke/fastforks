"""DEFOM-Stereo on Apple MPS with Metal kernels (`fastforks.kernels.stereo_metal`).

`prepare(net, lib)`, then `forward(net, lib, left, right, iters, scale_iters)` (== upstream `test_mode=True`):
  - the Depth Anything V2 ViT and depth head in float16; the DPT feature head and fnet's conv on its output in float32
    (refinenet1 / refinenet2 activations exceed the float16 range)
  - fnet / cnet, the context convs and both update blocks in float16, the ConvGRUs fused (`fuse_grus`)
  - the correlation pyramid in one buffer, both lookups (2 levels x 9 taps, 8 scales x 5 taps) as one kernel
  - float32 disparity state, the upsampling mask and the convex upsampling only after the last iteration
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.patch import base_type, override
from fastforks.kernels.stereo_metal.modules import DispCorr, fuse_grus

from ..net.update import ConvGRU
from ..net.utils import get_danv2_io_size

if TYPE_CHECKING:
    from fastforks._core.metal import MetalLibrary

    from ..net.defom_stereo import DEFOMStereo


def _float32_forward(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The module (float32 weights) on `x` in float32; the output in float16."""
    return base_type(self).forward(self, x.float()).half()


def prepare(net: DEFOMStereo, lib: MetalLibrary) -> None:
    """Switch `net` (a float32 portable DEFOM-Stereo) to the dtypes above and the fused ConvGRUs; in place."""
    da = net.defomencoder.depth_anything
    for m in (da.pretrained, da.depth_head, net.fnet, net.cnet, net.context_zqr_convs, net.update_block):
        m.half()
    net.scale_update_block.half()
    override(net.fnet.convd.float(), _float32_forward)
    fuse_grus(net, ConvGRU, lib)


def forward(
    net: DEFOMStereo, lib: MetalLibrary, left: torch.Tensor, right: torch.Tensor, iters: int, scale_iters: int
) -> torch.Tensor:
    """Disparity `[B, 1, H, W]` of the pair (0..255, a multiple of 32)."""
    a = net.args
    i1, i2 = (left - net.mean) / net.std, (right - net.mean) / net.std
    ih, iw, oh, ow = get_danv2_io_size(*i1.shape[2:], a.n_downsample)
    enc = net.defomencoder
    da = enc.depth_anything
    x = F.interpolate(torch.cat([i1, i2]), (ih, iw), mode="bilinear", align_corners=True).half()
    ph, pw = ih // 14, iw // 14
    feats = da.pretrained.get_intermediate_layers(x, da.intermediate_layer_idx[da.encoder], return_class_token=True)
    idepth = da.depth_head(feats, ph, pw, oh, ow).float()
    d_feats, dfeat1, dfeat2 = da.depth_feat([(t.float(), c.float()) for t, c in feats], ph, pw, oh, ow)
    disp = idepth / (idepth.flatten(1).amax(1).view(-1, 1, 1, 1) + 1e-8) * enc.idepth_scale * ow + 0.01

    i1, i2 = i1.half(), i2.half()
    cnet_list = net.cnet(i1, [d.half() for d in d_feats])
    fmap1, fmap2 = net.fnet([i1, i2], [dfeat1, dfeat2])
    hidden = [torch.tanh(c[0]) for c in cnet_list]
    ctx = [
        [t.contiguous() for t in conv(torch.relu(c[1])).split(conv.out_channels // 3, dim=1)]
        for c, conv in zip(cnet_list, net.context_zqr_convs, strict=True)
    ]
    corr = DispCorr(lib, fmap1, fmap2, a.corr_levels, a.corr_radius, a.scale_list, a.scale_corr_radius)
    limit = 2 ** (a.corr_levels - 1) * a.corr_radius
    block = net.update_block
    for itr in range(iters):
        scaling = itr < scale_iters
        block = net.scale_update_block if scaling else net.update_block
        hidden = block(
            hidden,
            ctx,
            corr.lookup(disp, scaling=scaling),
            disp.half(),
            iter32=a.n_gru_layers == 3,
            iter16=a.n_gru_layers >= 2,
            update=False,
        )
        out = block.disp_head(hidden[0]).float()
        disp = F.relu6(torch.exp(0.25 * out)) * disp if scaling else disp + out.clamp(-limit, limit)
    return net.upsample_flow(disp, 0.25 * block.mask(hidden[0]).float())
