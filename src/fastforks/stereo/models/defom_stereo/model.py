"""DEFOM-Stereo runners: weights -> network and the portable / kernels forward passes."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.mlx_cast import to_float32
from fastforks.shared import mlx_wrap
from fastforks.shared.depth_anything_v2 import use_sdpa_attention
from fastforks.stereo._runner import Forward, Runner

from .net.corr import CorrBlock1D
from .net.defom_stereo import DEFOMStereo
from .net.utils import get_danv2_io_size

if TYPE_CHECKING:
    from fastforks.stereo._runner import Context


@dataclass(frozen=True)
class Params:
    """Architecture (card `[params]`) and load options (card `[options]`)."""

    encoder: str  # DINOv2 encoder of the Depth Anything V2 branch
    iters: int
    scale_iters: int


def _net(ctx: Context, *, mixed_precision: bool = False) -> DEFOMStereo:
    args = types.SimpleNamespace(
        dinov2_encoder=ctx.params.encoder,
        idepth_scale=0.5,
        hidden_dims=[128] * 3,
        corr_levels=2,
        corr_radius=4,
        scale_list=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        scale_corr_radius=2,
        n_downsample=2,
        context_norm="batch",
        n_gru_layers=3,
        mixed_precision=mixed_precision,
        shared_backbone=False,
    )
    net = DEFOMStereo(args)
    net.load_state_dict(ctx.state_dict())
    return net.eval().to(ctx.device)


class Portable(Runner):
    """Plain PyTorch; bfloat16 on CUDA is upstream's autocast (`mixed_precision`), the correlation stays float32.

    float16 autocast overflows (NaN): the DAv2 activations exceed the float16 range.
    """

    dtypes = ("float32", "bfloat16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx, mixed_precision=ctx.dtype == "bfloat16")
        # fused attention: the eager q k^T matrix of the 7.5k-token Midd-H pair is 7 GB in float32 (softmax output)
        use_sdpa_attention(self.net.defomencoder.depth_anything.pretrained)

    def _run(self, net: DEFOMStereo, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        p = self.ctx.params
        if self.ctx.dtype != "bfloat16":
            return net(left, right, iters=p.iters, scale_iters=p.scale_iters, test_mode=True)[:, 0]
        # upstream's autocast regions take the default autocast dtype (float16): switch it for the call
        default = torch.get_autocast_dtype("cuda")
        torch.set_autocast_dtype("cuda", torch.bfloat16)
        try:
            return net(left, right, iters=p.iters, scale_iters=p.scale_iters, test_mode=True)[:, 0]
        finally:
            torch.set_autocast_dtype("cuda", default)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        return {"disparity": self._run(self.net, left, right)}

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        return Forward(self.net, self._run)


class Kernels(Runner):
    """Our kernels: fp16 NHWC update loop with fused GRU / lookup kernels; bf16 encoders (plain PyTorch)."""

    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network and the kernel state."""
        super().__init__(ctx)
        from .kernels.inference import forward_kernels, prepare_kernels  # noqa: PLC0415 - needs the kernel libraries

        self.net = _net(ctx)
        prepare_kernels(self.net, ctx.device)
        self._forward = forward_kernels

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        p = self.ctx.params
        return {"disparity": self._forward(self.net, left, right, iters=p.iters, scale_iters=p.scale_iters)[:, 0]}


class MetalKernels(Portable):
    """Apple MPS: float16 ViT / update loop, float32 DPT feature head, fused ConvGRUs and a one-kernel lookup."""

    module = Runner.module  # the Metal kernels are not exportable
    dtypes = ("float16",)

    def __init__(self, ctx: Context) -> None:
        """Build the network and switch it to the Metal kernels."""
        super().__init__(ctx)
        from fastforks._core import mps  # noqa: PLC0415 - MPS only
        from fastforks.kernels import stereo_metal  # noqa: PLC0415 - needs the kernel library

        from .kernels.metal import forward, prepare  # noqa: PLC0415

        mps.adapt(self.net)
        self.lib = stereo_metal.load(ctx.device)
        prepare(self.net, self.lib)
        self._forward = forward

    def _run(self, net: DEFOMStereo, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        p = self.ctx.params
        return self._forward(net, self.lib, left, right, p.iters, p.scale_iters)[:, 0]


class _Split(Portable):
    """`Portable` as three graphs for MLX: the DPT feature head (with fnet's conv on it) in float32 in a float16 run.

    As `MetalKernels`: refinenet1 / refinenet2 activations exceed the float16 range, the ViT, the depth head and the
    update loop do not. The disparity state and the convex upsampling are float32 islands.
    """

    dtypes = ("float32", "float16")

    def _encode(self, left: torch.Tensor, right: torch.Tensor) -> tuple:
        """The normalised pair, the depth head's inverse depth and the ViT's tokens (4 levels, then 4 class tokens)."""
        net = self.net
        ih, iw, oh, ow = get_danv2_io_size(*left.shape[2:], net.args.n_downsample)
        x = (torch.cat([left, right]) - net.mean) / net.std
        da = net.defomencoder.depth_anything
        feats = da.pretrained.get_intermediate_layers(
            F.interpolate(x, (ih, iw), mode="bilinear", align_corners=True),
            da.intermediate_layer_idx[da.encoder],
            return_class_token=True,
        )
        idepth = da.depth_head(feats, ih // 14, iw // 14, oh, ow)
        return (x, idepth, *(t for t, _ in feats), *(c for _, c in feats))

    def _dfeat(self, ph: int, pw: int, oh: int, ow: int, *tokens: torch.Tensor) -> tuple:
        """The DPT feature head (the cnet inputs) and fnet's `convd` on its per-view features."""
        net = self.net
        feats = list(zip(tokens[:4], tokens[4:], strict=True))
        d_feats, dfeat1, dfeat2 = net.defomencoder.depth_anything.depth_feat(feats, ph, pw, oh, ow)
        return (*d_feats, net.fnet.convd(torch.cat([dfeat1, dfeat2])))

    def _loop(self, x: torch.Tensor, idepth: torch.Tensor, *feats: torch.Tensor) -> torch.Tensor:
        """Context and feature encoders, correlation and update iterations (upstream `forward`, `test_mode=True`)."""
        net, a, p = self.net, self.net.args, self.ctx.params
        *d_feats, convd = feats
        bs = x.shape[0] // 2
        idepth = to_float32(idepth)
        ow = idepth.shape[-1]
        disp = idepth / (idepth.flatten(1).amax(1).view(-1, 1, 1, 1) + 1e-8) * net.defomencoder.idepth_scale * ow + 0.01
        cnet_list = net.cnet(x[:bs], d_feats)
        f = net.fnet
        y = f.layer3(f.layer2(f.layer1(f.relu1(f.norm1(f.conv1(x))))))
        fmap1, fmap2 = f.conv2(y + convd).split(bs)
        hidden = [torch.tanh(c[0]) for c in cnet_list]
        ctx = [
            list(conv(torch.relu(c[1])).split(conv.out_channels // 3, dim=1))
            for c, conv in zip(cnet_list, net.context_zqr_convs, strict=True)
        ]
        coords = net.initialize_coords(hidden[0])
        corr_fn = CorrBlock1D(
            fmap1,
            fmap2,
            coords,
            radius=a.corr_radius,
            num_levels=a.corr_levels,
            scale_list=a.scale_list,
            scale_corr_radius=a.scale_corr_radius,
        )
        limit = 2 ** (a.corr_levels - 1) * a.corr_radius
        up_mask = None
        for itr in range(p.iters):
            scaling = itr < p.scale_iters
            block = net.scale_update_block if scaling else net.update_block
            hidden, up_mask, out = block(
                hidden,
                ctx,
                corr_fn(disp, scaling=scaling),
                disp,
                iter32=a.n_gru_layers == 3,  # noqa: PLR2004 - upstream
                iter16=a.n_gru_layers >= 2,  # noqa: PLR2004 - upstream
                upsample=itr == p.iters - 1,
            )
            out = to_float32(out)
            disp = out * disp if scaling else disp + out.clamp(-limit, limit)
        return net.upsample_flow(disp, to_float32(up_mask))

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        ih, iw, oh, ow = get_danv2_io_size(*left.shape[2:], self.net.args.n_downsample)
        x, idepth, *tokens = self._encode(left, right)
        feats = self._dfeat(ih // 14, iw // 14, oh, ow, *tokens)
        return {"disparity": self._loop(x, idepth, *feats)[:, 0]}


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "kernels_mps": MetalKernels,
    "portable_mlx": mlx_wrap.port(
        _Split, attrs=(), methods=("_encode", "_dfeat", "_loop"), fp32_methods=("_dfeat",), release=True
    ),
}
