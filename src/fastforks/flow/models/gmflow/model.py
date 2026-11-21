"""GMFlow runners: weights -> upstream `UniMatch` network and the portable forward pass.

The forward pass is ours and follows upstream `UniMatch.forward` for `task="flow"` in eval mode with the settings of
upstream's `scripts/gmflow_evaluate.sh` (per scale: window splits of the Transformer, matching radius, propagation
radius; -1 = global). Differences that do not change the maths: the projection for the refinement is computed once
(upstream recomputes the same tensor in every refinement), and every attention (Transformer windows, global matching,
flow propagation) is one fused `scaled_dot_product_attention`, so the `[H*W, H*W]` score matrix (several GB at 1080p)
is never written. On CUDA in fp16 the CNN, the Transformer, the upsampling head and the refinement block run
under autocast; positions, matching, propagation, warping, correlation lookup and the flow stay float32.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import check_fits
from fastforks._core.mlx_cast import to_float32, to_run_dtype
from fastforks.flow._runner import Forward, Runner
from fastforks.kernels import flow_upsample
from fastforks.shared import mlx_wrap

from .net.attention import sdpa
from .net.geometry import coords_grid, flow_warp
from .net.matching import local_correlation_softmax, local_correlation_with_flow
from .net.transformer import TransformerLayer
from .net.unimatch import UniMatch
from .net.utils import feature_add_position, upsample_flow_with_mask

if TYPE_CHECKING:
    from fastforks.flow._runner import Context

SCALES = {  # card `scales` -> per scale (attn_splits, corr_radius, prop_radius), upstream scripts/gmflow_evaluate.sh
    1: ((2, -1, -1),),
    2: ((2, -1, -1), (8, 4, 1)),
}
REFINE_RADIUS = 4  # fixed: upstream's correlation radius of the refinement (the block's weights are made for 81 values)


@dataclass(frozen=True)
class Params:
    """Card `[params]`."""

    scales: int  # 1: matching at 1/8; 2: at 1/8, then 1/4 (upstream num_scales; upsample_factor 8 / 4)
    refine: int  # local regression refinements after the last scale (upstream num_reg_refine; 0 = no refinement block)


def attend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """`softmax(query @ key^T / sqrt(C)) @ value` for `[B, N, C]`, `[B, M, C]`, `[B, M, V]` (no `[N, M]` matrix)."""
    return sdpa(query, key, value)


def _headed(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    channels = v.shape[-1]
    v = F.pad(v, (0, -channels % 8))
    q, k, v = (x.contiguous().unsqueeze(1) for x in (q, k, v))
    bias = None if mask is None else to_run_dtype(mask, q.dtype).unsqueeze(1)  # .to: a no-op in the MLX twin
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias).squeeze(1)[..., :channels]


def forward(  # noqa: PLR0913 - network, image pair and inference options
    net: UniMatch, image1: torch.Tensor, image2: torch.Tensor, params: Params, *, fp16: bool, fused: bool = False
) -> torch.Tensor:
    """Flow `[B, 2, H, W]` of an ImageNet-normalised pair (H, W multiples of the card's padding)."""
    attention = _headed if fused else attend
    upsample = flow_upsample.convex if fused else upsample_flow_with_mask
    amp = torch.autocast(image1.device.type, torch.float16) if fp16 else contextlib.nullcontext()
    with amp:
        features1, features2 = net.extract_feature(image1, image2)  # low to high resolution
    flow = raw1 = raw2 = feature1 = image1.new_zeros(())  # set per scale
    for index, (splits, corr_radius, prop_radius) in enumerate(SCALES[params.scales]):
        raw1, raw2 = features1[index].float(), features2[index].float()
        feature1, feature2 = raw1, raw2
        if index > 0:
            flow = F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) * 2
            feature2 = flow_warp(feature2, flow)
        feature1, feature2 = feature_add_position(feature1, feature2, splits, net.feature_channels)
        with amp:
            feature1, feature2 = net.transformer(feature1, feature2, attn_type="swin", attn_num_splits=splits)
        feature1, feature2 = to_float32(feature1), to_float32(feature2)  # matching: float32 also in the MLX twin
        b, _, h, w = feature1.shape
        tokens1 = feature1.flatten(2).transpose(1, 2)  # [B, H*W, C]
        if corr_radius == -1:  # global matching: the expected position of the match
            grid = coords_grid(b, h, w, device=feature1.device)
            match = attention(tokens1, feature2.flatten(2).transpose(1, 2), grid.flatten(2).transpose(1, 2))
            predicted = match.transpose(1, 2).reshape(b, 2, h, w) - grid
        else:
            predicted = local_correlation_softmax(feature1, feature2, corr_radius)[0]
        flow = flow + predicted if index > 0 else predicted
        attn = net.feature_flow_attn
        if prop_radius > 0:
            flow = attn(feature1, flow, local_window_attn=True, local_window_radius=prop_radius)
        else:  # global propagation; upstream projects the key from the projected query (its own note)
            query = attn.q_proj(tokens1)
            flow = (
                attention(query, attn.k_proj(query), flow.flatten(2).transpose(1, 2))
                .transpose(1, 2)
                .reshape(b, 2, h, w)
            )
    if not params.refine:
        with amp:
            mask = net.upsampler(torch.cat((flow, feature1), dim=1))
        return upsample(flow, mask.float(), upsample_factor=net.upsample_factor)
    with amp:
        hidden, context = torch.chunk(net.refine_proj(feature1), chunks=2, dim=1)
        hidden, context = torch.tanh(hidden), torch.relu(context)
    mask = flow
    for step in range(params.refine):
        correlation = local_correlation_with_flow(
            to_float32(raw1), to_float32(raw2), flow=flow, local_radius=REFINE_RADIUS
        )
        with amp:
            _, up_mask, residual = net.refine(
                hidden,
                context,
                correlation,
                flow,
                upsample=step + 1 == params.refine,
            )
        if up_mask is not None:
            mask = up_mask
        flow = flow + residual.float()
    return upsample(flow, mask.float(), upsample_factor=net.upsample_factor)


class Portable(Runner):
    """Plain PyTorch; see the module docstring for what runs in fp16 on CUDA."""

    dtypes = ("float32", "float16")
    fused_upsample = False
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        p = ctx.params
        if p.scales not in SCALES or p.refine < 0:
            msg = f"{ctx.variant.ref}: card scales must be 1 or 2 and refine >= 0, not {p.scales!r}, {p.refine!r}"
            raise ValueError(msg)
        self.net = UniMatch(num_scales=p.scales, upsample_factor=8 // p.scales, reg_refine=p.refine > 0, task="flow")
        sd = ctx.state_dict(key="model")
        check_fits(self.net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
        self.net.load_state_dict(sd)
        self.net.eval().to(ctx.device)
        self.fp16 = ctx.dtype == "float16"

    def _flow(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        return forward(self.net, image1, image2, self.ctx.params, fp16=self.fp16, fused=self.fused_upsample)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flow of a padded batch."""
        return {"flow": self._flow(image1, image2)}

    def module(self) -> torch.nn.Module:
        """The network; returns the flow."""
        params, fp16 = self.ctx.params, self.fp16
        return Forward(self.net, lambda net, image1, image2: forward(net, image1, image2, params, fp16=fp16))


class Kernels(Portable):
    """Use explicit attention heads for fused SDPA and fuse the final convex upsampling."""

    module = Runner.module
    fused_upsample = True

    def __init__(self, ctx: Context) -> None:
        """Select the CUDA attention layout for this model instance."""
        super().__init__(ctx)
        for layer in self.net.transformer.modules():
            if isinstance(layer, TransformerLayer):
                layer.window_options = {"attention": _headed}


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(Portable, attrs=(), methods=("_flow",)),
}
