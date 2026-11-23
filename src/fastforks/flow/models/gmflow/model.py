"""GMFlow runners: weights -> upstream `UniMatch` network and the portable forward pass.

The forward pass is ours and follows upstream `UniMatch.forward` for `task="flow"` in eval mode with the settings of
upstream's `scripts/gmflow_evaluate.sh` (per scale: window splits of the Transformer, matching radius, propagation
radius; -1 = global). Differences that do not change the maths: the projection for the refinement is computed once
(upstream recomputes the same tensor in every refinement), and every attention (Transformer windows, global matching,
flow propagation) is one fused `scaled_dot_product_attention`, so the `[H*W, H*W]` score matrix (several GB at 1080p)
is never written (the kernels runner computes global matching and propagation as GEMMs over chunks of query rows,
`_chunked`). On CUDA in fp16 the CNN, the Transformer, the upsampling head and the refinement block run
under autocast; positions, matching, propagation, warping, correlation lookup and the flow stay float32 (the kernels
runner's refinement correlation reads the CNN's fp16 features and accumulates in float32, `_FlowCorrelation`).
"""

from __future__ import annotations

import contextlib
import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import check_fits
from fastforks._core.mlx_cast import to_float32, to_run_dtype
from fastforks._core.patch import override
from fastforks.flow._runner import Forward, Runner
from fastforks.kernels import flow_upsample
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import SmallShapeGraphs

from .net.attention import sdpa
from .net.geometry import coords_grid, flow_warp
from .net.matching import local_correlation_softmax, local_correlation_with_flow
from .net.transformer import FeatureTransformer, TransformerLayer
from .net.unimatch import UniMatch
from .net.utils import feature_add_position, generate_shift_window_attn_mask, upsample_flow_with_mask

if TYPE_CHECKING:
    from fastforks.flow._runner import Context

    from .net.reg_refine import BasicUpdateBlock

SCALES = {  # card `scales` -> per scale (attn_splits, corr_radius, prop_radius), upstream scripts/gmflow_evaluate.sh
    1: ((2, -1, -1),),
    2: ((2, -1, -1), (8, 4, 1)),
}
REFINE_RADIUS = 4  # fixed: upstream's correlation radius of the refinement (the block's weights are made for 81 values)


@dataclass(frozen=True)
class Params:
    """Card `[params]` + load options."""

    scales: int  # 1: matching at 1/8; 2: at 1/8, then 1/4 (upstream num_scales; upsample_factor 8 / 4)
    refine: int  # local regression refinements after the last scale (upstream num_reg_refine; 0 = no refinement block)
    cuda_graphs: bool  # option: kernels replays CUDA graphs for small repeated shapes


def attend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """`softmax(query @ key^T / sqrt(C)) @ value` for `[B, N, C]`, `[B, M, C]`, `[B, M, V]` (no `[N, M]` matrix)."""
    return sdpa(query, key, value)


def _headed(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    channels = v.shape[-1]
    v = F.pad(v, (0, -channels % 8))
    q, k, v = (x.contiguous().unsqueeze(1) for x in (q, k, v))
    bias = None if mask is None else to_run_dtype(mask, q.dtype).unsqueeze(1)  # .to: a no-op in the MLX twin
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias).squeeze(1)[..., :channels]


_MATCH_SCORES = 128 << 20  # fixed: bytes of one chunk of float32 matching scores (61 vs 80 ms fused at 1080p)


def _chunked(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """`attend` as float32 GEMMs and a softmax over chunks of query rows (CUDA global matching and propagation).

    The fused float32 SDPA kernel is slow for the 2-channel values of the match and the flow; a chunk of the scores
    (`_MATCH_SCORES`, not the whole `[N, M]` matrix) costs less and sums the softmax exactly (max error 5e-5 vs
    3.5e-4 px against float64 at 1080p).
    """
    b, n, m = query.shape[0], query.shape[1], key.shape[1]
    step = max(256, _MATCH_SCORES // (4 * b * m) // 256 * 256)
    key_t = (key * key.shape[-1] ** -0.5).transpose(1, 2)
    out = value.new_empty((b, n, value.shape[-1]))
    scores = None
    for start in range(0, n, step):
        rows = query[:, start : start + step]
        if scores is None or scores.shape[1] != rows.shape[1]:
            scores = torch.bmm(rows, key_t)
        else:
            torch.bmm(rows, key_t, out=scores)
        torch.bmm(scores.softmax(-1), value, out=out[:, start : start + step])
    return out


_CORR_GATHER = 24 << 20  # fixed: bytes of gathered feature rows per chunk, L2-resident (25.9 vs 28.2 ms at 256 MiB)


class _FlowCorrelation:
    """`local_correlation_with_flow` (zeros padding) as dot products at integer positions (CUDA refinement, NHWC).

    The window offsets are integers and the flow is shared by a pixel's window, so all its samples have the same
    bilinear fraction: each correlation is the bilinear blend of the dot products at the `(2R + 2)^2` integer positions
    around the flow target. Those are gathered rows of the second feature map in its own dtype (the fp16 features of
    the autocast CNN, exact) with float32 accumulation; upstream's `grid_sample` writes `(2R + 1)^2` float32 samples per
    pixel instead. 1080p, one refinement: 26 vs 36 ms; max difference 2e-4 on correlations up to 5.
    """

    def __init__(self, feature0: torch.Tensor, feature1: torch.Tensor, radius: int) -> None:
        b, c, h, w = feature0.shape
        self.shape, self.radius = (b, c, h, w), radius
        rows = feature1.flatten(2).transpose(1, 2)
        self.table = torch.cat([rows, rows.new_zeros((b, 1, c))], dim=1).flatten(0, 1)  # [B * (HW + 1), C]; 0 row
        self.query = feature0.flatten(2).transpose(1, 2).reshape(b * h * w, 1, c)
        self.offsets = torch.arange(-radius, radius + 2, device=feature0.device)
        self.base = torch.arange(b, device=feature0.device)[:, None, None] * (h * w + 1)  # each image's first row
        self.y = torch.arange(h, device=feature0.device, dtype=torch.float32)[:, None]
        self.x = torch.arange(w, device=feature0.device, dtype=torch.float32)[None]

    def __call__(self, flow: torch.Tensor) -> torch.Tensor:
        """`[B, (2R+1)^2, H, W]` float32 correlation at the flow `[B, 2, H, W]`."""
        b, c, h, w = self.shape
        k = 2 * self.radius + 2
        x, y = (self.x + flow[:, 0]).reshape(b, -1), (self.y + flow[:, 1]).reshape(b, -1)
        x0, y0 = x.floor(), y.floor()
        wx, wy = (x - x0)[..., None, None], (y - y0)[..., None, None]
        gx = x0.long()[..., None, None] + self.offsets  # [B, N, 1, k]
        gy = y0.long()[..., None, None] + self.offsets[:, None]  # [B, N, k, 1]
        inside = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
        index = (torch.where(inside, gy * w + gx, h * w) + self.base).reshape(b * h * w, k * k)
        dots = torch.empty((b * h * w, 1, k * k), device=flow.device, dtype=torch.float32)
        step = max(1, _CORR_GATHER // (k * k * c * self.table.element_size()))
        for start in range(0, b * h * w, step):
            rows = self.table.index_select(0, index[start : start + step].reshape(-1)).view(-1, k * k, c)
            torch.bmm(
                self.query[start : start + step],
                rows.transpose(1, 2),
                out_dtype=torch.float32,
                out=dots[start : start + step],
            )
        dots = dots.view(b, h * w, k, k)
        corr = (1 - wy) * ((1 - wx) * dots[:, :, :-1, :-1] + wx * dots[:, :, :-1, 1:]) + wy * (
            (1 - wx) * dots[:, :, 1:, :-1] + wx * dots[:, :, 1:, 1:]
        )
        return (corr.reshape(b, h, w, -1) / c**0.5).permute(0, 3, 1, 2)  # channels-last, as the refinement block runs


def forward(  # noqa: PLR0913 - network, image pair and inference options
    net: UniMatch, image1: torch.Tensor, image2: torch.Tensor, params: Params, *, fp16: bool, fused: bool = False
) -> torch.Tensor:
    """Flow `[B, 2, H, W]` of an ImageNet-normalised pair (H, W multiples of the card's padding)."""
    attention = _chunked if fused else attend
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
    if fused:  # the CNN's own features (fp16 under autocast), not the float32 copies; a channels-last refinement block
        last = len(SCALES[params.scales]) - 1
        correlate = _FlowCorrelation(features1[last], features2[last], REFINE_RADIUS)
        hidden, context = (x.contiguous(memory_format=torch.channels_last) for x in (hidden, context))
    else:
        correlate = lambda f: local_correlation_with_flow(  # noqa: E731
            to_float32(raw1), to_float32(raw2), flow=f, local_radius=REFINE_RADIUS
        )
    for step in range(params.refine):
        correlation = correlate(flow)
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


_GRAPH_PIXELS = 262_144  # fixed: batch x padded H x W; launch-bound up to 384x512 (1.14x), 544x960 gains under 2 %
_GRAPHS = 2  # fixed: graph cache policy


@functools.lru_cache(maxsize=8)
def _window_orders(h: int, w: int, splits: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Token gathers between image order and the two window orders of a `splits` x `splits` Swin layer.

    Returns (image -> unshifted, unshifted -> shifted, shifted -> unshifted, shifted -> image, unshifted -> image).
    The shifted order is upstream's: rolled by minus half a window, then split into windows.
    """
    wh, ww = h // splits, w // splits
    image = torch.arange(h * w, device=device).view(h, w)
    split = lambda t: t.view(splits, wh, splits, ww).permute(0, 2, 1, 3).reshape(-1)  # noqa: E731
    plain = split(image)
    shifted = split(torch.roll(image, shifts=(-(wh // 2), -(ww // 2)), dims=(0, 1)))
    to_image = lambda order: torch.empty_like(order).scatter_(0, order, torch.arange(order.numel(), device=device))  # noqa: E731
    plain_back, shifted_back = to_image(plain), to_image(shifted)
    return plain, plain_back[shifted], shifted_back[plain], shifted_back, plain_back


def _window_layer(
    layer: Any,  # noqa: ANN401 - a vendored TransformerLayer
    source: torch.Tensor,
    inputs: tuple[torch.Tensor, torch.Tensor],
    windows: int,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Upstream `TransformerLayer.forward` (swin, one head) on tokens already in its window order.

    `inputs` are the source and target already in the projections' autocast dtype: autocast would cast the float32
    tokens again for each of the three projections.
    """
    b, n, c = source.shape
    q, k, v = (
        proj(x).view(b * windows, n // windows, c)
        for proj, x in ((layer.q_proj, inputs[0]), (layer.k_proj, inputs[1]), (layer.v_proj, inputs[1]))
    )
    if mask is None:
        message = _headed(q, k, v)
    else:  # one image's windows at a time against the [K*K, N, N] mask, as upstream for large masks
        message = torch.cat(
            [
                _headed(qa, ka, va, mask)
                for qa, ka, va in zip(q.split(windows), k.split(windows), v.split(windows), strict=True)
            ]
        )
    del q, k, v
    message = layer.norm1(layer.merge(message.view(b, n, c)))
    if layer.no_ffn:
        return source + message
    return layer._add_feed_forward(source, message)  # noqa: SLF001 - upstream's CUDA float32 inference branch


def _window_transformer(
    self: FeatureTransformer,
    feature0: torch.Tensor,
    feature1: torch.Tensor,
    attn_type: str = "swin",
    attn_num_splits: int | None = None,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream `FeatureTransformer.forward` for 2-D Swin attention with the tokens kept in window order.

    Upstream rolls q, k and v of every shifted layer, splits all three into windows and merges and rolls the output
    back (eight copies per shifted layer). Every other op of a layer is per token, so here the tokens of both images
    are gathered into the next layer's window order once per layer; the attention sees the same windows in the same
    order, with upstream's mask.
    """
    if attn_type != "swin" or attn_num_splits is None or attn_num_splits <= 1 or torch.is_grad_enabled():
        msg = "the window-order GMFlow transformer is inference-only 2-D swin with splits > 1"
        raise RuntimeError(msg)
    del kwargs
    b, c, h, w = feature0.shape
    windows = attn_num_splits**2
    to_plain, plain_to_shifted, shifted_to_plain, shifted_back, plain_back = _window_orders(
        h, w, attn_num_splits, feature0.device
    )
    autocast = torch.is_autocast_enabled(feature0.device.type)
    dtype = torch.get_autocast_dtype(feature0.device.type) if autocast else feature0.dtype
    mask = generate_shift_window_attn_mask(
        input_resolution=(h, w),
        window_size_h=h // attn_num_splits,
        window_size_w=w // attn_num_splits,
        shift_size_h=h // attn_num_splits // 2,
        shift_size_w=w // attn_num_splits // 2,
        device=feature0.device,
        dtype=dtype,
    )
    tokens = torch.cat((feature0, feature1)).flatten(-2).transpose(1, 2).index_select(1, to_plain)  # [2B, HW, C]
    shifted = False
    for i, layer in enumerate(self.layers):
        if (i % 2 == 1) != shifted:
            tokens = tokens.index_select(1, shifted_to_plain if shifted else plain_to_shifted)
            shifted = not shifted
        layer_mask = mask if shifted else None
        low = tokens.to(dtype)
        other = torch.cat(low.chunk(2)[::-1])  # the cross-attention target: the other image before this block
        tokens = _window_layer(layer.self_attn, tokens, (low, low), windows, layer_mask)
        tokens = _window_layer(layer.cross_attn_ffn, tokens, (tokens.to(dtype), other), windows, layer_mask)
    tokens = tokens.index_select(1, shifted_back if shifted else plain_back)
    feature0, feature1 = tokens.view(2 * b, h, w, c).permute(0, 3, 1, 2).contiguous().chunk(2)
    return feature0, feature1


def _refine_forward(self: BasicUpdateBlock, net, inp, corr, flow, *, upsample=True):  # noqa: ANN001, ANN202, PLR0913 - upstream's names
    """Upstream `BasicUpdateBlock.forward`, channels-last, with the GRU input cast to the autocast dtype once.

    The motion features end in the float32 flow, so upstream's `[context, motion]` concatenation is float32 and each of
    the GRU's six convolutions (and its concatenations with the fp16 hidden state) re-casts all 256 channels. Every
    consumer casts it to the autocast dtype, so casting it once gives the same values.

    The block's convolutions run channels-last (Kernels moves its weights; the hidden state, context and correlation
    come channels-last): cuDNN's fp16 kernels are NHWC, so NCHW tensors were converted on every convolution.
    """
    flow = flow.contiguous(memory_format=torch.channels_last)
    inp = torch.cat([inp, self.encoder(flow, corr)], dim=1)
    if torch.is_autocast_enabled(inp.device.type):
        inp = inp.to(torch.get_autocast_dtype(inp.device.type))
    net = self.gru(net, inp)
    delta_flow = self.flow_head(net)
    mask = self.mask(net) if self.mask is not None and upsample else None
    return net, mask, delta_flow


class Kernels(Portable):
    """Use explicit attention heads for fused SDPA, chunked GEMMs for global matching and fuse the convex upsampling.

    A repeated input shape up to `_GRAPH_PIXELS` replays a CUDA graph (`fastforks.shared.cuda_graphs`): the eager
    pass is launch-bound there.
    """

    module = Runner.module
    fused_upsample = True

    def __init__(self, ctx: Context) -> None:
        """Select the CUDA attention layout for this model instance."""
        super().__init__(ctx)
        for layer in self.net.transformer.modules():
            if isinstance(layer, TransformerLayer):
                layer.window_options = {"attention": _headed}
        override(self.net.transformer, _window_transformer)
        if ctx.params.refine:
            self.net.refine.to(memory_format=torch.channels_last)
            override(self.net.refine, _refine_forward)
        self._graphs = SmallShapeGraphs(
            super()._flow, pixels=_GRAPH_PIXELS, graphs=_GRAPHS, enabled=ctx.params.cuda_graphs
        )

    def _flow(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        return self._graphs(image1, image2)

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        self._graphs.release()


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(Portable, attrs=(), methods=("_flow",)),
}
