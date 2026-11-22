"""Sample the last DeDoDe descriptor stage at the LoMa keypoints."""

from collections.abc import Callable

import torch
import torch.nn.functional as F

from fastforks._core.upload import device_tensor
from fastforks.shared.dedode import ConvRefiner, Descriptor


def _project(
    layer: ConvRefiner, features: torch.Tensor, context: torch.Tensor, y: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    feats = torch.cat((features, context), dim=1).half().contiguous(memory_format=torch.channels_last)
    x0 = layer.block1(feats)
    hidden = (layer.hidden_blocks(x0) + x0) / 1.4
    points = hidden[0, :, y, x].unsqueeze(0).contiguous(memory_format=torch.channels_last)
    return layer.out_conv(points)[0]


def _blend(  # noqa: PLR0913, PLR0917 - the last two scales and the interpolation stencil
    prior: torch.Tensor,
    delta: torch.Tensor,
    ix: torch.Tensor,
    iy: torch.Tensor,
    weights: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    h, w = size
    # Bilinear resize of the penultimate descriptor at those same pixel centres, with border replication.
    ch, cw = prior.shape[-2:]
    cx, cy = ((ix + 0.5) * (cw / w) - 0.5).clamp_min(0), ((iy + 0.5) * (ch / h) - 0.5).clamp_min(0)
    left, top = cx.long(), cy.long()
    right, bottom = (left + 1).clamp_max(cw - 1), (top + 1).clamp_max(ch - 1)
    fx, fy = cx - left, cy - top
    upper = (1 - fx) * prior[0, :, top, left] + fx * prior[0, :, top, right]
    lower = (1 - fx) * prior[0, :, bottom, left] + fx * prior[0, :, bottom, right]
    resized = ((1 - fy) * upper + fy * lower).to(prior.dtype)
    values = delta + resized
    return (values.float() * weights).sum(dim=1).mT


def sample(
    network: Descriptor, image: torch.Tensor, keypoints: torch.Tensor, *, blend: Callable = _blend
) -> torch.Tensor:
    """Evaluate the dense convolutions up to the final pointwise projection and sample its four neighbours."""
    features, sizes = network.encoder(image)
    features = list(features)
    scales = list(network.decoder.layers)
    prior = context = None
    for i, scale in enumerate(scales[:-1]):
        delta, context = network.decoder(features.pop(), context=context, scale=scale)
        prior = delta if prior is None else delta.add_(prior)
        size = sizes[-(i + 2)]
        context = F.interpolate(context, size=size, mode="bilinear", align_corners=False)
        if i < len(scales) - 2:
            prior = F.interpolate(prior, size=size, mode="bilinear", align_corners=False)
    layer = network.decoder.layers[scales[-1]]
    if prior is None or context is None or not isinstance(layer, ConvRefiner):
        msg = "LoMa sparse descriptors require a multi-scale DeDoDe decoder"
        raise ValueError(msg)
    h, w = features[0].shape[-2:]
    x, y = (
        ((keypoints.float() + 1) * device_tensor([w, h], device=keypoints.device, dtype=keypoints.dtype) - 1) / 2
    ).unbind(1)
    x0, y0 = x.floor(), y.floor()
    wx, wy = x - x0, y - y0
    ix = torch.stack((x0, x0 + 1, x0, x0 + 1)).long()
    iy = torch.stack((y0, y0, y0 + 1, y0 + 1)).long()
    weights = torch.stack(((1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy))
    weights = weights.masked_fill((ix < 0) | (ix >= w) | (iy < 0) | (iy >= h), 0)
    ix, iy = ix.clamp(0, w - 1), iy.clamp(0, h - 1)
    delta = _project(layer, features[0], context, iy, ix)
    return blend(prior, delta[: network.decoder.num_prototypes], ix, iy, weights, (h, w))
