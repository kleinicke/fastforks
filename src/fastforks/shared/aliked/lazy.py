"""ALIKED without its full-resolution 128-channel map (fastforks; the lightglue and flexglue kernels runners).

`extract_dense_map` upsamples the three coarse feature levels to the image size, concatenates them with the first
(a float32 `[B, 128, H, W]` map: 400 MB at 768 x 1024), feeds that to the score head's 1 x 1 convolution and
L2-normalizes it for SDDH, which reads only 3 x 3 patches around the keypoints and 16 bilinear samples per keypoint.
On an RTX 5060 Ti the concatenation, the convolution, the norm and the upsampling take about 7 ms per 768 x 1024
image. `install(aliked)` keeps the four levels as they are and:

- convolves each level with its slice of the score head's first (1 x 1, linear) convolution before upsampling it:
  8 channels upsampled instead of 32, the same map up to float32 rounding (the sum runs in another order);
- evaluates the descriptor map only at the pixels SDDH reads: the first level's values and the coarse levels
  bilinearly sampled at the pixel (`align_corners`, as their upsampling), normalized by their norm, then the patches
  and the samples' bilinear interpolation (zero padding) of those pixels as upstream's `grid_sample`.

`detection_maps` returns `(*levels, score_map, nms_scores)`; `forward` reads it. `graph_maps(aliked)` replays the
feature levels from a CUDA graph per input size. Scoring and NMS use ordinary allocator buffers.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

from fastforks._core.patch import override
from fastforks._core.sampling import grid_sample
from fastforks._core.upload import device_tensor
from fastforks.shared.cuda_graphs import GraphCache

from .net.aliked import InputPadder, simple_nms

if TYPE_CHECKING:
    from collections.abc import Callable

    from .net.aliked import ALIKED, SDDH

_EPS = 1e-12  # fixed: F.normalize's default eps (upstream's call)
_DIV_BY = 32  # fixed: extract_dense_map's padding


def install(aliked: ALIKED) -> None:
    """Make `aliked` compute its maps and descriptors from the feature levels (`detection_maps` returns those)."""
    head = aliked.desc_head
    if head.conv2D or head.mask or head.kernel_size <= 1:
        msg = "the level-wise maps cover ALIKED's released SDDH (kernel_size 3, no mask, no conv2D)"
        raise NotImplementedError(msg)
    override(aliked, detection_maps, "detection_maps")
    override(aliked, forward)


def graph_maps(aliked: ALIKED, graphs: int = 2) -> GraphCache:
    """Replay ALIKED's feature levels from a CUDA graph per input size; returns the cache.

    The graph's static outputs are handed to the keypoint selection and descriptors without a copy: the caller runs one
    extraction at a time, and none of the extraction's outputs is a view of them.
    """
    # The cache holds the bound method weakly, so it releases the pool when the model goes away.
    # Keep the full-resolution stem and projection out of the Linux native graph's private pool.
    coarse = sys.platform == "linux" and torch.cuda.get_allocator_backend() == "native"
    override(aliked, _coarse_levels if coarse else _feature_levels, "feature_levels")
    forward = cast("Callable[[torch.Tensor], tuple[torch.Tensor, ...]]", aliked.feature_levels)
    cache = GraphCache(forward, graphs=graphs)
    vars(aliked)["maps_graphs"] = cache
    vars(aliked)["coarse_maps_graphs"] = coarse
    return cache


def _static(out: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """A graph's static outputs as they are: read before the next replay (`graph_maps`)."""
    return out


def _stem(self: ALIKED, image: torch.Tensor) -> torch.Tensor:
    return self.block1(InputPadder(image.shape[-2], image.shape[-1], _DIV_BY).pad(image))


def _coarse_blocks(self: ALIKED, pooled: torch.Tensor) -> tuple[torch.Tensor, ...]:
    x2 = self.block2(pooled)
    x3 = self.block3(self.pool4(x2))
    x4 = self.block4(self.pool4(x3))
    return x2, x3, x4


def _coarse_projections(self: ALIKED, blocks: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(self.gate(head(x)) for head, x in zip((self.conv2, self.conv3, self.conv4), blocks, strict=True))


def _feature_levels(self: ALIKED, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """The four feature levels of the padded image, entirely eager or captured together."""
    with torch.autocast(image.device.type, dtype=torch.float16, enabled=self.fp16):
        x1 = _stem(self, image)
        blocks = _coarse_blocks(self, self.pool2(x1))
        return self.gate(self.conv1(x1)), *_coarse_projections(self, blocks)


def _coarse_levels(self: ALIKED, pooled: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """The three lower-resolution levels used by the Linux native graph."""
    with torch.autocast(pooled.device.type, dtype=torch.float16, enabled=self.fp16):
        return _coarse_projections(self, _coarse_blocks(self, pooled))


def _graph_feature_levels(self: ALIKED, image: torch.Tensor, cache: GraphCache) -> tuple[torch.Tensor, ...]:
    key = (image.device, image.dtype, tuple(image.shape))
    if not vars(self).get("coarse_maps_graphs", False):
        levels = cache.run(key, image, post=_static)
        return _feature_levels(self, image) if levels is None else levels

    with torch.autocast(image.device.type, dtype=torch.float16, enabled=self.fp16):
        x1 = _stem(self, image)
        pooled = self.pool2(x1)
    levels = cache.run(key, pooled, post=_static)
    if levels is None:
        levels = _coarse_levels(self, pooled)
    del pooled
    with torch.autocast(image.device.type, dtype=torch.float16, enabled=self.fp16):
        return self.gate(self.conv1(x1)), *levels


def detection_maps(self: ALIKED, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """The feature levels, score map and NMS, with the score workspace outside the graph's pool."""
    cache: GraphCache | None = vars(self).get("maps_graphs")
    levels = _feature_levels(self, image) if cache is None else _graph_feature_levels(self, image, cache)
    padder = InputPadder(image.shape[-2], image.shape[-1], _DIV_BY)
    with torch.autocast(image.device.type, enabled=False):  # upstream's float32 score head
        first, rest = cast("torch.nn.Conv2d", self.score_head[0]), self.score_head[1:]
        weights = first.weight.split([x.shape[1] for x in levels], dim=1)
        ups = (None, self.upsample2, self.upsample8, self.upsample32)
        s = None
        for x, weight, up in zip(levels, weights, ups, strict=True):
            y = F.conv2d(x.float(), weight)
            y = y if up is None else up(y)
            s = y if s is None else s + y
        score_map = padder.unpad(torch.sigmoid(rest(s)))
    return *levels, score_map, simple_nms(score_map, self.dkd.radius)


def forward(self: ALIKED, data: dict) -> dict:
    """`ALIKED.forward` on the level-wise `detection_maps`."""
    image = data["image"]
    maps = cast("tuple[torch.Tensor, ...]", self.detection_maps(image))  # this module's (or its graph's) outputs
    *levels, score_map, nms_scores = maps
    keypoints, kptscores, _ = self.dkd(score_map, image_size=data.get("image_size"), nms_scores=nms_scores)
    _, _, h, w = image.shape
    wh = device_tensor([w - 1, h - 1], device=image.device)
    out = {"keypoints": wh * (torch.stack(keypoints) + 1) / 2.0 + 0.0, "keypoint_scores": torch.stack(kptscores)}
    if self.describe:
        pad = InputPadder(h, w, _DIV_BY)._pad  # noqa: SLF001 - [left, right, top, bottom]
        out["descriptors"] = torch.stack(describe(self.desc_head, levels, (pad[2], pad[0]), (h, w), keypoints))
    return out


def describe(
    head: SDDH,
    levels: list[torch.Tensor],
    offset: tuple[int, int],
    size: tuple[int, int],
    keypoints: list[torch.Tensor],
) -> list[torch.Tensor]:
    """`SDDH.forward`'s descriptors on the normalized map of `levels` (its origin `offset` in them, its `size`)."""
    h, w = size
    wh = device_tensor([[w - 1, h - 1]], device=levels[0].device)
    max_offset = max(h, w) / 4.0
    ps, n_pos = head.kernel_size, head.n_pos
    steps = torch.arange(ps, device=levels[0].device)
    descriptors = []
    for ib, kptsi in enumerate(keypoints):
        image = (levels[0][ib], [x[ib : ib + 1].float() for x in levels[1:]], offset)  # the coarse levels cast once
        kptsi_wh = (kptsi / 2 + 0.5) * wh
        n_kpts = len(kptsi)
        # get_patches' pixels: patch[n, :, a, b] = map[:, cy + a, cx + b], the corner clamped into the map
        corner = (kptsi_wh.long() - ps / 2 + 1).long()
        cx = corner[:, 0].clamp(min=0, max=w - 1 - ps)
        cy = corner[:, 1].clamp(min=0, max=h - 1 - ps)
        py = (cy[:, None, None] + steps[None, :, None]).expand(-1, ps, ps)
        px = (cx[:, None, None] + steps[None, None, :]).expand(-1, ps, ps)
        patch = _pixels(image, py.reshape(-1), px.reshape(-1))  # [C, N * ps * ps]
        patch = patch.reshape(-1, n_kpts, ps, ps).permute(1, 0, 2, 3)  # [N_kpts, C, ps, ps]
        off = head.offset_conv(patch).clamp(-max_offset, max_offset)  # [N_kpts, 2*n_pos, 1, 1]
        off = off[:, :, 0, 0].view(n_kpts, 2, n_pos).permute(0, 2, 1)  # [N_kpts, n_pos, 2]
        pos = kptsi_wh.unsqueeze(1) + off
        pos = 2.0 * pos / wh[None] - 1
        features = _bilinear(image, size, pos.reshape(-1, 2))  # [C, N_kpts * n_pos]
        features = features.reshape(-1, n_kpts, n_pos, 1).permute(1, 0, 2, 3)  # [N_kpts, C, n_pos, 1]
        features = torch.selu_(head.sf_conv(features)).squeeze(-1)  # [N_kpts, C, n_pos]
        descs = torch.einsum("ncp,pcd->nd", features, head.agg_weights)  # [N_kpts, C]
        descriptors.append(F.normalize(descs, p=2.0, dim=1))
    return descriptors


_Image = tuple[torch.Tensor, list[torch.Tensor], tuple[int, int]]  # first level, float32 coarse levels, map origin


def _pixels(image: _Image, ys: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """The normalized descriptor map of one image at the integer pixels `(ys, xs)`: `[C, P]` float32."""
    fine, coarse, offset = image
    hp, wp = fine.shape[-2:]
    rows, cols = ys + offset[0], xs + offset[1]
    grid = torch.stack((2.0 * cols / (wp - 1) - 1, 2.0 * rows / (hp - 1) - 1), dim=-1)[None, None]  # [1, 1, P, 2]
    parts = [fine[:, rows, cols].float()]
    parts += [  # the upsampling's (align_corners) value at the pixel
        grid_sample(x, grid, mode="bilinear", align_corners=True)[0, :, 0] for x in coarse
    ]
    x = torch.cat(parts)
    return x / x.norm(2, 0, keepdim=True).clamp_min(_EPS)


def _bilinear(image: _Image, size: tuple[int, int], grid: torch.Tensor) -> torch.Tensor:
    """`grid_sample` of the normalized map (align_corners, zero padding) at `[P, 2]` points: `[C, P]` float32."""
    h, w = size
    gx, gy = grid.unbind(-1)
    ix, iy = (gx + 1) / 2 * (w - 1), (gy + 1) / 2 * (h - 1)
    x0, y0 = ix.floor(), iy.floor()
    x1, y1 = x0 + 1, y0 + 1
    # grid_sampler_2d's corners (nw, ne, sw, se) and weights, stacked: [4, P]
    cx, cy = torch.stack((x0, x1, x0, x1)), torch.stack((y0, y0, y1, y1))
    wx0, wx1, wy0, wy1 = x1 - ix, ix - x0, y1 - iy, iy - y0
    weights = torch.stack((wx0, wx1, wx0, wx1)) * torch.stack((wy0, wy0, wy1, wy1))
    inside = (cx >= 0) & (cx <= w - 1) & (cy >= 0) & (cy <= h - 1)
    weights = torch.where(inside, weights, 0).reshape(-1)
    ys = cy.clamp(0, h - 1).long().reshape(-1)
    xs = cx.clamp(0, w - 1).long().reshape(-1)
    values = _pixels(image, ys, xs) * weights  # [C, 4 P]
    return values.unflatten(1, (4, -1)).sum(1)
