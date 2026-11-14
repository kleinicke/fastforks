"""M-LSD runners: weights -> network, the portable forward pass, segment decoding and the line rasterizer.

Inference as controlnet_aux's `MLSDdetector` (ComfyUI's "M-LSD Lines"): the image resized to the detect size
(`edges._runner.to_detect`), RGB levels plus a ones channel scaled to `x / 127.5 - 1`, the network's half-resolution
center and displacement maps decoded as upstream's `pred_lines` (sigmoid, 3x3 max-pool NMS, top 200 centers, kept if
the score is above `thr_v` and the displacement length above `thr_d`), the segments truncated to integers and drawn
1 px wide (`cv2.line`, `LINE_8`, exact) at the detect size, then the map resized to the input size (bilinear). The
segments are also returned (`lines`, input pixels, before the truncation).

`portable`: upstream's modules in plain PyTorch, fp32 or fp16 (channels-last on CUDA); decoding and drawing in
float32 / int64 on the device, without host syncs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks.edges._runner import Forward, Runner, to_detect, to_input

from .net import MobileV2_MLSD_Large

if TYPE_CHECKING:
    from fastforks.edges._runner import Context

TOPK = 200  # fixed: upstream pred_lines: deccode_output_score_and_ptss(outputs, 200, 3)


@dataclass(frozen=True)
class Params:
    """Card `[params]` + `[options]`.

    Attributes:
        resolution: Detect resolution: the short side of the network input (rounded to a multiple of 64).
        thr_v: Center score threshold (kept if above).
        thr_d: Segment length threshold in half-resolution pixels (kept if above).
    """

    resolution: int
    thr_v: float
    thr_d: float


def decode(maps: torch.Tensor, thr_v: float, thr_d: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream `pred_lines`' decoding, batched: `[B, 9, h, w]` -> segments `[B, K, 4]` float64 and `keep` `[B, K]`.

    Segments are `2 * (center + displacement)` (x0, y0, x1, y1) in detect pixels, computed in float64 as numpy.
    """
    b, _, h, w = maps.shape
    maps = maps.float()
    heat = torch.sigmoid(maps[:, 0])
    heat = heat * (F.max_pool2d(heat, 3, stride=1, padding=1) == heat)
    scores, idx = heat.flatten(1).topk(min(TOPK, h * w), dim=1)
    disp = maps[:, 1:5].flatten(2).gather(2, idx[:, None].expand(b, 4, -1)).transpose(1, 2)  # [B, K, 4]
    dist = (disp[..., :2] - disp[..., 2:]).square().sum(-1).sqrt()
    keep = (scores > thr_v) & (dist > thr_d)
    xy = torch.stack((idx % w, idx // w), -1).repeat(1, 1, 2)
    return 2 * (xy.double() + disp.double()), keep


def _clip(seg: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, ...]:
    """OpenCV `clipLine` of int64 segments `[N, 4]` (its double divisions truncated) -> x1, y1, x2, y2, inside."""
    x1, y1, x2, y2 = seg.unbind(1)
    right, bottom = w - 1, h - 1

    def code(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x < 0).long() + (x > right).long() * 2 + (y < 0).long() * 4 + (y > bottom).long() * 8

    def move(a: torch.Tensor, a0: torch.Tensor, b0: torch.Tensor, db: torch.Tensor, da: torch.Tensor) -> torch.Tensor:
        """OpenCV's `b0 + (int64)((double)(a - a0) * db / da)`."""
        return b0 + torch.trunc((a - a0).double() * db.double() / da.double()).long()

    c1, c2 = code(x1, y1), code(x2, y2)
    todo = ((c1 & c2) == 0) & ((c1 | c2) != 0)
    m = todo & ((c1 & 12) != 0)  # y clipping of point 1, then point 2
    a = torch.where(c1 < 8, 0, bottom)  # noqa: PLR2004 - OpenCV's outcode bits
    x1, y1 = torch.where(m, move(a, y1, x1, x2 - x1, torch.where(m, y2 - y1, 1)), x1), torch.where(m, a, y1)
    c1 = torch.where(m, (x1 < 0).long() + (x1 > right).long() * 2, c1)
    m = todo & ((c2 & 12) != 0)
    a = torch.where(c2 < 8, 0, bottom)  # noqa: PLR2004 - OpenCV's outcode bits
    x2, y2 = torch.where(m, move(a, y2, x2, x2 - x1, torch.where(m, y2 - y1, 1)), x2), torch.where(m, a, y2)
    c2 = torch.where(m, (x2 < 0).long() + (x2 > right).long() * 2, c2)
    todo = todo & ((c1 & c2) == 0) & ((c1 | c2) != 0)
    m = todo & (c1 != 0)  # x clipping
    a = torch.where(c1 == 1, 0, right)
    x1, y1, c1 = (
        torch.where(m, a, x1),
        torch.where(m, move(a, x1, y1, y2 - y1, torch.where(m, x2 - x1, 1)), y1),
        torch.where(m, 0, c1),
    )
    m = todo & (c2 != 0)
    a = torch.where(c2 == 1, 0, right)
    x2, y2, c2 = (
        torch.where(m, a, x2),
        torch.where(m, move(a, x2, y2, y2 - y1, torch.where(m, x2 - x1, 1)), y2),
        torch.where(m, 0, c2),
    )
    return x1, y1, x2, y2, (c1 | c2) == 0


def draw(seg: torch.Tensor, keep: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """`cv2.line(thickness=1, LINE_8)` of int64 segments `[B, K, 4]` (where `keep`) -> `[B, h, w]` float32 0 / 1.

    Bresenham in closed form: along the major axis (`big` steps), after `k` steps the minor axis has moved
    `max(0, ceil((2 small k - big) / (2 big)))`, OpenCV's error-term walk after its clipping and left-to-right swap.
    """
    b, k = keep.shape
    x1, y1, x2, y2, inside = _clip(seg.reshape(-1, 4), h, w)
    swap = x2 < x1
    x1, x2, y1, y2 = (
        torch.where(swap, x2, x1),
        torch.where(swap, x1, x2),
        torch.where(swap, y2, y1),
        torch.where(swap, y1, y2),
    )
    dx, dy = x2 - x1, y2 - y1
    sy = torch.where(dy < 0, -1, 1)[:, None]
    dy = dy.abs()
    ymaj = (dy > dx)[:, None]
    big, small = torch.maximum(dx, dy)[:, None], torch.minimum(dx, dy)[:, None]
    step = torch.arange(max(h, w), device=seg.device)[None]  # big <= max(h, w) - 1 after clipping
    num = 2 * small * step - big
    minor = torch.where(num > 0, -torch.div(-num, 2 * big.clamp(min=1), rounding_mode="floor"), 0)
    px = x1[:, None] + torch.where(ymaj, minor, step)
    py = y1[:, None] + sy * torch.where(ymaj, step, minor)
    valid = (step <= big) & (inside & keep.reshape(-1))[:, None]
    image = torch.arange(b, device=seg.device).repeat_interleave(k)[:, None].expand_as(px)
    flat = torch.where(valid, (image * h + py.clamp(0, h - 1)) * w + px.clamp(0, w - 1), b * h * w)
    out = torch.zeros(b * h * w + 1, device=seg.device)
    out[flat.flatten()] = 1
    return out[:-1].view(b, h, w)


class Portable(Runner):
    """Upstream's network in plain PyTorch, float32 or float16 (channels-last on CUDA)."""

    Params = Params
    dtypes = ("float32", "float16")
    compiled_methods = ("net",)

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.dtype = getattr(torch, ctx.dtype)
        net = MobileV2_MLSD_Large()
        net.load_state_dict(ctx.state_dict())
        self.cl = ctx.device.type == "cuda"
        fmt = torch.channels_last if self.cl else torch.contiguous_format
        self.net = net.eval().to(ctx.device, self.dtype).to(memory_format=fmt)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Line map `[B, H, W]` and segments (`B` x `[N, 4]`, input pixels) of `[B, 3, H, W]` RGB in `[0, 1]`."""
        p = self.ctx.params
        x = to_detect(image, p.resolution)
        hd, wd = x.shape[-2:]
        x = torch.cat((x, torch.ones_like(x[:, :1])), 1) / 127.5 - 1
        if self.cl:
            x = x.contiguous(memory_format=torch.channels_last)
        seg, keep = decode(self.net(x.to(self.dtype)), p.thr_v, p.thr_d)
        edges = draw(seg.trunc().long(), keep, hd, wd)
        scale = seg.new_tensor([image.shape[-1] / wd, image.shape[-2] / hd]).repeat(2)
        lines = [(s[k] * scale).float() for s, k in zip(seg, keep, strict=True)]
        return {"edges": to_input(edges, image), "lines": lines}

    def module(self) -> torch.nn.Module:
        """The network with its pre- and post-processing: `[B, 3, H, W]` in `[0, 1]` -> edges `[B, H, W]`."""
        return Forward(self.net, lambda _net, image: self(image)["edges"])


RUNNERS = {"portable": Portable}
