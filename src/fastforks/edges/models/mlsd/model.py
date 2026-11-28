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

from fastforks._core.runner import host
from fastforks._core.upload import device_tensor
from fastforks.edges._runner import Forward, Runner, to_detect, to_input
from fastforks.shared import mlx_wrap
from fastforks.shared.cnn_kernels import prepare_generic
from fastforks.shared.cuda_graphs import SmallShapeGraphs

from .net import MobileV2_MLSD_Large

if TYPE_CHECKING:
    from fastforks.edges._runner import Context

TOPK = 200  # fixed: upstream pred_lines: deccode_output_score_and_ptss(outputs, 200, 3)


@dataclass(frozen=True)
class Params:
    """Card `[params]` + `[options]`.

    Attributes:
        short_side: The short side of the network input (controlnet_aux detect resolution; rounded to a multiple of 64).
        thr_v: Center score threshold (kept if above).
        thr_d: Segment length threshold in half-resolution pixels (kept if above).
        cuda_graphs: Kernels replays the network and the line decoding from a CUDA graph for small repeated shapes.
    """

    short_side: int
    thr_v: float
    thr_d: float
    cuda_graphs: bool = True


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

    def shared_outside() -> torch.Tensor:
        """Both endpoints lie beyond the same border (the outcodes have a common bit)."""
        return (
            ((x1 < 0) & (x2 < 0))
            | ((x1 > right) & (x2 > right))
            | ((y1 < 0) & (y2 < 0))
            | ((y1 > bottom) & (y2 > bottom))
        )

    c1, c2 = code(x1, y1), code(x2, y2)
    todo = ~shared_outside() & ((c1 != 0) | (c2 != 0))
    m = todo & ((y1 < 0) | (y1 > bottom))  # y clipping of point 1, then point 2
    a = torch.where(c1 < 8, 0, bottom)  # noqa: PLR2004 - OpenCV's outcode bits
    x1, y1 = torch.where(m, move(a, y1, x1, x2 - x1, torch.where(m, y2 - y1, 1)), x1), torch.where(m, a, y1)
    c1 = torch.where(m, (x1 < 0).long() + (x1 > right).long() * 2, c1)
    m = todo & ((y2 < 0) | (y2 > bottom))
    a = torch.where(c2 < 8, 0, bottom)  # noqa: PLR2004 - OpenCV's outcode bits
    x2, y2 = torch.where(m, move(a, y2, x2, x2 - x1, torch.where(m, y2 - y1, 1)), x2), torch.where(m, a, y2)
    c2 = torch.where(m, (x2 < 0).long() + (x2 > right).long() * 2, c2)
    todo = todo & ~shared_outside() & ((c1 != 0) | (c2 != 0))
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
    return x1, y1, x2, y2, (c1 == 0) & (c2 == 0)


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
    out.index_fill_(0, flat.flatten(), 1)  # a Python value: an index_put of 1 copies it from the host
    return out[:-1].view(b, h, w)


class Portable(Runner):
    """Upstream's network in plain PyTorch, float32 or float16 (channels-last on CUDA)."""

    Params = Params
    dtypes = ("float32", "float16")
    compiled_methods = ("_net",)  # the line decoding (data-dependent segment counts) stays eager

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.dtype = getattr(torch, ctx.dtype)
        net = MobileV2_MLSD_Large()
        net.load_state_dict(ctx.state_dict())
        self.cl = ctx.device.type == "cuda"
        fmt = torch.channels_last if self.cl else torch.contiguous_format
        self.net = net.eval().to(ctx.device, self.dtype).to(memory_format=fmt)

    def _net(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _edges_and_segments(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Compute the line map and fixed-size segment candidates before filtering."""
        x = to_detect(image, self.ctx.params.short_side)
        hd, wd = x.shape[-2:]
        edges, seg, keep = self._detect_lines(x)
        return to_input(edges.to(image.device), image), seg, keep, hd, wd

    def _detect_lines(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Line map `[B, hd, wd]`, segment candidates `[B, K, 4]` and `keep` `[B, K]` of `x` at the detect size."""
        p = self.ctx.params
        hd, wd = x.shape[-2:]
        x = torch.cat((x, torch.ones_like(x[:, :1])), 1) / 127.5 - 1
        if self.cl:
            x = x.contiguous(memory_format=torch.channels_last)
        maps = self._net(x.to(self.dtype))
        if maps.device.type == "mps":  # the decoding and drawing are float64 / int64 maths, which MPS lacks
            maps = maps.cpu()
        seg, keep = decode(maps, p.thr_v, p.thr_d)
        return draw(seg.trunc().long(), keep, hd, wd), seg, keep

    @host
    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Line map `[B, H, W]` and segments (`B` x `[N, 4]`, input pixels) of `[B, 3, H, W]` RGB in `[0, 1]`."""
        edges, seg, keep, hd, wd = self._edges_and_segments(image)
        scale = device_tensor([image.shape[-1] / wd, image.shape[-2] / hd], device=seg.device, dtype=seg.dtype).repeat(
            2
        )
        lines = [(s[k] * scale).float().to(image.device) for s, k in zip(seg, keep, strict=True)]
        return {"edges": edges, "lines": lines}

    def module(self) -> torch.nn.Module:
        """The network with its pre- and post-processing: `[B, 3, H, W]` in `[0, 1]` -> edges `[B, H, W]`."""
        return Forward(self.net, lambda _net, image: self._edges_and_segments(image)[0])


# kernels replays a CUDA graph up to this many images x H x W at the detect size: the fixed-size line decoding is
# ~400 tiny kernels, launch bound (RTX 5060 Ti, Windows, fp16, ms per image: 512 px 11.3 -> 3.0, 1024 x 768
# 11.7 -> 3.9; +160 MB reserved)
GRAPH_PIXELS = 4 * 512 * 512  # fixed: a speed threshold, not the architecture


class Kernels(Portable):
    """Run the convolution blocks with fused CUDA kernels."""

    module = Runner.module

    def __init__(self, ctx: Context) -> None:
        """Prepare the convolution blocks for inference."""
        super().__init__(ctx)
        prepare_generic(self.net, ctx.device)
        forward = super()._detect_lines
        self.graphs = SmallShapeGraphs(forward, pixels=GRAPH_PIXELS, enabled=ctx.params.cuda_graphs)

    def release_graphs(self) -> None:
        """Release the CUDA graphs and their pool."""
        self.graphs.release()

    def _detect_lines(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`Portable._detect_lines`, from a CUDA graph for a small repeated shape (the decoding is fixed-size)."""
        return self.graphs(x)


class _MlxPortable(Portable):
    """Apple MLX: the network written in MLX operations (`mlx_net`); the decoding and drawing as `Portable`."""

    compiled_methods = ()  # the network is MLX
    module = Runner.module  # the MLX network is not exportable

    def __init__(self, ctx: Context) -> None:
        """The MLX network from the conversion cache (the first load folds the norms into the convolutions)."""
        Runner.__init__(self, ctx)
        from fastforks.shared import mlx_port  # noqa: PLC0415 - imports mlx

        from . import mlx_net  # noqa: PLC0415 - imports mlx

        def real() -> MobileV2_MLSD_Large:
            net = MobileV2_MLSD_Large()
            net.load_state_dict(ctx.state_dict())
            return net.eval()

        self.dtype = torch.float32  # the input as the network takes it
        self.cl = False
        self.mlx = mlx_port.recorded(
            ctx,
            mlx_net.LAYOUT,
            lambda net: mlx_net.network(net, ctx.dtype),
            real=real,
            skeleton=MobileV2_MLSD_Large,
            sources=mlx_net.SOURCES,
        )
        from fastforks._core.mlx import request_cache  # noqa: PLC0415 - imports mlx

        request_cache(self, 512)  # the buffer cache the network's temporaries reuse (held while this runner lives)

    def _net(self, x: torch.Tensor) -> torch.Tensor:
        """The maps `[B, 9, h, w]` of `x` `[B, 4, H, W]`."""
        from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - imports mlx

        return to_torch(self.mlx(to_mx(x.permute(0, 2, 3, 1).contiguous()))).permute(0, 3, 1, 2)


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": _MlxPortable,
}
# FASTFORKS_MLX_GRAPH=1 (diagnostics, tools/mlx_core_ab.py): the exported-graph port the hand-written network replaces
GRAPH_RUNNERS = {"portable_mlx": mlx_wrap.port(Portable)}
