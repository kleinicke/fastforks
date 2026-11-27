"""HRNet runners: person crops -> keypoint heatmaps -> keypoints in the image (mmpose `inference_topdown`).

Per person box (`x0, y0, x1, y1`):

- crop (`GetBBoxCenterScale` + `TopdownAffine`, `fastforks.shared.topdown`): the box centre, its size x 1.25 with the
  shorter side widened to 3 : 4, sampled bilinearly the way OpenCV `warpAffine` does and rounded to uint8; RGB
  normalised with the mmpose means / stds in 0..255
- decode (`MSRAHeatmap.decode`): the maximum of each heatmap (its value is the keypoint score), moved a quarter
  heatmap pixel towards the larger neighbour along x and along y (no shift in the outer two columns / rows), times
  the 4 crop pixels per heatmap pixel; mapped back with `p / (w, h) * size + centre - size / 2`
- `flip_test` (option, off by default as in our other pose families; the reference configs and their COCO AP turn
  it on): the heatmaps of the mirrored crop, mirrored back with left / right keypoints swapped and shifted one
  column to the right (`shift_heatmap`), averaged with the plain ones
- `scores`: the mean keypoint score (the reference gives none)

The reference marks keypoints whose score is <= 0 with -1 before mapping them back; here they keep their argmax: the
score says they were not found.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from fastforks._core.checkpoint import NUMPY_ARRAYS, Inert, latin1_encode, numeric_scalar
from fastforks._optimize import optimize
from fastforks.kernels import pose_warp
from fastforks.pose._runner import BUCKETS, Forward, Runner, TopDown, in_buckets
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import SmallShapeGraphs
from fastforks.shared.topdown import FLIP, MEAN, STD, crop, crop_boxes

from .net import STRIDE, Arch, HRNet, convert

if TYPE_CHECKING:
    from fastforks.pose._runner import Context

REFINE = 0.25  # fixed: reference shift towards the larger neighbour, in heatmap pixels (refine_keypoints)


@dataclass(frozen=True)
class Params(Arch):
    """Card `[params]` + variant `params` + `[options]`."""

    flip_test: bool  # average with the mirrored crop (option; the reference's COCO AP protocol)
    cuda_graphs: bool  # kernels: replay a CUDA graph per person bucket (option)


def decode(hm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(crop coordinates `[P, K, 2]` x, y in input pixels, scores `[P, K]`) of float32 heatmaps `[P, K, H, W]`."""
    h, w = hm.shape[-2:]
    flat = hm.flatten(2)
    score, idx = flat.max(-1, keepdim=True)
    x, y = idx % w, idx // w

    def step(d: int, inside: torch.Tensor) -> torch.Tensor:
        """Sign of the heatmap difference between the neighbours `d` entries after and before the maximum."""
        ahead = flat.gather(-1, (idx + d).clamp(max=h * w - 1))
        behind = flat.gather(-1, (idx - d).clamp(min=0))
        return torch.sign(ahead - behind) * inside

    dx = step(1, (x > 1) & (x < w - 1) & (y > 0))
    dy = step(w, (y > 1) & (y < h - 1) & (x > 0))
    return (torch.cat([x + dx * REFINE, y + dy * REFINE], -1)) * STRIDE, score[..., 0]


class Portable(TopDown):
    """Plain PyTorch: crops on the device, float16 network on CUDA, float32 decode."""

    dtypes = ("float32", "float16")
    Params = Params

    crop = staticmethod(crop)

    def __init__(self, ctx: Context) -> None:
        """Build the network with the checkpoint (BatchNorm folded) on the device."""
        super().__init__(ctx)
        a: Params = ctx.params
        net = HRNet(a)
        net.load_state_dict(convert(ctx.state_dict(key="state_dict", strip="", extra=extra())))
        self.net = net.eval().to(ctx.device, ctx.torch_dtype)
        if ctx.device.type == "cuda" and ctx.dtype == "float16":
            # channels-last weights: cuDNN's fp16 kernels skip the NCHW<->NHWC conversion around each conv
            # (w32, 15 people at 1080p 45.4 -> 37.2 ms, RTX 5060 Ti, Windows); float32 stays NCHW
            self.net.to(memory_format=torch.channels_last)
        self.mean = torch.tensor(MEAN, device=ctx.device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=ctx.device).view(1, 3, 1, 1)
        self.flip = torch.tensor(FLIP[ctx.variant.output_info["label_set"]], device=ctx.device)
        self.net_size = torch.tensor([a.width, a.height], device=ctx.device, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """float32 heatmaps of normalised float32 crops (the network pass the runners differ in)."""
        return self.net(x.to(self.ctx.torch_dtype)).float()

    def heatmaps(self, crops: torch.Tensor) -> torch.Tensor:
        """float32 heatmaps `[P, K, h / 4, w / 4]` of crops `[P, 3, h, w]` in 0..255."""
        x = (crops - self.mean) / self.std
        if not self.ctx.params.flip_test:
            return self.forward(x)
        p = x.shape[0]
        hm = self.forward(torch.cat([x, x.flip(-1)]))
        back = hm[p:].flip(-1).index_select(1, self.flip)
        back = torch.cat([back[..., :1], back[..., :-1]], -1)  # one column to the right, the first one kept
        return (hm[:p] + back) * 0.5

    def regions(self, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Crop centres and sizes of the person boxes."""
        return crop_boxes(boxes, self.ctx.params.height, self.ctx.params.width)

    def crops(self, image: torch.Tensor, centre: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
        """Crops in 0..255 at the network size (`crop`: the runner's crop operator)."""
        return self.crop(image, centre, size, self.ctx.params.height, self.ctx.params.width)

    def people(self, crops: torch.Tensor, centre: torch.Tensor, size: torch.Tensor) -> dict[str, torch.Tensor]:
        """Heatmap decoding, mapped back to the input."""
        coords, score = decode(self.heatmaps(crops))
        kp = coords * (size / self.net_size)[:, None] + (centre - size * 0.5)[:, None]
        return {"keypoints": kp, "keypoint_scores": score, "scores": score.mean(1)}

    def module(self) -> torch.nn.Module:
        """The network: crops `[B, 3, h, w]` RGB in [0, 1] at the network size -> heatmaps `[B, K, h / 4, w / 4]`."""

        def fn(_net: Any, crops: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - the network
            return self.heatmaps(crops * 255)

        return Forward(self.net, fn)


class Kernels(Portable):
    """CUDA: the network through `fastforks._optimize` (fp16, channels-last, fp16-accumulating convs for big batches).

    Normalisation, the network and decoding replay as one CUDA graph per person bucket (`BUCKETS`, zero crops pad a
    batch up; `cuda_graphs = false`: eager). Cropping and the mapping back run eagerly.
    """

    crop = staticmethod(pose_warp.crop)

    def __init__(self, ctx: Context) -> None:
        """The portable network, then its optimized version (replaces it: a second set of weights was 121 MiB)."""
        super().__init__(ctx)
        self.net = self.fast = optimize(self.net.float(), mode="kernels", device=ctx.device, family="cnn")
        a: Params = ctx.params
        self.graphs = SmallShapeGraphs(
            self._decoded, pixels=BUCKETS[-1] * a.height * a.width, enabled=ctx.params.cuda_graphs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """float32 heatmaps of normalised float32 crops."""
        return self.fast(x).float()

    def _decoded(self, crops: torch.Tensor) -> torch.Tensor:
        """`[P, K, 3]` float32 (x, y in network input pixels, score) of crops `[P, 3, h, w]` in 0..255."""
        coords, score = decode(self.heatmaps(crops))
        return torch.cat([coords, score[..., None]], -1)

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        self.graphs.release()

    def people(self, crops: torch.Tensor, centre: torch.Tensor, size: torch.Tensor) -> dict[str, torch.Tensor]:
        """The graphed network and decoding per person bucket, mapped back to the input."""
        out = in_buckets(self.graphs, crops)
        kp = out[..., :2] * (size / self.net_size)[:, None] + (centre - size * 0.5)[:, None]
        score = out[..., 2]
        return {"keypoints": kp, "keypoint_scores": score, "scores": score.mean(1)}


def extra() -> dict[tuple[str, str], object]:
    """What the mmpose checkpoints pickle besides tensors: numpy arrays / scalars of the meta, and the training log."""
    logs = {
        ("mmengine.logging.history_buffer", "HistoryBuffer"): Inert,
        ("mmengine.logging.message_hub", "MessageHub"): Inert,
    }
    scalars = {(m, "scalar"): numeric_scalar for m in ("numpy.core.multiarray", "numpy._core.multiarray")}
    return {**NUMPY_ARRAYS, ("_codecs", "encode"): latin1_encode, **logs, **scalars}


def _network(ctx: Context) -> HRNet:
    """The network with the checkpoint (BatchNorm folded): float32, CPU."""
    net = HRNet(ctx.params)
    net.load_state_dict(convert(ctx.state_dict(key="state_dict", strip="", extra=extra())))
    return net.eval()


class _MlxPortable(Portable):
    """Apple MLX: the network written in MLX operations (`mlx_net`), crops and decoding as portable."""

    module = Runner.module  # the MLX network is not exportable
    CACHE_MB: ClassVar[dict[str, int]] = {}  # the buffer cache this mode asks for, by variant ('*': all)

    def __init__(self, ctx: Context) -> None:
        """The MLX network from the conversion cache (the first load converts the checkpoint's weights)."""
        TopDown.__init__(self, ctx)
        from fastforks.shared import mlx_port  # noqa: PLC0415 - imports mlx

        from . import mlx_net  # noqa: PLC0415 - imports mlx

        a: Params = ctx.params
        self.mlx = mlx_port.recorded(
            ctx,
            mlx_net.LAYOUT,
            lambda net: mlx_net.network(net, ctx.dtype),
            real=lambda: _network(ctx),
            skeleton=lambda: HRNet(a),
            sources=mlx_net.SOURCES,
        )
        self.net = None
        self.mean = torch.tensor(MEAN, device=ctx.device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=ctx.device).view(1, 3, 1, 1)
        self.flip = torch.tensor(FLIP[ctx.variant.output_info["label_set"]], device=ctx.device)
        self.net_size = torch.tensor([a.width, a.height], device=ctx.device, dtype=torch.float32)
        if mb := self.CACHE_MB.get(ctx.variant.id, self.CACHE_MB.get("*", 0)):  # held while this runner lives
            from fastforks._core.mlx import request_cache  # noqa: PLC0415 - MLX only

            request_cache(self, mb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """float32 heatmaps of normalised float32 crops."""
        from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - imports mlx

        return to_torch(self.mlx(to_mx(x.float())))


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": _MlxPortable,
}
# FASTFORKS_MLX_GRAPH=1 (diagnostics, tools/mlx_core_ab.py): the exported-graph port the hand-written network replaced
GRAPH_RUNNERS = {"portable_mlx": mlx_wrap.port(Portable)}
