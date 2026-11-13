"""Runner base of the RT-DETR-style detection families (RT-DETRv2, D-FINE).

The portable pass, the official postprocessor and the kernels tier (fused convolutions, CUDA graph replay).

A family subclasses `Portable` (its `Params`, `label`, `build`, `float_heads`) and `Kernels`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from fastforks._core.errors import InputError
from fastforks._core.half import fp16_accumulate
from fastforks.detection._convs import FusedConvs
from fastforks.detection._resize import pil_bilinear
from fastforks.detection._rtdetr import ConvBn, RepVgg
from fastforks.detection._runner import Forward, Runner

if TYPE_CHECKING:
    from types import ModuleType

    from fastforks.detection._runner import Context
    from fastforks.shared.cuda_graphs import GraphCache

_SEEN = 64  # fixed: shapes counted towards a capture (GraphCache's default bound)


def prepare(image: torch.Tensor, size: int) -> torch.Tensor:
    """`[1, 3, H, W]` RGB in `[0, 1]` -> `[1, 3, size, size]` (PIL bilinear on the uint8 image)."""
    x = image.mul(255).round_()
    if tuple(image.shape[-2:]) != (size, size):
        x = pil_bilinear(x, (size, size))
    return x / 255


class Portable(Runner):
    """Plain PyTorch; float16 backbone, encoder and decoder layers on CUDA (a family sets the hooks above)."""

    dtypes = ("float32", "float16")
    label: str  # the family, in messages

    def __init__(self, ctx: Context) -> None:
        """Build the network (BatchNorm folded, RepVGG merged), convert it to the compute dtype on the device."""
        super().__init__(ctx)
        self.mf = torch.channels_last if ctx.device.type == "cuda" else torch.contiguous_format
        self.dt = ctx.torch_dtype
        self.net = self.build(ctx).to(ctx.device, self.dt).to(memory_format=self.mf)
        self.float_heads()

    def build(self, ctx: Context) -> torch.nn.Module:
        """The network with the checkpoint loaded (BatchNorm folded, RepVGG merged), in eval mode."""
        raise NotImplementedError

    def float_heads(self) -> None:
        """Keep query selection, box refinement and the heads in float32."""
        raise NotImplementedError

    def module(self) -> torch.nn.Module:
        """The network: RGB `[B, 3, H, W]` in `[0, 1]`, H and W multiples of 32 (no resize, no padding; default 640).

        Returns the class logits `[B, 300, 80]` (apply a sigmoid) and the boxes `[B, 300, 4]` (centre x / y, width,
        height; 0..1). The 300 queries are the encoder's top-k, so their order is not stable under small changes.
        """
        return Forward(
            self.net,
            lambda net, x: net(_quantise(x, self.label).to(self.dt)),
            output_names=("logits", "boxes"),
            query_scores=lambda logits: logits.sigmoid().amax(-1),
        )

    def __call__(self, image: torch.Tensor, score_threshold: float) -> dict[str, torch.Tensor]:
        """Official postprocessor on one image."""
        p: Any = self.ctx.params
        h, w = image.shape[-2:]
        x = prepare(image, p.size).to(self.dt).contiguous(memory_format=self.mf)
        logits, boxes = self._forward(x)
        scores, idx = logits.float().sigmoid()[0].flatten().topk(p.num_queries)
        labels, query = idx % p.num_classes, idx // p.num_classes
        cx, cy, bw, bh = boxes[0].float()[query].unbind(-1)
        out = torch.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), -1) * boxes.new_tensor([w, h, w, h])
        keep = scores >= score_threshold
        n = p.max_detections
        return {"boxes": out[keep][:n], "scores": scores[keep][:n], "labels": labels[keep][:n]}

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(x)


_MULTIPLE = 32  # fixed: the encoder's stride, as `size_range()` (`module()` takes multiples; the model call pads)


def _quantise(x: torch.Tensor, label: str) -> torch.Tensor:
    """`module()`'s input step: RGB rounded to 8 bits, as the model call's; H and W must be multiples of 32."""
    h, w = x.shape[-2:]
    if h % _MULTIPLE or w % _MULTIPLE:
        msg = (
            f"{label} module() / export takes H and W that are multiples of {_MULTIPLE} (it does not pad), got "
            f"{h}x{w}; the model call pads any size"
        )
        raise InputError(msg)
    return x.mul(255).round().div(255)


class Kernels(Portable):
    """cuDNN fused conv + ReLU (+ residual), fp16-accumulate convs (`cudnn_hacc`) and GEMMs, CUDA graph replay."""

    conv_modules: tuple[type, ...] = (ConvBn, RepVgg)  # modules whose `cv` runs their convolutions
    convs: type[FusedConvs] = FusedConvs  # how they run
    cudnn_hacc: ModuleType  # the family's kernel library and graph cache (the family's file imports them: card checks)
    GraphCache: type[GraphCache]

    def __init__(self, ctx: Context) -> None:
        """Build the network, route its convs through `cudnn_hacc` / cuDNN fused calls."""
        super().__init__(ctx)
        cv = self.convs(self.cudnn_hacc.load(ctx.device))
        for m in self.net.modules():
            if isinstance(m, self.conv_modules):
                m.cv = cv  # ty: ignore[invalid-assignment]
        self.classes = ctx.params.num_classes
        self._capture_at = self.cudnn_hacc.capture_after() + 1  # the call of a shape that captures its graph
        self._graphs = (
            self.GraphCache(self._eager, graphs=1, capture_after=self._capture_at - 1)
            if ctx.params.cuda_graphs
            else None
        )
        self._seen: dict[tuple, int] = {}  # calls per shape until its capture

    def _eager(self, x: torch.Tensor) -> torch.Tensor:
        """Class logits and boxes as one float32 `[B, Q, classes + 4]` tensor (one graph output)."""
        with fp16_accumulate():
            return torch.cat(self.net(x), dim=-1)

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = None
        if self._graphs is not None:
            key = (x.device, x.dtype, tuple(x.shape))
            fresh = key not in self._graphs.graphs
            if fresh:
                self._seen[key] = self._seen.pop(key, 0) + 1
                while len(self._seen) > _SEEN:
                    del self._seen[next(iter(self._seen))]
                if self._seen[key] == self._capture_at:  # the eager calls' cached blocks: not kept next to the pool
                    torch.cuda.empty_cache()
            out = self._graphs.run(key, x)
            if fresh and key in self._graphs.graphs:  # just captured: the capture warm-up's blocks
                self._seen.pop(key, None)
                torch.cuda.empty_cache()
        if out is None:
            out = self._eager(x)
        logits, boxes = out.split([self.classes, 4], dim=-1)
        return logits, boxes

    def release_graphs(self) -> None:
        """Free the CUDA graph and its memory pool."""
        if self._graphs is not None:
            self._graphs.release()
