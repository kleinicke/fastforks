"""RF-DETR Keypoint: square bilinear RGB resize, ImageNet normalization, person queries and COCO joints.

The pinned upstream PostProcess preserves uncertainty-fused person scores and per-joint findability confidence.
The background slot has no joints; only the person class is returned. No NMS or GT person boxes are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import Inert
from fastforks._core.patch import base_type, override
from fastforks._core.upload import device_tensor
from fastforks.kernels import rfdetr_input
from fastforks.pose._runner import Forward, Runner
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import GraphCache
from fastforks.shared.rfdetr.attention import pack_attention
from fastforks.shared.rfdetr.build import Arch, export_path, network

if TYPE_CHECKING:
    from fastforks.pose._runner import Context

MEAN = (0.485, 0.456, 0.406)  # fixed: upstream ImageNet RGB normalization
STD = (0.229, 0.224, 0.225)  # fixed: upstream ImageNet RGB normalization
PERSON = 1  # fixed: checkpoint schema [background, COCO person]


@dataclass(frozen=True)
class Params(Arch):
    """Card architecture plus the public confidence threshold."""

    score_threshold: float


def _keypoint_delta(head: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Convert the head output to float32 before decoding joint coordinates."""
    return base_type(head).forward(head, x).float()


def _person_slots(net: Any, points: torch.Tensor, batch: int, queries: int) -> torch.Tensor:  # noqa: ANN401 - LWDETR
    """Pad the compact person joints with the background class in one operation."""
    joints = net.num_keypoints_per_class[PERSON]
    if points.ndim == 4 and points.shape[:3] == (batch, queries, joints):  # noqa: PLR2004
        return F.pad(points, (0, 0, joints, 0))
    return base_type(net)._format_keypoint_output(net, points, batch, queries)  # noqa: SLF001


class Portable(Runner):
    """The pinned upstream network with preprocessing and float32 coordinate decoding."""

    needs_boxes: ClassVar[bool] = False
    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build lazily with optional libraries, load checked weights and prepare normalization."""
        super().__init__(ctx)
        self.net = network(ctx.params)
        state = ctx.state_dict(key="model", strip="", extra={("argparse", "Namespace"): Inert})
        # Upstream models/weights.py explicitly ignores this obsolete, unused keypoint projection.
        # Remove only its four known tensors; every remaining checkpoint key is checked strictly.
        state = dict(state)
        for layer in (0, 2):
            for suffix in ("weight", "bias"):
                state.pop(f"keypoint_head.keypoint_proj.{layer}.{suffix}", None)
        self.net.load_state_dict(state)
        self.net = self.net.eval().to(ctx.device, ctx.torch_dtype)
        if ctx.mode == "portable" and ctx.device.type == "cuda":
            export_path(self.net)
        if ctx.mode == "portable" and ctx.device.type == "cuda" and ctx.dtype == "float16":
            pack_attention(self.net)
            override(cast("torch.nn.Module", self.net.keypoint_embed), _keypoint_delta)
            if tuple(ctx.params.num_keypoints_per_class) == (0, 17):
                override(self.net, _person_slots, "_format_keypoint_output")
        from fastforks.shared.rfdetr.net.models.postprocess import PostProcess  # noqa: PLC0415 - after optional dependencies

        self.postprocess = PostProcess(
            num_select=ctx.params.num_select,
            num_keypoints_per_class=list(ctx.params.num_keypoints_per_class),
            trace_alpha=ctx.params.trace_alpha,
        )
        self.mean = torch.tensor(MEAN, device=ctx.device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=ctx.device).view(1, 3, 1, 1)

    def _prepare(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize the square resized image in the upstream operation order."""
        size = self.ctx.params.resolution
        resized = F.interpolate(image.float(), size=(size, size), mode="bilinear", align_corners=False)
        return ((resized - self.mean) / self.std).to(self.ctx.torch_dtype)

    def _raw(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.ctx.device.type != "cuda":
            return self.net(self._prepare(image))
        boxes, logits, points = self.net(self._prepare(image))
        return {"pred_boxes": boxes, "pred_logits": logits, "pred_keypoints": points}

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002
        """Decode the preview checkpoint's person slot with a single output compaction."""
        raw = self._raw(image)
        post = self.postprocess
        obj, labels, query = post._select_topk(raw["pred_logits"].float())  # noqa: SLF001
        height, width = image.shape[-2:]
        size = device_tensor([[height, width]], device=image.device)
        xyxy = post._gather_and_scale_boxes(raw["pred_boxes"].float(), query, size)[0]  # noqa: SLF001
        joints = post.num_keypoints_per_class[PERSON]
        points = raw["pred_keypoints"][0, :, joints:].index_select(0, query[0]).float()
        scores = obj[0]
        if post.trace_alpha > 0 and points.shape[-1] >= 7:  # noqa: PLR2004 - upstream Cholesky layout
            trace = post._keypoint_log_mean_trace(points)  # noqa: SLF001 - preserve upstream uncertainty
            scores = (scores.log() - post.trace_alpha * trace).sigmoid().clamp_max_(1 - 2**-24)
        keep = ((labels[0] == PERSON) & (scores > self.ctx.params.score_threshold)).nonzero().flatten()
        kept = points.index_select(0, keep)
        return {
            "keypoints": kept[..., :2] * size[0].flip(0),
            "keypoint_scores": kept[..., 2].sigmoid(),
            "boxes": xyxy.index_select(0, keep),
            "scores": scores.index_select(0, keep),
        }

    def module(self) -> torch.nn.Module:
        """Resize/normalize RGB `[B,3,H,W]` in [0,1], then return raw boxes, logits and keypoints."""

        def fn(_net: Any, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # noqa: ANN401 - vendored network
            raw = self._raw(image)
            return raw["pred_boxes"], raw["pred_logits"], raw["pred_keypoints"]

        return Forward(self.net, fn, output_names=("boxes", "logits", "keypoints"))


class Kernels(Portable):
    """Half-precision upstream network with packed-QKV fastbb backbone attention."""

    module = Runner.module  # custom attention is not an exportable PyTorch operator

    def __init__(self, ctx: Context) -> None:
        """Load the same weights, then replace the DINOv2 attention with the declared kernel backend."""
        super().__init__(ctx)
        from fastforks.shared.rfdetr.kernels import (  # noqa: PLC0415 - CUDA backend only
            prepare_graph,
            use_fast_attention,
        )

        use_fast_attention(self.net)
        # The authors offer this fixed-shape inference path and spatial-tensor cache for graph capture.
        prepare_graph(self.net)
        self.graphs = GraphCache(self.net, graphs=1)

    def _raw(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x = rfdetr_input.prepare(image.float(), self.mean, self.std, self.ctx.params.resolution)
        raw = self.graphs.run((x.device, x.dtype, tuple(x.shape)), x)
        if raw is None:
            raw = self.net(x)
        boxes, logits, points = cast("tuple[torch.Tensor, torch.Tensor, torch.Tensor]", raw)
        return {"pred_boxes": boxes, "pred_logits": logits, "pred_keypoints": points}

    def release_graphs(self) -> None:
        """Release the fixed network-input graph and its private memory pool."""
        self.graphs.release()


class _MlxPortable(Portable):
    """Apple MLX: the network written in MLX operations (`fastforks.shared.rfdetr.mlx_net`)."""

    module = Runner.module  # the network is MLX, not an exportable PyTorch module

    def __init__(self, ctx: Context) -> None:
        """Build the PyTorch network (float32, CPU) and copy its weights to MLX."""
        super().__init__(ctx)
        from fastforks.shared.rfdetr import mlx_net  # noqa: PLC0415 - imports mlx

        self.mlx = mlx_net.MlxRFDETR(self.net, mlx_wrap.run_dtype(ctx), MEAN, STD)
        del self.net  # the PyTorch twin: only its weights were needed

    def _raw(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.mlx(image)  # float32 CPU tensors


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(_MlxPortable, attrs=()),
}
