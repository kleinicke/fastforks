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
from fastforks.pose._runner import Forward, Runner
from fastforks.shared.cuda_graphs import GraphCache
from fastforks.shared.rfdetr.build import Arch, network

if TYPE_CHECKING:
    from fastforks.pose._runner import Context

MEAN = (0.485, 0.456, 0.406)  # fixed: upstream ImageNet RGB normalization
STD = (0.229, 0.224, 0.225)  # fixed: upstream ImageNet RGB normalization
PERSON = 1  # fixed: checkpoint schema [background, COCO person]


@dataclass(frozen=True)
class Params(Arch):
    """Card architecture plus the public confidence threshold."""

    score_threshold: float


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
        return self.net(self._prepare(image))

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - bottom-up runner contract
        """Return every person above the threshold, with 17 COCO joints in source-image coordinates."""
        raw = {
            k: v.float() for k, v in self._raw(image).items() if k in {"pred_logits", "pred_boxes", "pred_keypoints"}
        }
        size = torch.tensor([image.shape[-2:]], device=image.device)
        result = self.postprocess(raw, target_sizes=size)[0]
        keep = (result["labels"] == PERSON) & (result["scores"] > self.ctx.params.score_threshold)
        points = result["keypoints"][keep]
        return {
            "keypoints": points[..., :2],
            "keypoint_scores": points[..., 2],
            "boxes": result["boxes"][keep],
            "scores": result["scores"][keep],
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
        from fastforks.shared.rfdetr.kernels import prepare_graph, use_fast_attention  # noqa: PLC0415 - CUDA backend only

        use_fast_attention(self.net)
        # The authors offer this fixed-shape inference path and spatial-tensor cache for graph capture.
        prepare_graph(self.net)
        self.graphs = GraphCache(self.net, graphs=1)

    def _raw(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self._prepare(image)
        raw = self.graphs.run((x.device, x.dtype, tuple(x.shape)), x)
        if raw is None:
            raw = self.net(x)
        boxes, logits, points = cast("tuple[torch.Tensor, torch.Tensor, torch.Tensor]", raw)
        return {"pred_boxes": boxes, "pred_logits": logits, "pred_keypoints": points}

    def release_graphs(self) -> None:
        """Release the fixed network-input graph and its private memory pool."""
        self.graphs.release()


RUNNERS = {"portable": Portable, "kernels": Kernels}
