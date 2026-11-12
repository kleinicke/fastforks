"""RF-DETR: square bilinear RGB resize, ImageNet normalization, the pinned upstream network and PostProcess.

PostProcess takes the top `num_select` (query, class) pairs of the sigmoid scores; the COCO category ids are mapped to
`coco-80` and the unused ids dropped. Seg variants upsample the selected mask logits to the input size (> 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import Inert
from fastforks.detection._runner import Forward, Runner
from fastforks.shared.cuda_graphs import GraphCache
from fastforks.shared.rfdetr.build import Arch, network

if TYPE_CHECKING:
    from fastforks.detection._runner import Context

MEAN = (0.485, 0.456, 0.406)  # fixed: upstream ImageNet RGB normalization
STD = (0.229, 0.224, 0.225)  # fixed: upstream ImageNet RGB normalization
UNUSED_IDS = (12, 26, 29, 30, 45, 66, 68, 69, 71, 83)  # fixed: COCO category ids without a class (1..90)
RAW = ("pred_boxes", "pred_logits", "pred_masks")  # forward_export's output order


@dataclass(frozen=True)
class Params(Arch):
    """Card architecture plus the public options."""

    score_threshold: float
    max_detections: int


def class_map(num_ids: int) -> torch.Tensor:
    """COCO category id (index) -> `coco-80` label; -1 for id 0 and the unused ids."""
    out, nxt = torch.full((num_ids,), -1, dtype=torch.int64), 0
    for i in range(1, min(num_ids, 91)):
        if i not in UNUSED_IDS:
            out[i], nxt = nxt, nxt + 1
    return out


class Portable(Runner):
    """The pinned upstream network; float16 on CUDA, PostProcess in float32."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build lazily with optional libraries, load checked weights and prepare normalization."""
        super().__init__(ctx)
        self.net = network(ctx.params)
        state = dict(ctx.state_dict(key="model", strip="", extra={("argparse", "Namespace"): Inert}))
        # Upstream models/weights.py: the empty keypoint-schema buffer is deterministic and absent from these
        # detection checkpoints; every other key is checked strictly.
        state.setdefault("_kp_active_mask", self.net.get_buffer("_kp_active_mask"))
        # Upstream models/weights.py: a checkpoint trained with more queries (Seg L: 300) keeps the first
        # `num_queries` of each of the `group_detr` query groups.
        p = ctx.params
        for k in ("refpoint_embed.weight", "query_feat.weight"):
            w = state[k]
            if w.shape[0] != p.num_queries * p.group_detr and w.shape[0] % p.group_detr == 0:
                state[k] = w.view(p.group_detr, -1, w.shape[-1])[:, : p.num_queries].reshape(-1, w.shape[-1])
        self.net.load_state_dict(state)
        self.net = self.net.eval().to(ctx.device, ctx.torch_dtype)
        from fastforks.shared.rfdetr.net.models.postprocess import PostProcess  # noqa: PLC0415 - after optional dependencies

        self.postprocess = PostProcess(num_select=ctx.params.num_select)
        self.ids = class_map(ctx.params.num_classes + 1).to(ctx.device)
        self.mean = torch.tensor(MEAN, device=ctx.device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=ctx.device).view(1, 3, 1, 1)

    def _prepare(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize the square resized image in the upstream operation order."""
        size = self.ctx.params.resolution
        resized = F.interpolate(image.float(), size=(size, size), mode="bilinear", align_corners=False)
        return ((resized - self.mean) / self.std).to(self.ctx.torch_dtype)

    def _raw(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return {k: v for k, v in self.net(self._prepare(image)).items() if k in RAW}

    def __call__(self, image: torch.Tensor, score_threshold: float) -> dict[str, torch.Tensor]:
        """Every detection above the threshold (best first, at most `max_detections`), in input pixels."""
        raw = {k: v.float() for k, v in self._raw(image).items()}
        size = torch.tensor([image.shape[-2:]], device=image.device)
        r = self.postprocess(raw, target_sizes=size, score_threshold=score_threshold)[0]
        labels = self.ids[r["labels"]]
        keep = ((labels >= 0) & (r["scores"] > score_threshold)).nonzero()[:, 0][: self.ctx.params.max_detections]
        out = {"boxes": r["boxes"][keep], "scores": r["scores"][keep], "labels": labels[keep]}
        if "masks" in r:
            out["masks"] = r["masks"][keep, 0]
        return out

    def module(self) -> torch.nn.Module:
        """The network with its input handling: RGB `[B, 3, H, W]` in [0, 1] -> raw outputs.

        Boxes `[B, Q, 4]` (centre x / y, width, height; 0..1), class logits `[B, Q, 91]` (sigmoid; index = COCO
        category id) and, for Seg, mask logits `[B, Q, h, w]`.
        """
        names = ("boxes", "logits", "masks")[: 2 + self.ctx.params.segmentation_head]

        def fn(_net: Any, image: torch.Tensor) -> tuple[torch.Tensor, ...]:  # noqa: ANN401 - vendored network
            raw = self._raw(image)
            return tuple(raw[k] for k in RAW[: len(names)])

        return Forward(self.net, fn, output_names=names, query_scores=lambda logits: logits.sigmoid().amax(-1))


class Kernels(Portable):
    """Half-precision upstream network with packed-QKV fastbb backbone attention and CUDA graph replay."""

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
        return dict(zip(RAW, cast("tuple[torch.Tensor, ...]", raw), strict=False))

    def release_graphs(self) -> None:
        """Release the fixed network-input graph and its private memory pool."""
        self.graphs.release()


RUNNERS = {"portable": Portable, "kernels": Kernels}
