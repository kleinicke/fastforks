"""D-FINE runner: weights -> network, and the portable inference pass.

Follows the official deploy script (`references/deploy/dfine_torch.py`): the input is resized to exactly 640 x 640
(aspect ratio ignored, PIL bilinear), values in `[0, 1]`, no mean / std; HGNetV2 -> hybrid encoder -> 300 queries chosen
from the encoder output -> deformable decoder whose layers refine each box edge as a distribution over distances
(fine-grained distribution refinement); the postprocess is RT-DETR's, so this family reuses its runner: portable runs
the backbone, encoder and decoder layers in float16 on CUDA, the query selection, the box refinement and the heads in
float32; kernels adds the fp16-accumulate convolutions and replays the network from a CUDA graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastforks.detection import _rtdetr_runner as base
from fastforks.kernels import cudnn_hacc
from fastforks.shared.cuda_graphs import GraphCache

from .net import Arch, DFine, convert

if TYPE_CHECKING:
    import torch

    from fastforks.detection._runner import Context


@dataclass(frozen=True)
class Params(Arch):
    """Card `[params]` + variant `params` + `[options]`."""

    size: int  # square network input
    score_threshold: float  # option: minimum class score
    max_detections: int  # option: detections per image
    cuda_graphs: bool  # option: kernels replays the network from a CUDA graph


class Portable(base.Portable):
    """Plain PyTorch; float16 backbone, encoder and decoder layers on CUDA (channels-last)."""

    Params = Params
    label = "D-FINE"

    def build(self, ctx: Context) -> torch.nn.Module:
        """The network with the checkpoint loaded (BatchNorm folded, RepVGG merged), in eval mode."""
        net = DFine(ctx.params)
        key = "model" if ctx.weights()[0].suffix == ".pth" else None  # the authors' (DEIM) training checkpoints
        net.load_state_dict(convert(ctx.state_dict(key=key), ctx.params.dec_layers))
        return net.eval()

    def float_heads(self) -> None:
        """Keep query selection, box refinement and the heads in float32."""
        net: Any = self.net  # the DFine built by `build`
        for m in (
            net.enc_output,
            net.enc_score_head,
            net.enc_bbox_head,
            net.decoder.query_pos_head,
            net.decoder.pre_bbox_head,
            net.decoder.bbox_embed,
            net.decoder.class_embed,
            net.decoder.lqe,
        ):
            m.float()


class Kernels(base.Kernels, Portable):
    """`cudnn_hacc` convolutions, fp16-accumulate GEMMs and CUDA graph replay (the RT-DETR runner's)."""

    cudnn_hacc = cudnn_hacc
    GraphCache = GraphCache


RUNNERS = {"portable": Portable, "kernels": Kernels}
