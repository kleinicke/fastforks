"""RT-DETRv2 runner: weights -> network, and the portable inference pass.

Follows the official deploy script (`references/deploy/rtdetrv2_torch.py`):

- input: resized to exactly 640 x 640 (aspect ratio ignored, PIL bilinear reproduced bit-exactly in torch), values in
  `[0, 1]`, no mean / std
- PResNet-vd -> hybrid encoder (one AIFI layer, FPN + PAN) -> 300 queries chosen from the encoder output -> deformable
  decoder; class scores are sigmoids
- postprocess: the 300 x 80 scores are ranked jointly, the best 300 are kept (a query can appear with several classes),
  boxes centre / size -> corners times the original image size, then `score_threshold` and `max_detections` apply.

Portable runs the backbone, encoder and decoder layers in float16 on CUDA (channels-last); the query selection, the
box refinement and the heads stay float32. Kernels runs the convs as cuDNN fused conv + bias + ReLU (+ residual) or
fp16-accumulate convs (`cudnn_hacc`), the GEMMs with fp16 accumulation, and replays the network (fixed 640 x 640
input) from a CUDA graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastforks.detection import _rtdetr_runner as base
from fastforks.kernels import cudnn_hacc
from fastforks.shared.cuda_graphs import GraphCache

from .net import Arch, RtDetr, fold_bn

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
    """Plain PyTorch; float16 backbone, encoder and decoder layers on CUDA."""

    Params = Params
    label = "RT-DETRv2"  # in messages

    def build(self, ctx: Context) -> torch.nn.Module:
        """The network with the checkpoint loaded (BatchNorm folded, RepVGG merged), in eval mode."""
        net = RtDetr(ctx.params)
        net.load_state_dict(fold_bn(ctx.state_dict(key=("ema", "module"), strip="")))
        return net.eval()

    def float_heads(self) -> None:
        """Keep query selection, box refinement and the heads in float32."""
        dec: Any = self.net.decoder
        for m in (
            dec.enc_output,
            dec.enc_score_head,
            dec.enc_bbox_head,
            dec.query_pos_head,
            dec.dec_score_head,
            dec.dec_bbox_head,
        ):
            m.float()


class Kernels(base.Kernels, Portable):
    """cuDNN fused conv + ReLU (+ residual), fp16-accumulate convs (`cudnn_hacc`) and GEMMs, CUDA graph replay."""

    cudnn_hacc = cudnn_hacc
    GraphCache = GraphCache


RUNNERS = {"portable": Portable, "kernels": Kernels}
