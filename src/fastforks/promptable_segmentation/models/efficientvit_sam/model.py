"""EfficientViT-SAM runners: weights -> network, and the portable / kernels image encoders (shared prompt side).

Both follow upstream `EfficientViTSamPredictor`: `set_image` (resize the longest side to the encoder's `input_size`,
normalise, pad, encode) and `predict_torch` (prompt coordinates scaled into SAM's 1024 frame, prompt encoder + mask
decoder in float32, mask `postprocess_masks`: upsample to the 1024 square, crop, upsample to the input size).

The image encoder (conv stages, ReLU linear attention, conv neck) runs through `fastforks._optimize` like the
classification EfficientViT: BatchNorm folding and fp16 channels-last in `portable`, the fused convolution and
LiteMLA kernels in `kernels`. Its last `LayerNorm2d` runs in float32 on the float32 output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from torch import fx

from fastforks._core.checkpoint import load_strict
from fastforks._optimize import optimize
from fastforks.promptable_segmentation._runner import Forward, Runner
from fastforks.shared.cuda_graphs import GraphCache

from .fast import Trunk, decode, prepare, preprocess_shape, tensor_eps, wide_points
from .net.backbone import EfficientViTLargeBackbone
from .net.norm import set_norm_eps
from .net.ops import LiteMLA
from .net.sam import EfficientViTSamImageEncoder, SamNeck
from .net.segment_anything.mask_decoder import MaskDecoder
from .net.segment_anything.prompt_encoder import PromptEncoder
from .net.segment_anything.transformer import TwoWayTransformer

if TYPE_CHECKING:
    from fastforks._optimize import FastModel
    from fastforks.promptable_segmentation._runner import Context

PROMPT_DIM = 256  # fixed: SAM's prompt / mask-decoder embedding width
TOKENS = 64  # fixed: SAM's image embedding side
# RTX 5060 Ti, prompt decode eager -> graph: 1 prompt 4.02 -> 1.46 ms, 4 prompts 6.06 -> 5.41 ms, 16 prompts
# 24.7 -> 24.2 ms (GPU-bound from there on, where a graph only holds memory)
GRAPH_PROMPTS = 4  # fixed: measured, see above
GRAPHS = 4  # fixed: prompt layouts with a graph at a time (points, a box, both, with a mask)
NORM_EPS = 1e-6  # fixed: upstream `sam_model_zoo.create_efficientvit_sam_model` (`set_norm_eps(model, 1e-6)`)

# upstream `efficientvit_sam_*` (models/efficientvit/sam.py) and `efficientvit_backbone_l*`: backbone and neck arguments
_L = {"width_list": [32, 64, 128, 256, 512]}
_XL: dict[str, Any] = {
    "width_list": [32, 64, 128, 256, 512, 1024],
    "block_list": ["res", "fmb", "fmb", "fmb", "att@3", "att@3"],
    "expand_list": [1, 4, 4, 4, 4, 6],
    "fewer_norm_list": [False, False, False, False, True, True],
}
_L_NECK: dict[str, Any] = {"fid_list": ["stage4", "stage3", "stage2"], "in_channel_list": [512, 256, 128]}
_XL_NECK: dict[str, Any] = {"fid_list": ["stage5", "stage4", "stage3"], "in_channel_list": [1024, 512, 256]}
ARCHS: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "l0": ({**_L, "depth_list": [1, 1, 1, 4, 4]}, {**_L_NECK, "head_depth": 4, "expand_ratio": 1}),
    "l1": ({**_L, "depth_list": [1, 1, 1, 6, 6]}, {**_L_NECK, "head_depth": 8, "expand_ratio": 1}),
    "l2": ({**_L, "depth_list": [1, 2, 2, 8, 8]}, {**_L_NECK, "head_depth": 12, "expand_ratio": 1}),
    "xl0": ({**_XL, "depth_list": [0, 1, 1, 2, 3, 3]}, {**_XL_NECK, "head_depth": 6, "expand_ratio": 4}),
    "xl1": ({**_XL, "depth_list": [1, 2, 2, 4, 6, 6]}, {**_XL_NECK, "head_depth": 12, "expand_ratio": 4}),
}


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params`."""

    image_size: int  # SAM's prompt frame (longest image side for prompt coordinates and mask upsampling)
    arch: str  # key of `ARCHS`
    input_size: int  # encoder input side (longest image side, padded to a square)


class Sam(torch.nn.Module):
    """Upstream `EfficientViTSam`'s three modules under its names (the checkpoint's key prefixes)."""

    def __init__(self, p: Params) -> None:
        """Upstream `efficientvit_sam_<arch>` (untrained), norm eps as the model zoo sets them."""
        super().__init__()
        backbone, neck = ARCHS[p.arch]
        self.image_encoder = EfficientViTSamImageEncoder(
            EfficientViTLargeBackbone(**backbone), SamNeck(head_width=256, middle_op="fmb", **neck)
        )
        self.prompt_encoder = PromptEncoder(
            embed_dim=PROMPT_DIM,
            image_embedding_size=(TOKENS, TOKENS),
            input_image_size=(p.image_size, p.image_size),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embedding_dim=PROMPT_DIM, mlp_dim=2048, num_heads=8),
            transformer_dim=PROMPT_DIM,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        set_norm_eps(self, NORM_EPS)


def _sam(ctx: Context) -> Sam:
    sam = Sam(ctx.params)
    load_strict(sam, ctx.state_dict(), f"{ctx.variant.ref} checkpoint")
    return sam.eval()


def _timm_layout(net: torch.nn.Module) -> None:
    """Give upstream `LiteMLA`s the attribute layout of timm's (`Identity` where upstream stores `None`).

    `fastforks._optimize` swaps a LiteMLA for its kernel form by that layout; a `ConvLayer` calls a truthy norm / act,
    so the identities change nothing.
    """
    for m in net.modules():
        if isinstance(m, LiteMLA):
            for conv in (m.qkv, m.proj):
                for name in ("norm", "act"):
                    if getattr(conv, name) is None:
                        setattr(conv, name, torch.nn.Identity())


class _Sam(Runner):
    """Encoder through `fastforks._optimize` and the prompt side (float32, upstream modules), for both backends."""

    batch = 1
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build on the CPU, optimise the encoder trunk for the mode, move the float32 rest to the device."""
        super().__init__(ctx)
        sam = _sam(ctx)
        trunk = Trunk(sam.image_encoder, ctx.params.input_size)
        _timm_layout(trunk)
        self.trunk: FastModel = optimize(
            trunk, mode=ctx.mode, device=ctx.device, fp16=ctx.dtype == "float16", family="cnn"
        )
        self.norm = cast("torch.nn.Module", sam.image_encoder.norm).to(ctx.device)
        del sam.image_encoder
        self.sam = sam.to(ctx.device)
        if ctx.device.type == "cuda":  # the fp32 weights of the optimise passes stay cached otherwise
            torch.cuda.empty_cache()

    def _encode(self, trunk: torch.nn.Module, image: torch.Tensor, *, exact: bool) -> torch.Tensor:
        x = prepare(image, self.ctx.io, self.ctx.params.input_size, exact=exact)
        return self.norm(trunk(x))

    def embed(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Image embedding `[1, 256, 64, 64]` float32."""
        return {"image_embeddings": self._encode(self.trunk, image, exact=image.device.type == "cpu")}

    def predict(  # noqa: PLR0913 - one argument per prompt kind
        self,
        features: dict[str, torch.Tensor],
        size: tuple[int, int],
        *,
        points: torch.Tensor | None,
        labels: torch.Tensor | None,
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
        multimask: bool,
    ) -> dict[str, torch.Tensor]:
        """Upstream `EfficientViTSamPredictor.predict_torch` (prompts scaled like its `apply_coords`)."""
        (h, w), s = size, self.ctx.params.image_size
        nh, nw = preprocess_shape(h, w, s)
        dev = self.ctx.device
        f64 = torch.device("cpu") if dev.type == "mps" else dev  # MPS has no float64
        xy = torch.tensor([nw / w, nh / h], device=f64, dtype=torch.float64)
        pts = None if points is None or labels is None else ((points.to(f64).double() * xy).float().to(dev), labels)
        box = None if boxes is None else (boxes.view(-1, 2, 2).to(f64).double() * xy).float().view(-1, 4).to(dev)
        low, iou = self._decode(features["image_embeddings"], pts, box, masks)
        keep = slice(1, None) if multimask else slice(0, 1)  # upstream `MaskDecoder.forward`
        return {"pred_masks": low[:, keep], "iou_scores": iou[:, keep]}

    def _decode(
        self,
        embedding: torch.Tensor,
        points: tuple[torch.Tensor, torch.Tensor] | None,
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All four masks and scores for prompts in the 1024 frame (`fast.decode`)."""
        pts, labels = points or (None, None)
        pe, dec = self.sam.prompt_encoder, self.sam.mask_decoder
        return decode(pe, dec, embedding, points=pts, labels=labels, boxes=boxes, masks=masks)

    def upsample(self, pred_masks: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Upstream `EfficientViTSam.postprocess_masks`."""
        s = self.ctx.params.image_size
        nh, nw = preprocess_shape(*size, s)
        m = F.interpolate(pred_masks, (s, s), mode="bilinear", align_corners=False)[..., :nh, :nw]
        return F.interpolate(m, size, mode="bilinear", align_corners=False)

    def networks(self) -> tuple[torch.nn.Module, ...]:
        """The whole model: the optimised encoder trunk, its norm, prompt encoder and mask decoder."""
        return (self.trunk, self.norm, self.sam)


class Portable(_Sam):
    """Plain PyTorch: BatchNorm folded into the convolutions; float16 channels-last image encoder on CUDA."""

    dtypes = ("float32", "float16")

    def __init__(self, ctx: Context) -> None:
        """As `_Sam`; the attention epsilons as tensors, which the ONNX export keeps (`fast.tensor_eps`)."""
        super().__init__(ctx)
        tensor_eps(self.trunk.net, ctx.device)

    def module(self) -> torch.nn.Module:
        """The image encoder only (no prompt side): `[1, 3, H, W]` in `[0, 1]` -> `image_embeddings` `[1, 256, 64, 64]`.

        Torch resize, as on CUDA.
        """
        return Forward(
            self.trunk.traceable(),
            lambda net, image: self._encode(net, image, exact=False),
            output_names=("image_embeddings",),
        )


class Kernels(_Sam):
    """Our kernels: fused fp16 convolutions and the LiteMLA linear-attention kernel (`fastforks._optimize`).

    The linear 1x1 convs from 1024 and more channels accumulate in float32 (`fast.wide_points`). The prompt encoder
    and mask decoder are launch-bound, so up to `GRAPH_PROMPTS` prompts of a repeated prompt layout replay a CUDA
    graph (`fastforks.shared.cuda_graphs`; `release_graphs()` frees its pool).
    """

    def __init__(self, ctx: Context) -> None:
        """As `_Sam`, plus the float32-accumulate point convs and the prompt-decode graph cache."""
        super().__init__(ctx)
        net = self.trunk.net
        if not isinstance(net, fx.GraphModule):
            msg = "the kernels image encoder is not a traced graph"
            raise TypeError(msg)
        self.trunk.report["points"] = wide_points(net)
        self.cache = GraphCache(self._graph_forward, graphs=GRAPHS)

    def _decode(
        self,
        embedding: torch.Tensor,
        points: tuple[torch.Tensor, torch.Tensor] | None,
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """As `_Sam._decode`, from a CUDA graph per prompt layout (prompts, points per prompt, box, mask)."""
        given = [t for t in (*(points or ()), boxes, masks) if t is not None]
        if not given or len(given[0]) > GRAPH_PROMPTS:
            return super()._decode(embedding, points, boxes, masks)
        key = (len(given[0]), points[0].shape[1] if points else 0, boxes is not None, masks is not None)
        out = self.cache.run(key, embedding, *given)
        return super()._decode(embedding, points, boxes, masks) if out is None else out

    def _graph_forward(self, embedding: torch.Tensor, *given: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`fast.decode` on the cache's static inputs; a prompt kind is told by its rank (labels: integers)."""
        by = {(t.ndim, t.is_floating_point()): t for t in given}
        points, labels, boxes, masks = (by.get(k) for k in ((3, True), (2, False), (2, True), (4, True)))
        pe, dec = self.sam.prompt_encoder, self.sam.mask_decoder
        return decode(pe, dec, embedding, points=points, labels=labels, boxes=boxes, masks=masks)

    def release_graphs(self) -> None:
        """Free the prompt-decode CUDA graphs and their memory pool (they are captured again when needed)."""
        self.cache.release()


RUNNERS = {"portable": Portable, "kernels": Kernels}
