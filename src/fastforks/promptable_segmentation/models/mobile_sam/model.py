"""MobileSAM runners: weights -> network, and the portable / kernels image encoders (shared prompt side).

Both follow upstream `SamPredictor`: `set_image` (resize the longest side to 1024, normalise, pad, encode) and
`predict_torch` (prompt coordinates scaled into the 1024 frame, prompt encoder + mask decoder in float32, mask
`postprocess_masks`: upsample to the 1024 square, crop, upsample to the input size). The prompt side is Segment
Anything's own (`fastforks.shared.sam_prompt`).

The TinyViT image encoder (MBConv stage, three stages of window attention + depthwise conv + MLP, conv neck) runs
through `fastforks._optimize`: BatchNorm folding and fp16 channels-last in `portable`, the fused convolution kernels
in `kernels`. Its neck's `LayerNorm2d`s run in float32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks._optimize import optimize
from fastforks.promptable_segmentation._runner import Forward, Runner
from fastforks.shared.cuda_graphs import GraphCache
from fastforks.shared.sam_prompt import decode, prepare, preprocess_shape
from fastforks.shared.sam_prompt.net.mask_decoder import MaskDecoder
from fastforks.shared.sam_prompt.net.prompt_encoder import PromptEncoder
from fastforks.shared.sam_prompt.net.transformer import TwoWayTransformer

from .fast import Trunk
from .net.tiny_vit_sam import TinyViT

if TYPE_CHECKING:
    from fastforks._optimize import FastModel
    from fastforks.promptable_segmentation._runner import Context

PROMPT_DIM = 256  # fixed: SAM's prompt / mask-decoder embedding width
TOKENS = 64  # fixed: SAM's image embedding side
GRAPH_PROMPTS = 4  # fixed: as EfficientViT-SAM (same prompt side: a graph pays off up to 4 prompts)
GRAPHS = 4  # fixed: prompt layouts with a graph at a time (points, a box, both, with a mask)


@dataclass(frozen=True)
class Params:
    """Card `[params]`."""

    image_size: int  # encoder input side and the prompt frame (longest image side, padded to a square)


class Sam(torch.nn.Module):
    """Upstream `Sam`'s three modules under its names (the checkpoint's key prefixes)."""

    def __init__(self, p: Params) -> None:
        """Upstream `build_sam_vit_t` (untrained)."""
        super().__init__()
        self.image_encoder = TinyViT(
            img_size=p.image_size,
            in_chans=3,
            num_classes=1000,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
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


def _sam(ctx: Context) -> Sam:
    sam = Sam(ctx.params)
    load_strict(sam, ctx.state_dict(), f"{ctx.variant.ref} checkpoint")
    return sam.eval()


class _Sam(Runner):
    """Encoder through `fastforks._optimize` and the prompt side (float32, upstream modules), for both backends."""

    batch = 1
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build on the CPU, optimise the encoder trunk for the mode, move the float32 rest to the device."""
        super().__init__(ctx)
        sam = _sam(ctx)
        self.trunk: FastModel = optimize(
            Trunk(sam.image_encoder), mode=ctx.mode, device=ctx.device, fp16=ctx.dtype == "float16", family="cnn"
        )
        self.norm = sam.image_encoder.neck[3].to(ctx.device)
        del sam.image_encoder
        self.sam = sam.to(ctx.device)
        if ctx.device.type == "cuda":  # the fp32 weights of the optimise passes stay cached otherwise
            torch.cuda.empty_cache()

    def _encode(self, trunk: torch.nn.Module, image: torch.Tensor, *, exact: bool) -> torch.Tensor:
        x = prepare(image, self.ctx.io, self.ctx.params.image_size, exact=exact)
        return self.norm(trunk(x).float())

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
        """Upstream `SamPredictor.predict_torch` (prompts scaled like `ResizeLongestSide.apply_coords`)."""
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
        """All four masks and scores for prompts in the 1024 frame (`sam_prompt.decode`)."""
        pts, labels = points or (None, None)
        pe, dec = self.sam.prompt_encoder, self.sam.mask_decoder
        return decode(pe, dec, embedding, points=pts, labels=labels, boxes=boxes, masks=masks)

    def upsample(self, pred_masks: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Upstream `Sam.postprocess_masks`."""
        s = self.ctx.params.image_size
        nh, nw = preprocess_shape(*size, s)
        m = F.interpolate(pred_masks, (s, s), mode="bilinear", align_corners=False)[..., :nh, :nw]
        return F.interpolate(m, size, mode="bilinear", align_corners=False)

    def networks(self) -> tuple[torch.nn.Module, ...]:
        """The whole model: the optimised encoder trunk, its last norm, prompt encoder and mask decoder."""
        return (self.trunk, self.norm, self.sam)


class Portable(_Sam):
    """Plain PyTorch: BatchNorm folded into the convolutions; float16 channels-last image encoder on CUDA."""

    dtypes = ("float32", "float16")

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
    """Our kernels: the encoder's convolutions as fused fp16 kernels (`fastforks._optimize`).

    The prompt encoder and mask decoder are launch-bound, so up to `GRAPH_PROMPTS` prompts of a repeated prompt layout
    replay a CUDA graph (`fastforks.shared.cuda_graphs`; `release_graphs()` frees its pool).
    """

    def __init__(self, ctx: Context) -> None:
        """As `_Sam`, plus the prompt-decode graph cache."""
        super().__init__(ctx)
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
        """`sam_prompt.decode` on the cache's static inputs; a prompt kind is told by its rank (labels: integers)."""
        by = {(t.ndim, t.is_floating_point()): t for t in given}
        points, labels, boxes, masks = (by.get(k) for k in ((3, True), (2, False), (2, True), (4, True)))
        pe, dec = self.sam.prompt_encoder, self.sam.mask_decoder
        return decode(pe, dec, embedding, points=points, labels=labels, boxes=boxes, masks=masks)

    def release_graphs(self) -> None:
        """Free the prompt-decode CUDA graphs and their memory pool (they are captured again when needed)."""
        self.cache.release()


RUNNERS = {"portable": Portable, "kernels": Kernels}
