"""HQ-SAM runners: weights -> network, and the portable / kernels image encoders (shared float32 prompt side).

Both follow upstream `SamPredictor` of SysCV/sam-hq: `set_image` (resize the longest side to `image_size`, normalise,
pad, encode) and `predict_torch` with its default `hq_token_only=False` (prompt coordinates scaled like the image,
prompt encoder + HQ mask decoder in float32, mask `postprocess_masks`: upsample to the padded square, crop, upsample
to the input size). The HQ feature depends on the image only, so `embed` computes it once and the prompt calls reuse
it (upstream recomputes it per call).
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks._core.upload import device_tensor
from fastforks.kernels.dpt_fused import load as load_dpt_fused
from fastforks.promptable_segmentation._runner import Forward, Runner
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import GraphCache

from .fast import decode, encode, hq_features, prepare, preprocess_shape, rel_tables, select
from .fused import FusedEncoder
from .net.image_encoder import ImageEncoderViT
from .net.mask_decoder_hq import MaskDecoderHQ
from .net.prompt_encoder import PromptEncoder
from .net.transformer import TwoWayTransformer

if TYPE_CHECKING:
    from fastforks.promptable_segmentation._runner import Context

PATCH = 16  # fixed: SAM's ViT patch size (all released checkpoints)
PROMPT_DIM = 256  # fixed: SAM's prompt / mask-decoder embedding width (all released checkpoints)
GRAPH_PROMPTS = 4  # fixed: RTX 5060 Ti: the graph takes one prompt from 4.4 to 2.6 ms; bound as EfficientViT-SAM
GRAPHS = 4  # fixed: prompt layouts with a graph at a time (points, a box, both, with a mask)


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params`."""

    image_size: int  # encoder input side (longest image side, padded to a square)
    window_size: int  # windowed-attention block size
    global_attn: int  # number of global-attention blocks, evenly spaced (upstream global_attn_indexes)
    embed_dim: int
    depth: int
    num_heads: int


class Sam(torch.nn.Module):
    """Upstream `Sam`'s three modules under its names (the checkpoint's key prefixes)."""

    def __init__(self, p: Params) -> None:
        """Upstream `build_sam._build_sam` for `p` (untrained)."""
        super().__init__()
        tokens = p.image_size // PATCH
        every = p.depth // p.global_attn
        self.image_encoder = ImageEncoderViT(
            depth=p.depth,
            embed_dim=p.embed_dim,
            img_size=p.image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),  # ty: ignore[invalid-argument-type] - upstream annotates a class, calls it
            num_heads=p.num_heads,
            patch_size=PATCH,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=tuple(range(every - 1, p.depth, every)),
            window_size=p.window_size,
            out_chans=PROMPT_DIM,
        )
        self.prompt_encoder = PromptEncoder(
            embed_dim=PROMPT_DIM,
            image_embedding_size=(tokens, tokens),
            input_image_size=(p.image_size, p.image_size),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoderHQ(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embedding_dim=PROMPT_DIM, mlp_dim=2048, num_heads=8),
            transformer_dim=PROMPT_DIM,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            vit_dim=p.embed_dim,
        )


def _mapped_checkpoint(ctx: Context) -> bool:
    """Use strict assignment from the verified Linux CUDA checkpoint."""
    return (
        sys.platform == "linux"
        and ctx.device.type == "cuda"
        and ctx.variant.ref == "sam-hq-vit-h@1"
        and ctx.user_weights is None
        and ctx.user_state is None
    )


def _sam(ctx: Context) -> Sam:
    assign = _mapped_checkpoint(ctx)
    with torch.device("meta") if assign else nullcontext():
        sam = Sam(ctx.params)
    state = ctx.state_dict()
    if assign and "image_encoder.pos_embed" in state:
        # The checkpoint's channel-major table becomes NHWC-contiguous in the original copy.
        state["image_encoder.pos_embed"] = state["image_encoder.pos_embed"].contiguous()
    load_strict(sam, state, f"{ctx.variant.ref} checkpoint", assign=assign)
    return sam.eval()


class _Sam(Runner):
    """The prompt side, shared by both backends (float32, upstream modules)."""

    batch = 1
    Params = Params
    sam: Sam

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
        xy = device_tensor([nw / w, nh / h], device=f64, dtype=torch.float64)
        pts = None if points is None or labels is None else ((points.to(f64).double() * xy).float().to(dev), labels)
        box = None if boxes is None else (boxes.view(-1, 2, 2).to(f64).double() * xy).float().view(-1, 4).to(dev)
        all_masks, iou = self._decode(features["image_embeddings"], features["hq_features"], pts, box, masks)
        low, score = select(all_masks, iou, multimask=multimask)
        return {"pred_masks": low, "iou_scores": score}

    def _decode(
        self,
        embedding: torch.Tensor,
        hq: torch.Tensor,
        points: tuple[torch.Tensor, torch.Tensor] | None,
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All five masks and four scores for prompts in the `image_size` frame (`fast.decode`)."""
        pts, labels = points or (None, None)
        pe, dec = self.sam.prompt_encoder, self.sam.mask_decoder
        return decode(pe, dec, embedding, hq, points=pts, labels=labels, boxes=boxes, masks=masks)

    def upsample(self, pred_masks: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Upstream `Sam.postprocess_masks`."""
        s = self.ctx.params.image_size
        nh, nw = preprocess_shape(*size, s)
        m = F.interpolate(pred_masks, (s, s), mode="bilinear", align_corners=False)[..., :nh, :nw]
        return F.interpolate(m, size, mode="bilinear", align_corners=False)


class Portable(_Sam):
    """Plain PyTorch: upstream modules, SDPA relative-position attention; float16 image encoder on CUDA."""

    dtypes = ("float32", "float16")

    def __init__(self, ctx: Context) -> None:
        """Build the network; the image encoder in the compute dtype."""
        super().__init__(ctx)
        self.sam = _sam(ctx)
        enc = self.sam.image_encoder
        if ctx.torch_dtype == torch.float16 and _mapped_checkpoint(ctx):
            # Upload the encoder directly in FP16; the prompts and HQ decoder run in FP32.
            enc.to(device=ctx.device, dtype=ctx.torch_dtype)
            self.sam.to(ctx.device)
        else:
            self.sam.to(ctx.device)
            enc.to(ctx.torch_dtype)
        tokens = ctx.params.image_size // PATCH
        self.rel = [rel_tables(b, tokens, ctx.torch_dtype) for b in enc.blocks]

    def _encode(self, image: torch.Tensor, *, exact: bool) -> tuple[torch.Tensor, torch.Tensor]:
        x = prepare(image, self.ctx.io, self.ctx.params.image_size, exact=exact)
        embedding, early = encode(self.sam.image_encoder, x, self.rel)
        return embedding, hq_features(self.sam.mask_decoder, embedding, early)

    def _enc(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding, early = encode(self.sam.image_encoder, x, self.rel)
        return embedding, hq_features(self.sam.mask_decoder, embedding, early)

    def embed(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Image embedding `[1, 256, 64, 64]` and HQ feature `[1, 32, 256, 256]`, float32."""
        x = prepare(image, self.ctx.io, self.ctx.params.image_size, exact=image.device.type == "cpu")
        embedding, hq = self._enc(x)
        return {"image_embeddings": embedding, "hq_features": hq}

    def module(self) -> torch.nn.Module:
        """The image side only (no prompts): `[1, 3, H, W]` in `[0, 1]` -> `image_embeddings`, `hq_features`.

        Torch resize, as on CUDA.
        """
        return Forward(
            self.sam,
            lambda _net, image: self._encode(image, exact=False),
            output_names=("image_embeddings", "hq_features"),
        )

    def networks(self) -> tuple[torch.nn.Module, ...]:
        """The whole model: image encoder, prompt encoder and mask decoder (its transposed convs)."""
        return (self.sam,)


class Kernels(_Sam):
    """dpt_fused kernels: fused-epilogue CUTLASS GEMMs (fp16 accumulation) and the neck conv; SDPA flash attention.

    The prompt encoder and mask decoder are launch-bound, so up to `GRAPH_PROMPTS` prompts of a repeated prompt
    layout replay a CUDA graph (`fastforks.shared.cuda_graphs`; `release_graphs()` frees its pool).
    """

    def __init__(self, ctx: Context) -> None:
        """Build on the CPU, convert the encoder, move the float32 prompt side to the device."""
        super().__init__(ctx)
        sam = _sam(ctx)
        self.encoder = FusedEncoder(sam.image_encoder, ctx.device, load_dpt_fused(ctx.device))
        del sam.image_encoder
        self.sam = sam.to(ctx.device)
        self.cache = GraphCache(self._graph_forward, graphs=GRAPHS, share_inputs=(0, 1))

    def embed(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Image embedding `[1, 256, 64, 64]` and HQ feature `[1, 32, 256, 256]`, float32."""
        x = prepare(image, self.ctx.io, self.ctx.params.image_size, exact=False)
        embedding, early = self.encoder(x)
        return {"image_embeddings": embedding, "hq_features": hq_features(self.sam.mask_decoder, embedding, early)}

    def _decode(
        self,
        embedding: torch.Tensor,
        hq: torch.Tensor,
        points: tuple[torch.Tensor, torch.Tensor] | None,
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """As `_Sam._decode`, from a CUDA graph per prompt layout (prompts, points per prompt, box, mask)."""
        given = [t for t in (*(points or ()), boxes, masks) if t is not None]
        if not given or len(given[0]) > GRAPH_PROMPTS:
            return super()._decode(embedding, hq, points, boxes, masks)
        key = (len(given[0]), points[0].shape[1] if points else 0, boxes is not None, masks is not None)
        out = self.cache.run(key, embedding, hq, *given)
        return super()._decode(embedding, hq, points, boxes, masks) if out is None else out

    def _graph_forward(
        self, embedding: torch.Tensor, hq: torch.Tensor, *given: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`fast.decode` on the cache's static inputs; a prompt kind is told by its rank (labels: integers)."""
        by = {(t.ndim, t.is_floating_point()): t for t in given}
        points, labels, boxes, masks = (by.get(k) for k in ((3, True), (2, False), (2, True), (4, True)))
        pe, dec = self.sam.prompt_encoder, self.sam.mask_decoder
        return decode(pe, dec, embedding, hq, points=points, labels=labels, boxes=boxes, masks=masks)

    def release_graphs(self) -> None:
        """Free the prompt-decode CUDA graphs and their memory pool (they are captured again when needed)."""
        self.cache.release()

    def networks(self) -> tuple[torch.nn.Module, ...]:
        """The float32 prompt side and HQ head (the fused encoder holds plain tensors)."""
        return (self.sam,)


class _MlxPortable(Portable):
    """Apple MLX: the image encoder and HQ feature written in MLX operations (`shared.mlx_sam`), prompts as portable."""

    module = Runner.module  # the MLX encoder is not exportable

    def __init__(self, ctx: Context) -> None:
        """The portable network; the image encoder's (and HQ feature's) weights copied to MLX, the encoder dropped."""
        super().__init__(ctx)
        from fastforks.shared import mlx_sam  # noqa: PLC0415 - imports mlx

        enc, dec = self.sam.image_encoder, self.sam.mask_decoder
        self.mlx = mlx_sam.MlxSamEncoder(enc, self.rel, mlx_wrap.run_dtype(ctx), hq=dec)
        del self.sam.image_encoder
        self.rel = []

    def _enc(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - imports mlx

        embedding, hq = self.mlx(to_mx(x.float().permute(0, 2, 3, 1)))
        return to_torch(embedding).permute(0, 3, 1, 2).contiguous(), to_torch(hq).permute(0, 3, 1, 2).contiguous()


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(_MlxPortable, attrs=()),
}
