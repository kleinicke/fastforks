"""SAM runners: weights -> network, and the portable / kernels image encoders (shared float32 prompt side).

Both follow upstream `SamPredictor`: `set_image` (resize the longest side to `image_size`, normalise, pad, encode)
and `predict_torch` (prompt coordinates scaled like the image, prompt encoder + mask decoder in float32, mask
`postprocess_masks`: upsample to the padded square, crop, upsample to the input size).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks.kernels.dpt_fused import load as load_dpt_fused
from fastforks.promptable_segmentation._runner import Forward, Runner

from .fast import encode, prepare, preprocess_shape, rel_tables
from .fused import FusedEncoder
from .net.image_encoder import ImageEncoderViT
from .net.mask_decoder import MaskDecoder
from .net.prompt_encoder import PromptEncoder
from .net.sam import Sam
from .net.transformer import TwoWayTransformer

if TYPE_CHECKING:
    from fastforks.promptable_segmentation._runner import Context

PATCH = 16  # fixed: SAM's ViT patch size (all released checkpoints)
PROMPT_DIM = 256  # fixed: SAM's prompt / mask-decoder embedding width (all released checkpoints)


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params`."""

    image_size: int  # encoder input side (longest image side, padded to a square)
    window_size: int  # windowed-attention block size
    global_attn: int  # number of global-attention blocks, evenly spaced (upstream global_attn_indexes)
    embed_dim: int
    depth: int
    num_heads: int


def build(p: Params) -> Sam:
    """Upstream `build_sam._build_sam` for `p` (untrained)."""
    tokens = p.image_size // PATCH
    every = p.depth // p.global_attn
    return Sam(
        image_encoder=ImageEncoderViT(
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
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=PROMPT_DIM,
            image_embedding_size=(tokens, tokens),
            input_image_size=(p.image_size, p.image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embedding_dim=PROMPT_DIM, mlp_dim=2048, num_heads=8),
            transformer_dim=PROMPT_DIM,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )


def _sam(ctx: Context) -> Sam:
    sam = build(ctx.params)
    sam.load_state_dict(ctx.state_dict())
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
        xy = torch.tensor([nw / w, nh / h], device=f64, dtype=torch.float64)
        pts = None if points is None or labels is None else ((points.to(f64).double() * xy).float().to(dev), labels)
        box = None if boxes is None else (boxes.view(-1, 2, 2).to(f64).double() * xy).float().view(-1, 4).to(dev)
        pe = self.sam.prompt_encoder
        sparse, dense = pe(points=pts, boxes=box, masks=masks)
        low, iou = self.sam.mask_decoder(
            image_embeddings=features["image_embeddings"],
            image_pe=pe.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask,
        )
        return {"pred_masks": low, "iou_scores": iou}

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
        self.sam = _sam(ctx).to(ctx.device)
        enc = self.sam.image_encoder.to(ctx.torch_dtype)
        tokens = ctx.params.image_size // PATCH
        self.rel = [rel_tables(b, tokens, ctx.torch_dtype) for b in enc.blocks]

    def _encode(self, image: torch.Tensor, *, exact: bool) -> torch.Tensor:
        x = prepare(image, self.ctx.io, self.ctx.params.image_size, exact=exact)
        return encode(self.sam.image_encoder, x, self.rel)

    def embed(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Image embedding `[1, 256, 64, 64]` float32."""
        return {"image_embeddings": self._encode(image, exact=image.device.type == "cpu")}

    def module(self) -> torch.nn.Module:
        """The image encoder only (no prompt side): `[1, 3, H, W]` in `[0, 1]` -> `image_embeddings` `[1, 256, 64, 64]`.

        Torch resize, as on CUDA.
        """
        return Forward(
            self.sam.image_encoder,
            lambda _net, image: self._encode(image, exact=False),
            output_names=("image_embeddings",),
        )

    def networks(self) -> tuple[torch.nn.Module, ...]:
        """The whole model: image encoder, prompt encoder and mask decoder (its transposed convs)."""
        return (self.sam,)


class Kernels(_Sam):
    """dpt_fused kernels: fused-epilogue CUTLASS GEMMs (fp16 accumulation) and the neck conv; SDPA flash attention."""

    def __init__(self, ctx: Context) -> None:
        """Build on the CPU, convert the encoder, move the float32 prompt side to the device."""
        super().__init__(ctx)
        sam = _sam(ctx)
        self.encoder = FusedEncoder(sam.image_encoder, ctx.device, load_dpt_fused(ctx.device))
        del sam.image_encoder
        self.sam = sam.to(ctx.device)

    def embed(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Image embedding `[1, 256, 64, 64]` float32."""
        x = prepare(image, self.ctx.io, self.ctx.params.image_size, exact=False)
        return {"image_embeddings": self.encoder(x)}


RUNNERS = {"portable": Portable, "kernels": Kernels}
