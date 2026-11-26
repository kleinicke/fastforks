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

from fastforks._core.upload import device_tensor
from fastforks.kernels.dpt_fused import load as load_dpt_fused
from fastforks.promptable_segmentation._runner import Forward, Runner
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import GraphCache
from fastforks.shared.sam_prompt import decode

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
GRAPH_PROMPTS = 4  # fixed: as Mobile-SAM (same prompt side: a graph pays off up to 4 prompts)
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
    cuda_graphs: bool  # kernels: replay CUDA graphs of the prompt decode (option)


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
        xy = device_tensor([nw / w, nh / h], device=f64, dtype=torch.float64)
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
        """Upstream `PromptEncoder.forward` + `MaskDecoder.predict_masks`: all four masks and scores (1024 frame)."""
        pe = self.sam.prompt_encoder
        sparse, dense = pe(points=points, boxes=boxes, masks=masks)
        return self.sam.mask_decoder.predict_masks(
            image_embeddings=embedding,
            image_pe=pe.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
        )

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

    def _enc(self, x: torch.Tensor) -> torch.Tensor:
        return encode(self.sam.image_encoder, x, self.rel)

    def embed(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Image embedding `[1, 256, 64, 64]` float32."""
        x = prepare(image, self.ctx.io, self.ctx.params.image_size, exact=image.device.type == "cpu")
        return {"image_embeddings": self._enc(x)}

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
    """dpt_fused kernels: fused-epilogue CUTLASS GEMMs (fp16 accumulation) and the neck conv; SDPA flash attention.

    The prompt encoder and mask decoder are launch-bound (~200 small kernels: 6 ms per call for 1.3 ms of GPU work on
    an RTX 5060 Ti under Windows), so up to `GRAPH_PROMPTS` prompts of a repeated prompt layout replay a CUDA graph
    (`fastforks.shared.cuda_graphs`; `release_graphs()` frees its pool). The graph runs `sam_prompt.decode`: the same
    values as upstream's forward without its host read-backs.
    """

    def __init__(self, ctx: Context) -> None:
        """Build on the CPU, convert the encoder, move the float32 prompt side to the device."""
        super().__init__(ctx)
        sam = _sam(ctx)
        self.encoder = FusedEncoder(sam.image_encoder, ctx.device, load_dpt_fused(ctx.device))
        del sam.image_encoder
        self.sam = sam.to(ctx.device)
        self.cache = (
            GraphCache(self._graph_forward, graphs=GRAPHS, share_inputs=(0,)) if ctx.params.cuda_graphs else None
        )

    def embed(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Image embedding `[1, 256, 64, 64]` float32."""
        x = prepare(image, self.ctx.io, self.ctx.params.image_size, exact=False)
        return {"image_embeddings": self.encoder(x)}

    def _decode(
        self,
        embedding: torch.Tensor,
        points: tuple[torch.Tensor, torch.Tensor] | None,
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """As `_Sam._decode`, from a CUDA graph per prompt layout (prompts, points per prompt, box, mask)."""
        given = [t for t in (*(points or ()), boxes, masks) if t is not None]
        if self.cache is None or not given or len(given[0]) > GRAPH_PROMPTS:
            return super()._decode(embedding, points, boxes, masks)
        key = (len(given[0]), points[0].shape[1] if points else 0, boxes is not None, masks is not None)
        out = self.cache.run(key, embedding, *given)
        return super()._decode(embedding, points, boxes, masks) if out is None else out

    def _graph_forward(self, embedding: torch.Tensor, *given: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`sam_prompt.decode` on the cache's static inputs; a prompt kind is told by its rank (labels: integers)."""
        by = {(t.ndim, t.is_floating_point()): t for t in given}
        points, labels, boxes, masks = (by.get(k) for k in ((3, True), (2, False), (2, True), (4, True)))
        pe, dec = self.sam.prompt_encoder, self.sam.mask_decoder
        return decode(pe, dec, embedding, points=points, labels=labels, boxes=boxes, masks=masks)  # ty: ignore[invalid-argument-type] - upstream's modules, as sam_prompt's

    def release_graphs(self) -> None:
        """Free the prompt-decode CUDA graphs and their memory pool (captured again when needed)."""
        if self.cache is not None:
            self.cache.release()


class _MlxPortable(Portable):
    """Apple MLX: the image encoder written in MLX operations (`shared.mlx_sam`), the prompt side as portable."""

    module = Runner.module  # the MLX encoder is not exportable

    def __init__(self, ctx: Context) -> None:
        """The portable network; the image encoder's weights copied to MLX (and dropped from the twin)."""
        super().__init__(ctx)
        from fastforks.shared import mlx_sam  # noqa: PLC0415 - imports mlx

        self.mlx = mlx_sam.MlxSamEncoder(self.sam.image_encoder, self.rel, mlx_wrap.run_dtype(ctx))
        del self.sam.image_encoder
        self.rel = []

    def _enc(self, x: torch.Tensor) -> torch.Tensor:
        from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - imports mlx

        y = self.mlx(to_mx(x.float().permute(0, 2, 3, 1)))
        return to_torch(y).permute(0, 3, 1, 2).contiguous()


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(_MlxPortable, attrs=()),
}
# FASTFORKS_MLX_GRAPH=1 (diagnostics, tools/mlx_core_ab.py): the exported-graph port the hand-written network replaced
GRAPH_RUNNERS = {"portable_mlx": mlx_wrap.port(Portable, attrs=(), methods=("_enc",), net_attr="sam")}
