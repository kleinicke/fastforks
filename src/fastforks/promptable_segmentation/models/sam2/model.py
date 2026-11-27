"""SAM 2.1 image runners: weights -> network, and the portable / kernels image encoders (shared float32 prompt side).

Both follow upstream `SAM2ImagePredictor` (`build_sam2` with its default post-processing settings): `set_image`
(resize to a square, normalise, encode, project the high-resolution levels, add `no_mem_embed`) and `predict` (prompt
coordinates normalised by the image size and scaled to the square; a box becomes two corner points with labels 2 / 3
in front of the points; prompt encoder + mask decoder in float32; masks bilinearly upsampled to the input size;
returned low-resolution logits clamped to +-32). The single-mask output falls back to the best multimask output when
it is unstable (upstream `dynamic_multimask_via_stability`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks._core.deferred_init import skipped_init
from fastforks._core.preprocess import IMAGENET_MEAN, IMAGENET_STD
from fastforks._core.runner import host
from fastforks.kernels.dpt_fused import load as load_dpt_fused
from fastforks.promptable_segmentation._runner import Forward, Runner
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import GraphCache

from .fast import Neck, neck, pos_embed, prepare, trunk
from .fused import FusedEncoder
from .net.hieradet import Hiera
from .net.image_encoder import FpnNeck, ImageEncoder
from .net.mask_decoder import MaskDecoder
from .net.position_encoding import PositionEmbeddingSine
from .net.prompt_encoder import PromptEncoder
from .net.transformer import TwoWayTransformer
from .video import VIDEO_KEYS, HostCopies, Tracker, VideoModules, prefetch, resize_frame

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks.promptable_segmentation._runner import Context

    from .mlx_video import MlxTracker

PROMPT_DIM = 256  # fixed: SAM 2's FPN / prompt / mask-decoder width (all released checkpoints)
PATCH_STRIDE = 4  # fixed: Hiera's patch-embedding stride (all released checkpoints)
STRIDE = 16  # fixed: SAM 2's image-embedding stride (all released checkpoints)
STABILITY = (0.05, 0.98)  # fixed: upstream build_sam2 post-processing (stability delta, threshold)
LOGIT_CLAMP = 32.0  # fixed: upstream SAM2ImagePredictor clamps the returned low-resolution logits
GRAPH_PROMPTS = 4  # fixed: kernels replays the prompt decode of up to this many prompts (as HQ-SAM / EfficientViT-SAM)
GRAPHS = 4  # fixed: prompt layouts with a decode graph at a time (points, a box, both, with a mask)
# The video-memory modules of the checkpoint (image mode does not use them); every other tensor must fit.
VIDEO = (*VIDEO_KEYS, "no_mem_pos_enc")  # `video` loads VIDEO_KEYS on first use; no_mem_pos_enc is never used


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params`."""

    image_size: int  # square encoder input side
    embed_dim: int  # Hiera stage-1 width (doubles per stage)
    num_heads: int  # Hiera stage-1 heads (double per stage)
    stages: list[int]  # blocks per Hiera stage
    global_att_blocks: list[int]  # blocks with global attention
    window_spec: list[int]  # attention window per stage
    pos_embed_size: list[int]  # background position-embedding grid (upstream window_pos_embed_bkg_spatial_size)
    kf_score_weight: float  # SAMURAI motion weight
    samurai: bool  # option: `video` as SAMURAI
    cuda_graphs: bool  # replay the prompt decoder on CUDA
    cuda_graphs_encoder: bool  # also replay the image encoder when CUDA graphs are enabled


class Sam2Image(torch.nn.Module):
    """The image-mode modules of upstream `SAM2Base` (same state-dict names)."""

    def __init__(self, p: Params) -> None:
        """Upstream `build_sam2` model config for `p`, image part (untrained)."""
        super().__init__()
        hiera = Hiera(
            embed_dim=p.embed_dim,
            num_heads=p.num_heads,
            stages=tuple(p.stages),
            global_att_blocks=tuple(p.global_att_blocks),
            window_spec=tuple(p.window_spec),
            window_pos_embed_bkg_spatial_size=(p.pos_embed_size[0], p.pos_embed_size[1]),
        )
        self.image_encoder = ImageEncoder(
            trunk=hiera,
            neck=FpnNeck(
                position_encoding=PositionEmbeddingSine(num_pos_feats=PROMPT_DIM, normalize=True, temperature=10000),
                d_model=PROMPT_DIM,
                backbone_channel_list=[p.embed_dim * 2**i for i in reversed(range(len(p.stages)))],
                fpn_top_down_levels=[2, 3],
                fpn_interp_model="nearest",
            ),
            scalp=1,
        )
        tokens = p.image_size // STRIDE
        # On the CPU also in a meta-device skeleton (the MLX cache replay): its `torch.randn` has no meta kernel and
        # goes through the decomposition that imports `torch._dynamo` (about 1 s of a load).
        with torch.device("cpu"):
            self.sam_prompt_encoder = PromptEncoder(
                embed_dim=PROMPT_DIM,
                image_embedding_size=(tokens, tokens),
                input_image_size=(p.image_size, p.image_size),
                mask_in_chans=16,
            )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embedding_dim=PROMPT_DIM, mlp_dim=2048, num_heads=8),
            transformer_dim=PROMPT_DIM,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=True,
            iou_prediction_use_sigmoid=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,
            dynamic_multimask_via_stability=True,
            dynamic_multimask_stability_delta=STABILITY[0],
            dynamic_multimask_stability_thresh=STABILITY[1],
        )
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, PROMPT_DIM))


def _net(ctx: Context) -> Sam2Image:
    net = Sam2Image(ctx.params)
    load_strict(net, ctx.state_dict(key="model"), f"{ctx.variant.ref} checkpoint", unused=VIDEO)
    return net.eval()


class _Sam2(Runner):
    """The prompt side, shared by both backends (float32, upstream modules)."""

    batch = 1
    Params = Params
    low_res_clamp = LOGIT_CLAMP
    video = True
    net: Sam2Image
    no_mem: torch.Tensor  # [256] float32 on the device
    _video: Tracker | None = None

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
        """Upstream `SAM2ImagePredictor._prep_prompts` + `_predict` (logits before the clamp)."""
        pts = self._points(size, points, labels, boxes)
        low, iou = self._decode(features, pts, masks, multimask=multimask)
        return {"pred_masks": low, "iou_scores": iou}

    def _points(
        self,
        size: tuple[int, int],
        points: torch.Tensor | None,
        labels: torch.Tensor | None,
        boxes: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Upstream `_prep_prompts`: the box corners (labels 2 / 3) then the points, in encoder-input pixels.

        No host copy: `[w, h]` is filled on the device (`torch.tensor(..., device=cuda)` waited for the encoder, so
        the prompt side could not be launched while the encoder ran).
        """
        (h, w), s, dev = size, self.ctx.params.image_size, self.ctx.device
        wh = torch.full((2,), float(w), device=dev)
        wh[1:].fill_(h)  # `wh[1] = h` copies a host scalar (a sync)
        coords, labs = [], []
        if boxes is not None:
            coords.append(boxes.float().view(-1, 2, 2) / wh * s)
            labs.append(torch.arange(2, 4, device=dev)[None].expand(boxes.shape[0], 2))
        if points is not None and labels is not None:
            coords.append(points.float() / wh * s)
            labs.append(labels)
        return (torch.cat(coords, 1), torch.cat(labs, 1)) if coords else None

    def _decode(
        self,
        features: dict[str, torch.Tensor],
        pts: tuple[torch.Tensor, torch.Tensor] | None,
        masks: torch.Tensor | None,
        *,
        multimask: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prompt encoder + mask decoder: low-resolution logits `[P, M, 256, 256]` and scores `[P, M]`."""
        pe = self.net.sam_prompt_encoder
        sparse, dense = pe(points=pts, boxes=None, masks=masks)
        low, iou, _, _ = self.net.sam_mask_decoder(
            image_embeddings=features["image_embed"],
            image_pe=pe.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask,
            repeat_image=sparse.shape[0] > 1,
            high_res_features=[features["feat_s0"], features["feat_s1"]],
        )
        return low, iou

    def upsample(self, pred_masks: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Upstream `SAM2Transforms.postprocess_masks` (no hole / sprinkle removal, as `SAM2ImagePredictor`)."""
        return F.interpolate(pred_masks, size, mode="bilinear", align_corners=False)

    def _tracker(self) -> Tracker:
        """The video modules, loaded on the first `video` call (image use does not hold them)."""
        if self._video is None:
            ctx = self.ctx
            mods = VideoModules.load(
                ctx.state_dict(key="model"), ctx.params.image_size // STRIDE, f"{ctx.variant.ref} checkpoint"
            ).to(ctx.device)
            mods.memory_attention.to(ctx.torch_dtype)
            mods.memory_encoder.to(ctx.torch_dtype)
            dec = self.net.sam_mask_decoder
            self._video = Tracker(
                mods, self.net.sam_prompt_encoder, dec, self.no_mem, ctx.torch_dtype, ctx.params.image_size
            )
        return self._video

    @host
    def track(  # noqa: PLR0913 - one argument per prompt kind
        self,
        frames: Callable[[int], torch.Tensor],
        n_frames: int,
        size: tuple[int, int],
        *,
        points: torch.Tensor | None,
        labels: torch.Tensor | None,
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Upstream `SAM2VideoPredictor`: frame 0 prompts, then `propagate_in_video` (`video.py`; option `samurai`)."""
        tracker, s = self._tracker(), self.ctx.params.image_size
        coords = labs = mask = None
        if masks is not None:
            mask = masks.float()[:, None]
            if mask.shape[-2:] != (s, s):
                mask = F.interpolate(mask, (s, s), mode="bilinear", align_corners=False, antialias=True)
                mask = (mask >= 0.5).float()  # noqa: PLR2004 - upstream add_new_mask
            p = mask.shape[0]
        else:
            pts = self._points(size, points, labels, boxes)
            if pts is None:
                msg = "SAM 2 video: no point / box prompt"
                raise ValueError(msg)
            coords, labs = pts[0], pts[1].int()
            p = coords.shape[0]
        # On the CPU, filled frame by frame (as the original's offloaded state): device memory stays bounded by the
        # memory bank instead of growing with the video length x resolution.
        out = {
            "masks": torch.empty(n_frames, p, *size, dtype=torch.bool),
            "pred_masks": torch.empty(n_frames, p, *self.mask_size),
            "object_score_logits": torch.empty(n_frames, p),
        }

        host = HostCopies(out)  # no host sync per frame

        def emit(t: int, low: torch.Tensor, obj: torch.Tensor) -> None:
            host.put(t, masks=self.upsample(low, size)[:, 0] > 0, pred_masks=low[:, 0], object_score_logits=obj[:, 0])

        feats = (self.embed(resize_frame(x, s)) for x in prefetch(frames, n_frames))
        kf = self.ctx.params.kf_score_weight if self.ctx.params.samurai else None
        tracker.run(feats, n_frames, coords=coords, labels=labs, mask=mask, emit=emit, kf_weight=kf)
        host.flush()
        return out


class Portable(_Sam2):
    """Plain PyTorch: upstream Hiera blocks, the folded neck; float16 image encoder on CUDA."""

    dtypes = ("float32", "float16")

    def __init__(self, ctx: Context) -> None:
        """Build the network; the image encoder in the compute dtype."""
        super().__init__(ctx)
        net = _net(ctx)
        n = Neck.build(net.image_encoder, net.sam_mask_decoder, net.no_mem_embed)
        t = net.image_encoder.trunk
        self.pos = pos_embed(t, ctx.params.image_size // PATCH_STRIDE).to(ctx.device, ctx.torch_dtype)
        self.trunk = t.to(ctx.device, ctx.torch_dtype)
        self.neck = n.to(ctx.device, ctx.torch_dtype)
        self.no_mem = self.neck.no_mem
        del net.image_encoder
        self.net = net.to(ctx.device)

    def _encode(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x = prepare(image, self.ctx.io, self.ctx.params.image_size)
        return neck(self.neck, trunk(self.trunk, x, self.pos))

    def embed(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Image embedding `[1, 256, 64, 64]` and the projected high-resolution maps, float32."""
        return self._encode(image)

    def module(self) -> torch.nn.Module:
        """The image encoder (no prompt side): `[1, 3, H, W]` in `[0, 1]` -> `image_embeddings` `[1, 256, 64, 64]`."""
        return Forward(
            self.trunk, lambda _net, image: self._encode(image)["image_embed"], output_names=("image_embeddings",)
        )

    def networks(self) -> tuple[torch.nn.Module, ...]:
        """The trunk and the prompt side (the mask decoder's transposed convs); the folded neck holds no modules."""
        return (self.trunk, self.net)


class Kernels(_Sam2):
    """dpt_fused kernels: fused-epilogue CUTLASS GEMMs (fp16 accumulation); SDPA flash attention.

    With `cuda_graphs` the decoder replays one CUDA graph per prompt layout up to `GRAPH_PROMPTS` prompts.
    `cuda_graphs_encoder` also enables an encoder graph, for workloads that benefit from keeping its workspace.
    """

    def __init__(self, ctx: Context) -> None:
        """Build on the CPU, convert the encoder, move the float32 prompt side to the device."""
        super().__init__(ctx)
        net = _net(ctx)
        n = Neck.build(net.image_encoder, net.sam_mask_decoder, net.no_mem_embed)
        k = load_dpt_fused(ctx.device)
        self.encoder = FusedEncoder(net.image_encoder, n, ctx.params.image_size, ctx.device, k)
        self.no_mem = n.no_mem.to(ctx.device)
        del net.image_encoder
        self.net = net.to(ctx.device)
        on = ctx.params.cuda_graphs
        self.encoder_graph = GraphCache(self._encode, graphs=1) if on and ctx.params.cuda_graphs_encoder else None
        self.decode_graphs = GraphCache(self._graph_decode, graphs=GRAPHS, share_inputs=(0, 1, 2)) if on else None

    def embed(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Image embedding `[1, 256, 64, 64]` and the projected high-resolution maps, float32."""
        x = prepare(image, self.ctx.io, self.ctx.params.image_size)
        out = None if self.encoder_graph is None or not x.is_cuda else self.encoder_graph.run((x.device,), x)
        return self.encoder(x) if out is None else out

    def _encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encoder(x)

    def _decode(
        self,
        features: dict[str, torch.Tensor],
        pts: tuple[torch.Tensor, torch.Tensor] | None,
        masks: torch.Tensor | None,
        *,
        multimask: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """As `_Sam2._decode`, from a CUDA graph per prompt layout (prompts, points per prompt, mask)."""
        given = [t for t in (*(pts or ()), masks) if t is not None]
        graphs = self.decode_graphs
        if graphs is None or not given or len(given[0]) > GRAPH_PROMPTS or not given[0].is_cuda:
            return super()._decode(features, pts, masks, multimask=multimask)
        key = tuple((t.dtype, tuple(t.shape)) for t in given)
        inputs = (features["image_embed"], features["feat_s0"], features["feat_s1"], *given)
        out = graphs.run(key, *inputs, post=lambda o: tuple(t.clone() for t in (o[:2] if multimask else o[2:])))
        return super()._decode(features, pts, masks, multimask=multimask) if out is None else out

    def _graph_decode(
        self, embed: torch.Tensor, s0: torch.Tensor, s1: torch.Tensor, *given: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """`_Sam2._decode` on the cache's static inputs, both outputs: multimask logits + scores, then single.

        The mask decoder's own `forward` with `multimask_output` True / False (upstream `MaskDecoder.forward`: tokens
        1-3, or token 0 with `dynamic_multimask_via_stability`), from one `predict_masks`. Inputs: points `[P, N, 2]`
        + labels `[P, N]`, and / or a mask `[P, 1, 256, 256]` (told apart by rank).
        """
        by = {t.ndim: t for t in given}
        pts = (by[3], by[2]) if 3 in by else None  # noqa: PLR2004 - coordinates [P, N, 2]
        pe, dec = self.net.sam_prompt_encoder, self.net.sam_mask_decoder
        sparse, dense = pe(points=pts, boxes=None, masks=by.get(4))
        # upstream annotates two outputs; it returns masks, scores, mask tokens and object scores
        low, iou = dec.predict_masks(
            image_embeddings=embed,
            image_pe=pe.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            repeat_image=sparse.shape[0] > 1,
            high_res_features=[s0, s1],
        )[:2]
        single = dec._dynamic_multimask_via_stability(low, iou)  # noqa: SLF001 - upstream forward's single-mask choice
        return low[:, 1:], iou[:, 1:], *single

    def release_graphs(self) -> None:
        """Free the encoder and prompt-decode CUDA graphs and their memory pools (they are captured again)."""
        for graphs in (self.encoder_graph, self.decode_graphs):
            if graphs is not None:
                graphs.release()


def _replay_shapes(net: Sam2Image, tokens: int) -> tuple[Neck, torch.Tensor]:
    """The shapes of `fast.Neck.build` and `fast.pos_embed` for a cache replay's meta skeleton, without computing.

    A matmul, a sum or an interpolation of meta tensors runs PyTorch's Python decomposition, which imports
    `torch._dynamo` (about 1 s of a load); `torch.empty` does not.
    """

    def empty(*shape: int) -> torch.Tensor:
        return torch.empty(shape, device="meta")

    def lin(conv: Any, last: Any = None) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN401 - `Conv2d`
        out = int((last or conv).out_channels)
        return empty(out, int(conv.in_channels)), empty(out)

    enc: Any = net.image_encoder
    c, dec = enc.neck.convs, net.sam_mask_decoder
    dim = int(enc.trunk.pos_embed.shape[1])
    neck = Neck(
        lin(c[0].conv),
        lin(c[1].conv),
        lin(c[2].conv, dec.conv_s1),
        lin(c[3].conv, dec.conv_s0),
        empty(net.no_mem_embed.numel()),
    )
    return neck, empty(1, tokens, tokens, dim)


class _MlxPortable(Portable):
    """Apple MLX: the image encoder (`mlx_net`), the prompt side (`mlx_heads`) and the video tracker (`mlx_video`).

    Nothing computes in PyTorch: the weights come from the conversion cache (no PyTorch network is built on a cache
    hit), the prompts are prepared in numpy. Only the task's tensors in and out are PyTorch's.
    """

    module = Runner.module  # the MLX encoder is not exportable

    def __init__(self, ctx: Context) -> None:
        """The MLX networks from the conversion cache (the first load converts the checkpoint's weights)."""
        Runner.__init__(self, ctx)
        from fastforks.shared import mlx_port  # noqa: PLC0415 - imports mlx

        from . import mlx_heads, mlx_net  # noqa: PLC0415 - imports mlx

        self.twins = mlx_net.Twins()

        def make(net: Sam2Image) -> tuple[mlx_net.MlxEncoder, mlx_heads.MlxPrompts, mlx_heads.MlxDecoder]:
            t, tokens = net.image_encoder.trunk, ctx.params.image_size // PATCH_STRIDE
            if t.pos_embed.is_meta:  # a replay reads shapes only
                n, pos = _replay_shapes(net, tokens)
            else:
                n = Neck.build(net.image_encoder, net.sam_mask_decoder, net.no_mem_embed)
                pos = pos_embed(t, tokens)
            encoder = mlx_net.MlxEncoder(t, n, pos, ctx.dtype, twins=self.twins)
            prompts = mlx_heads.MlxPrompts(net.sam_prompt_encoder)
            return encoder, prompts, mlx_heads.MlxDecoder(net.sam_mask_decoder, prompts.dense_pe)

        self.mlx, self.mlx_prompts, self.mlx_decoder = mlx_port.recorded(
            ctx,
            mlx_heads.LAYOUT,
            make,
            real=lambda: _net(ctx),
            skeleton=lambda: Sam2Image(ctx.params),
            sources=mlx_heads.SOURCES,
        )
        self.net = None  # ty: ignore[invalid-assignment] - no PyTorch network on MLX
        self._mlx_video: MlxTracker | None = None

    def _encode(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.mlx(prepare(image, self.ctx.io, self.ctx.params.image_size))

    def _prompts(
        self,
        size: tuple[int, int],
        points: torch.Tensor | None,
        labels: torch.Tensor | None,
        boxes: torch.Tensor | None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """`_Sam2._points` in numpy: the box corners (labels 2 / 3) then the points, in encoder-input pixels."""
        (h, w), s = size, self.ctx.params.image_size
        wh = np.array([w, h], dtype=np.float32)
        coords, labs = [], []
        if boxes is not None:
            b = boxes.float().cpu().numpy().reshape(-1, 2, 2)
            coords.append(b / wh * s)
            labs.append(np.broadcast_to(np.array([[2, 3]]), (len(b), 2)))
        if points is not None and labels is not None:
            coords.append(points.float().cpu().numpy() / wh * s)
            labs.append(labels.cpu().numpy())
        return (np.concatenate(coords, 1), np.concatenate(labs, 1).astype(np.int32)) if coords else None

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
        """As `_Sam2.predict`: the prompt encoder and the mask decoder on MLX, one graph."""
        from fastforks._core.mlx import mx, to_torch  # noqa: PLC0415 - imports mlx

        pts, a = self._prompts(size, points, labels, boxes), self.twins.array
        p = pts[0].shape[0] if pts is not None else 1 if masks is None else masks.shape[0]  # upstream's batch
        sparse = None if pts is None else self.mlx_prompts.sparse(*pts)
        dense = self.mlx_prompts.no_mask
        if masks is not None:
            dense = self.mlx_prompts.dense(mx.array(masks.float().cpu().numpy()[:, 0]))
        embed, s0, s1 = a(features["image_embed"]), a(features["feat_s0"]), a(features["feat_s1"])
        low, iou = self.mlx_decoder.predict(embed, dense, sparse, s0, s1, p, multimask=multimask)
        mx.eval(low, iou)
        return {"pred_masks": to_torch(low), "iou_scores": to_torch(iou)}

    def _mlx_tracker(self) -> MlxTracker:
        """The video modules on MLX, built on the first `video` call (from the conversion cache, as the image side)."""
        if self._mlx_video is None:
            from fastforks.shared import mlx_port  # noqa: PLC0415 - imports mlx

            from . import mlx_heads, mlx_video  # noqa: PLC0415 - imports mlx

            ctx = self.ctx
            tokens = ctx.params.image_size // STRIDE

            def skeleton() -> VideoModules:
                # Unfilled CPU tensors (6 M parameters), not the meta device: the vendored modules' `torch.arange`
                # and products there, and the layer-scale fold, would import `torch._dynamo` (about 1 s).
                with torch.device("cpu"), skipped_init():
                    return VideoModules(tokens)

            self._mlx_video = mlx_port.recorded(
                ctx,
                mlx_heads.VIDEO_LAYOUT,
                lambda mods: mlx_video.MlxTracker(
                    mods, self.mlx, self.mlx_prompts, self.mlx_decoder, ctx.dtype, ctx.params.image_size
                ),
                real=lambda: VideoModules.load(ctx.state_dict(key="model"), tokens, f"{ctx.variant.ref} checkpoint"),
                skeleton=skeleton,
                sources=mlx_heads.SOURCES,
            )
        return self._mlx_video

    @host
    def track(  # noqa: PLR0913 - one argument per prompt kind
        self,
        frames: Callable[[int], torch.Tensor],
        n_frames: int,
        size: tuple[int, int],
        *,
        points: torch.Tensor | None,
        labels: torch.Tensor | None,
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """As `_Sam2.track`, the whole loop on MLX (`mlx_video.MlxTracker.track`); the frames are prepared in numpy."""
        from fastforks._core.mlx import mx  # noqa: PLC0415 - imports mlx

        from .mlx_video import frame  # noqa: PLC0415 - imports mlx

        tracker, ctx = self._mlx_tracker(), self.ctx
        s, io = ctx.params.image_size, ctx.io
        coords = labs = mask = None
        if masks is not None:
            mask = tracker.prompt_mask(mx.array(masks.float().cpu().numpy()))
            p = mask.shape[0]
        else:
            pts = self._prompts(size, points, labels, boxes)
            if pts is None:
                msg = "SAM 2 video: no point / box prompt"
                raise ValueError(msg)
            coords, labs = pts
            p = coords.shape[0]
        mean, std = (IMAGENET_MEAN, IMAGENET_STD) if io.input == "imagenet" else (io.mean, io.std)
        feed = prefetch(lambda t: frame(frames(t).float().cpu().numpy(), s, mean, std), n_frames)
        kf = ctx.params.kf_score_weight if ctx.params.samurai else None
        out = tracker.track(feed, n_frames, size, p, coords=coords, labels=labs, mask=mask, kf_weight=kf)
        return {k: torch.from_numpy(v) for k, v in out.items()}

    def networks(self) -> tuple[torch.nn.Module, ...]:
        """No PyTorch network: every weight lives on MLX."""
        return ()


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": _MlxPortable,
}
# FASTFORKS_MLX_GRAPH=1 (diagnostics, tools/mlx_core_ab.py): the exported-graph port the hand-written network replaced
GRAPH_RUNNERS = {"portable_mlx": mlx_wrap.port(Portable, attrs=(), methods=("embed",), net_attr="trunk")}
