"""SAM 3 runners: weights -> network, and the portable / kernels passes (image concepts; `track`: `video.py`).

Both follow upstream `Sam3Processor` / transformers `Sam3Model` (text and exemplar-box prompts) and its
post-processing:

- input: the image as uint8, `Resize((1008, 1008))` (antialiased bilinear on uint8, as torchvision), `/ 255`,
  normalised with mean / std 0.5; CPU: the uint8 resize itself (exact), CUDA: a float resize rounded to integers
- image: ViT trunk (windows of 24 x 24 tokens, 4 global layers, axial RoPE) + the neck's 4x / 2x / 1x levels
- text: CLIP byte-level BPE (pure Python, `tokenizer.py`), the real tokens through the CLIP text encoder, projected
- exemplar boxes (optional, positive or negative): normalised by the image size, `(cx, cy, w, h)`, encoded by the
  geometry encoder (ROI align on the 72 x 72 level, see `net/sam3.py`); boxes without text use the text "visual"
- prompt: the text tokens + the geometry encoder's box tokens and CLS token (always run, as upstream)
- DETR encoder on the 72 x 72 level, decoder with 200 queries + presence token, dot-product scores, mask decoder
- output (the API): score = sigmoid(logit) * sigmoid(presence), kept above `threshold` (upstream
  `confidence_threshold` 0.5), boxes scaled to the input size, mask logits bilinearly upsampled to the input size and
  thresholded at 0

Portable runs the trunk, neck, text encoder, DETR encoder, mask decoder and every attention block in float16 on CUDA;
the residual streams, norms and MLPs of the geometry encoder and DETR decoder, and the scoring, in
float32. Kernels (`fast.py`) replaces the trunk and neck by fp16-accumulate
kernels and runs the heads with fp16 accumulation.

`track` (video, frame-0 points / boxes / masks) loads the tracker on its first call: the frames go through the same
trunk (kernels: the fp16-accumulate trunk) and the tracker's own neck (kernels: the neck kernels), the tracker heads
in float32 and its memory modules in the compute dtype.
"""

from __future__ import annotations

import bisect
import threading
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks._core.errors import InputError
from fastforks.promptable_segmentation._runner import Context, Forward, NoParams, Runner, scale
from fastforks.shared.cuda_graphs import GraphCache

from .net.sam3 import (
    GRID,
    UNUSED,
    Attention,
    ClipTextModel,
    Exemplars,
    FpnLayer,
    Neck,
    Sam3,
    TextEncoder,
    VisionEncoder,
    ViT,
    grid_position_encoding,
    roi_weights,
    rope_pairs,
    rope_tables,
)
from .net.tracker import TrackerModel
from .tokenizer import ClipTokenizer
from .video import HostCopies, Tracker, prefetch

if TYPE_CHECKING:
    from collections.abc import Callable

SIZE = 1008  # fixed: SAM 3 input side (upstream Sam3Processor resolution)
PATCH = 14  # fixed: ViT patch size
WINDOW = 24  # fixed: ViT attention window (also the stored position-embedding grid, 336 / 14)
GLOBAL_LAYERS = (7, 15, 23, 31)  # fixed: ViT layers with global attention
VOCAB, CONTEXT = 49408, 32  # fixed: CLIP vocabulary and SAM 3's text context length
QUERIES = 200  # fixed: DETR queries
MASK = 288  # fixed: mask-logit side (the 4x neck level)
BUCKETS = (8, 16, CONTEXT)  # fixed: kernels pads the tokens to one of these counts (one CUDA graph each)
BOX_BUCKET = 16  # fixed: kernels pads 1-16 exemplar boxes to this count (more run without a CUDA graph)
VISUAL = "visual"  # fixed: the text of a boxes-only prompt (upstream Sam3Processor)
PREFIX = "detector_model."  # fixed: the image detector's keys in model.safetensors (the rest is the video tracker)
TRACKER = ("tracker_model.", "tracker_neck.")  # the video tracker's keys (not loaded); every other tensor must fit
TRACKER_UNUSED = ("fpn_layers.3.",)  # fixed: the tracker neck's 0.5x level (upstream drops it)


def build() -> Sam3:
    """The SAM 3 image detector (untrained; build under `torch.device("meta")` to skip initialisation)."""
    vit = ViT(dim=1024, depth=32, heads=16, mlp=4736, patch=PATCH, window=WINDOW, globals_=GLOBAL_LAYERS)
    text = ClipTextModel(vocab=VOCAB, context=CONTEXT, dim=1024, depth=24, heads=16, mlp=4096)
    return Sam3(VisionEncoder(vit, Neck(1024, 256)), TextEncoder(text))


def load_net(ctx: Context) -> Sam3:
    """`build()` with the checkpoint's detector weights (CPU, float32; the tracker's keys are skipped)."""
    sd = {k.removeprefix(PREFIX): v for k, v in ctx.state_dict(0, strip="").items()}
    with torch.device("meta"):
        net = build()
    load_strict(net, sd, f"{ctx.variant.ref} checkpoint", unused=(*TRACKER, *UNUSED), assign=True)
    return net.eval()


def load_tracker(ctx: Context) -> tuple[TrackerModel, Neck]:
    """The video tracker (`tracker_model.*`) and its neck (`tracker_neck.*`; CPU, float32)."""
    sd = ctx.state_dict(0, strip="")
    what = f"{ctx.variant.ref} checkpoint"
    with torch.device("meta"):
        tracker, neck = TrackerModel(), Neck(1024, 256)
    for mod, prefix, unused in ((tracker, TRACKER[0], ()), (neck, TRACKER[1], TRACKER_UNUSED)):
        part = {k.removeprefix(prefix): v for k, v in sd.items() if k.startswith(prefix)}
        load_strict(mod, part, what, unused=unused, assign=True)
    return tracker.eval(), neck.eval()


def half_parts(net: Sam3) -> list[torch.nn.Module]:
    """The detector parts that run in float16 on CUDA.

    The DETR encoder, the mask decoder, and the attention blocks of the geometry encoder and the DETR decoder (their
    residual streams, norms and MLPs stay float32).
    """
    attn = [m for m in (*net.geometry_encoder.modules(), *net.detr_decoder.modules()) if isinstance(m, Attention)]
    return [net.detr_encoder, net.mask_decoder, *attn]


def exemplars(boxes: torch.Tensor, labels: torch.Tensor | None, device: torch.device, pad: int = 0) -> Exemplars:
    """Exemplar boxes `[N, 4]` (normalised `x0, y0, x1, y1`, float32) and labels `[N]` (1 / 0; `None`: all 1).

    `pad >= N`: padded to `pad` boxes (zero boxes and ROI weights, label 0), marked in `valid` (`None` without `pad`).
    """
    boxes = boxes.float().cpu()
    n = boxes.shape[0]
    x0, y0, x1, y1 = boxes.unbind(-1)  # as Sam3Processor: box_xyxy_to_cxcywh
    cxcywh = torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)], dim=-1)
    lab = torch.ones(n, dtype=torch.long) if labels is None else labels.long().cpu()
    pool_y, pool_x = roi_weights(cxcywh, GRID, GRID)
    valid = None
    if pad:
        extra = pad - n
        cxcywh = F.pad(cxcywh, (0, 0, 0, extra))
        lab = F.pad(lab, (0, extra))
        pool_y, pool_x = F.pad(pool_y, (0, 0, 0, 0, 0, extra)), F.pad(pool_x, (0, 0, 0, 0, 0, extra))
        valid = (torch.arange(pad) < n).float()[None].to(device)
    return Exemplars(
        cxcywh[None].to(device), lab[None].to(device), pool_y[None].to(device), pool_x[None].to(device), valid
    )


def prepare(image: torch.Tensor, ctx: Context, *, exact: bool) -> torch.Tensor:
    """`[1, 3, H, W]` RGB in `[0, 1]` -> the normalised float32 `[1, 3, 1008, 1008]` input.

    `exact`: the uint8 resize of upstream (CPU only); else a float resize rounded to integers (within 1 / 255).
    """
    u = (image * 255).round()
    if exact:
        x = F.interpolate(u.to(torch.uint8), (SIZE, SIZE), mode="bilinear", antialias=True, align_corners=False)
    else:
        x = F.interpolate(u, (SIZE, SIZE), mode="bilinear", antialias=True, align_corners=False).round().clamp(0, 255)
    return scale(x.float() / 255, ctx.io)


class _Sam3(Runner):
    """Text and / or exemplar-box prompt -> all instances; the detector heads in float32 (shared by both backends)."""

    Params = NoParams
    prompts = frozenset({"text", "boxes"})
    mask_size = (MASK, MASK)
    compiled_methods = ("embed", "concept")  # `segment_concept` tokenizes on the host, then calls `concept`
    video = True
    video_prompts = frozenset({"points", "boxes", "masks"})  # the tracker's prompts (frame 0)
    net: Sam3
    _video: tuple[Tracker, object] | None = None  # the tracker, its neck as the backend runs it

    def __init__(self, ctx: Context) -> None:
        """Tokenizer (weight files 1, 2), the 72 x 72 position encoding."""
        super().__init__(ctx)
        files = ctx.weights()
        self.tokenizer = ClipTokenizer(files[1], files[2], CONTEXT)
        self.pos = grid_position_encoding(GRID, GRID, ctx.device)

    def text(self, ids: torch.Tensor) -> torch.Tensor:
        """Token ids `[1, T]` -> projected text features `[1, T, 256]` float32."""
        net = self.net
        h = net.text_encoder.text_model(ids)
        return net.text_projection(h.float())

    def segment_concept(
        self,
        features: dict[str, torch.Tensor],
        text: str | None,
        boxes: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Every query's prediction for a text and / or exemplar-box prompt on one image.

        Args:
            features: `embed` output.
            text: The concept; `None`: "visual" (boxes only, as upstream).
            boxes: Exemplar boxes `[N, 4]` normalised `x0, y0, x1, y1` (CPU float32), or `None`.
            labels: `[N]` 1 (positive) / 0 (negative); `None`: all positive.

        Returns:
            `scores [200]` (sigmoid(logit) * sigmoid(presence)), `boxes [200, 4]` (normalised `x0, y0, x1, y1`),
            `pred_masks [200, 288, 288]` mask logits, all float32.
        """
        dev = self.ctx.device
        ids = torch.tensor([self.tokenizer(VISUAL if text is None else text)], device=dev)
        ex = None if boxes is None or not len(boxes) else exemplars(boxes, labels, dev)
        return self.concept(ids, [features["fpn0"], features["fpn1"], features["fpn2"]], ex=ex)

    def concept(
        self,
        ids: torch.Tensor,
        fpn: list[torch.Tensor],
        valid: torch.Tensor | None = None,
        ex: Exemplars | None = None,
    ) -> dict[str, torch.Tensor]:
        """`segment_concept` on token ids `[1, T]` and the neck levels; `valid [1, T]` float marks padding with 0."""
        logits, presence, boxes, masks = self.net.detect(fpn, self.pos, self.text(ids), valid, ex)
        return {"scores": (logits.sigmoid() * presence.sigmoid())[0], "boxes": boxes[0], "pred_masks": masks[0]}

    def predict(  # noqa: PLR0913 - the interface's signature
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
        """Not used: SAM 3 takes concept prompts (the core sends text and boxes to `segment_concept`).

        Raises:
            InputError: Always.
        """
        del features, size, points, labels, boxes, masks, multimask
        msg = "SAM 3 takes concept prompts (text=... and / or exemplar input_boxes=...)"
        raise InputError(msg)

    def upsample(self, pred_masks: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Mask logits `[P, M, 288, 288]` -> `[P, M, H, W]` (bilinear, as upstream post-processing)."""
        return F.interpolate(pred_masks, size, mode="bilinear", align_corners=False)

    def _tracker(self) -> tuple[Tracker, object]:
        """The tracker and its neck, loaded on the first `track` call (image use does not hold them).

        The heads run in float32, the memory modules and the neck in the compute dtype.
        """
        if self._video is None:
            ctx = self.ctx
            m, neck = load_tracker(ctx)
            for mod in (m.memory_attention, m.memory_encoder):
                mod.to(ctx.torch_dtype)
            m = m.to(ctx.device)
            self._video = (Tracker(m, ctx.torch_dtype), self._track_neck(neck))
        return self._video

    def _track_neck(self, neck: Neck) -> object:
        """The tracker neck as the backend runs it."""
        raise NotImplementedError

    def _track_embed(self, image: torch.Tensor, neck: object) -> list[torch.Tensor]:
        """Frame `[1, 3, H, W]` -> the tracker neck's levels 288 / 144 / 72."""
        raise NotImplementedError

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
        """Transformers `Sam3TrackerVideoModel`: frame 0 prompts, then forward propagation (`video.py`)."""
        (tracker, neck), dev = self._tracker(), self.ctx.device
        coords = labs = mask = None
        if masks is not None:
            mask = masks.float()[:, None]
            if mask.shape[-2:] != (SIZE, SIZE):
                mask = F.interpolate(mask, (SIZE, SIZE), mode="bilinear", align_corners=False, antialias=True)
                mask = (mask >= 0.5).float()  # noqa: PLR2004 - upstream processor
            p = mask.shape[0]
        else:
            (h, w) = size
            wh = torch.tensor([w, h], device=dev, dtype=torch.float32)
            cs, ls = [], []
            if boxes is not None:
                cs.append(boxes.float().view(-1, 2, 2) / wh * SIZE)
                ls.append(torch.tensor([[2, 3]], device=dev).expand(boxes.shape[0], 2))
            if points is not None and labels is not None:
                cs.append(points.float() / wh * SIZE)
                ls.append(labels)
            coords, labs = torch.cat(cs, 1), torch.cat(ls, 1).int()
            p = coords.shape[0]
        out = {
            "masks": torch.empty(n_frames, p, *size, dtype=torch.bool),
            "pred_masks": torch.empty(n_frames, p, *self.mask_size),
            "object_score_logits": torch.empty(n_frames, p),
        }
        host = HostCopies(out)  # no host sync per frame
        md = tracker.m.mask_decoder

        def emit(t: int, low: torch.Tensor, obj: torch.Tensor) -> None:
            host.put(t, masks=self.upsample(low, size)[:, 0] > 0, pred_masks=low[:, 0], object_score_logits=obj[:, 0])

        def feats(x: torch.Tensor) -> dict[str, torch.Tensor]:
            l0, l1, l2 = (f.float() for f in self._track_embed(x, neck))
            return {"raw": l2, "feat_s0": md.conv_s0(l0), "feat_s1": md.conv_s1(l1)}

        tracker.run(
            (feats(x) for x in prefetch(frames, n_frames)), n_frames, coords=coords, labels=labs, mask=mask, emit=emit
        )
        host.flush()
        return out


class Portable(_Sam3):
    """Plain PyTorch; float16 on CUDA except parts of the geometry encoder, DETR decoder and scoring (`half_parts`)."""

    dtypes = ("float32", "float16")

    def __init__(self, ctx: Context) -> None:
        """Build the network (q / k / v packed), the RoPE tables."""
        super().__init__(ctx)
        net = load_net(ctx)
        net.vision_encoder.backbone.pack()
        for mod in (net.vision_encoder, net.text_encoder, *half_parts(net)):
            mod.to(ctx.torch_dtype)
        self.net = net.to(ctx.device)
        dt = ctx.torch_dtype
        win = rope_pairs(*rope_tables(WINDOW, 1.0, ctx.device), dt)
        self.ropes = {WINDOW: win, 0: rope_pairs(*rope_tables(GRID, WINDOW / GRID, ctx.device), dt)}

    def _encode(self, image: torch.Tensor, *, exact: bool) -> list[torch.Tensor]:
        v = self.net.vision_encoder
        x = prepare(image, self.ctx, exact=exact)
        t = v.backbone(x, self.ropes)
        return v.neck(t.permute(0, 3, 1, 2))

    def embed(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Neck levels `fpn0 [1, 256, 288, 288]`, `fpn1 [.., 144, 144]`, `fpn2 [.., 72, 72]` (compute dtype)."""
        fpn = self._encode(image, exact=image.device.type == "cpu")
        return {f"fpn{i}": f for i, f in enumerate(fpn)}

    def module(self) -> torch.nn.Module:
        """The image encoder: `[1, 3, H, W]` in `[0, 1]` -> the neck levels `fpn0`, `fpn1`, `fpn2` (as `embed`)."""
        return Forward(
            self.net.vision_encoder,
            lambda _net, image: tuple(f.float() for f in self._encode(image, exact=False)),
            output_names=("fpn0", "fpn1", "fpn2"),
        )

    def networks(self) -> tuple[torch.nn.Module, ...]:
        """The whole model (vision and text encoders, DETR decoder, mask head; the tracker once loaded)."""
        return (self.net,) if self._video is None else (self.net, self._video[0].m, cast("Neck", self._video[1]))

    def _track_neck(self, neck: Neck) -> object:
        return neck.to(self.ctx.device, self.ctx.torch_dtype)

    def _track_embed(self, image: torch.Tensor, neck: object) -> list[torch.Tensor]:
        x = prepare(image, self.ctx, exact=image.device.type == "cpu")
        return cast("Neck", neck)(self.net.vision_encoder.backbone(x, self.ropes).permute(0, 3, 1, 2))


class Kernels(_Sam3):
    """fp16-accumulate trunk and neck (see `fast.py`), the other parts as `Portable`, GEMMs with fp16 accumulation.

    The text encoder and the heads run as one CUDA graph per token bucket (`BUCKETS`: the tokens are padded after
    `eot` and the padding is masked out) and box bucket (none, or 1-16 exemplar boxes padded to `BOX_BUCKET` and
    masked out), so at most six graphs and their outputs stay allocated; more than 16 boxes run without a graph. Calls
    of one model from several threads are serialised (the graphs share static buffers), and a call on another stream
    first waits for the previous call's work on them.
    """

    def __init__(self, ctx: Context) -> None:
        """Check the kernel libraries, build on the CPU, convert the trunk and neck, move the rest to the device."""
        super().__init__(ctx)
        from fastforks.kernels import cudnn_hacc, fastbb  # noqa: PLC0415 - optional kernel libraries

        from .fast import FastEncoder, fp16_accumulate  # noqa: PLC0415 - needs the kernel libraries

        fastbb.load(ctx.device)  # BackendUnavailableError before the 3.4 GB checkpoint is read
        cudnn_hacc.load(ctx.device)
        net = load_net(ctx)
        self.encoder = FastEncoder(net.vision_encoder, ctx.device)
        self.fp16_accumulate = fp16_accumulate
        # one graph per token bucket, with / without boxes
        self.cache = GraphCache(self._graph_forward, graphs=2 * len(BUCKETS))
        self.lock = threading.Lock()
        del net.vision_encoder
        for mod in (net.text_encoder, *half_parts(net)):
            mod.half()
        self.net = net.to(ctx.device)

    def embed(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """As `Portable.embed` (float16, channels-last)."""
        fpn = self.encoder(prepare(image, self.ctx, exact=False))
        return {f"fpn{i}": f for i, f in enumerate(fpn)}

    def _track_neck(self, neck: Neck) -> object:
        layers = cast("list[FpnLayer]", list(neck.fpn_layers))
        return [self.encoder._fpn(f, self.ctx.device) for f in layers]  # noqa: SLF001 - the encoder's converter

    def _track_embed(self, image: torch.Tensor, neck: object) -> list[torch.Tensor]:
        """The fp16-accumulate trunk (shared with `embed`) and the tracker neck as the encoder's neck kernels."""
        ops = cast("list[list[tuple[str, torch.Tensor, torch.Tensor]]]", neck)
        with self.fp16_accumulate():
            t = self.encoder.trunk(prepare(image, self.ctx, exact=False)).permute(0, 3, 1, 2)
            return [self.encoder._level(o, t) for o in ops]  # noqa: SLF001 - the encoder's neck kernels

    def segment_concept(
        self,
        features: dict[str, torch.Tensor],
        text: str | None,
        boxes: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """As `_Sam3.segment_concept`, as one CUDA graph per token and box bucket, GEMMs with fp16 accumulation.

        The text encoder and the heads are launch-bound.
        """
        fpn = [features["fpn0"], features["fpn1"], features["fpn2"]]
        n = 0 if boxes is None else len(boxes)
        if n > BOX_BUCKET:  # rare: eager, no graph (locked: the fp16-accumulate switch is process-wide)
            with self.lock, self.fp16_accumulate():
                return super().segment_concept(features, text, boxes, labels)
        ids = self.tokenizer(VISUAL if text is None else text)
        bucket = BUCKETS[bisect.bisect_left(BUCKETS, len(ids))]
        padded = torch.tensor([[*ids, *[self.tokenizer.eot] * (bucket - len(ids))]])
        valid = (torch.arange(bucket) < len(ids)).float()[None]
        ex = None if boxes is None or not n else exemplars(boxes, labels, self.ctx.device, BOX_BUCKET)
        key = (bucket, BOX_BUCKET if n else 0)
        with self.lock:
            out = self.cache.run(
                key,
                padded.to(self.ctx.device),
                valid.to(self.ctx.device),
                *fpn,
                *(f for f in ex or () if f is not None),
            )
            if out is None:  # first call of the key / capture failed: eager
                with self.fp16_accumulate():
                    return super().segment_concept(features, text, boxes, labels)
            return out

    def _graph_forward(self, ids: torch.Tensor, valid: torch.Tensor, *rest: torch.Tensor) -> dict[str, torch.Tensor]:
        """`concept` on the cache's static inputs: token ids, padding mask, the three neck levels, exemplar fields."""
        with self.fp16_accumulate():
            return self.concept(ids, list(rest[:3]), valid, Exemplars(*rest[3:]) if rest[3:] else None)


RUNNERS = {"portable": Portable, "kernels": Kernels}
