"""Grounding DINO runners: transformers' `GroundingDinoForObjectDetection` network, our forward, cached text features.

As the Hugging Face `GroundingDinoImageProcessor` (transformers 5.19, torchvision backend): resize so the short side is
800 px and the long side at most 1333 (bilinear with antialiasing on uint8 values: torch's integer kernel on the CPU,
elsewhere in float and rounded, as torchvision does), ImageNet mean / std; one image, so no padding and an all-valid
pixel mask.

The queries are class names (`text`, else the card's label set) joined into one caption `"a . b . c ."` as in the
authors' COCO evaluation (`" . ".join(names) + " ."`), tokenized with BERT WordPiece (uncased); the model reads 256
tokens (a longer caption raises `InputError`). The text tower (BERT + projection) only depends on the caption, so it
runs once per query list (the last `_TEXT_CACHE` lists are kept). The image side runs every call: Swin backbone, the
fusion encoder and the decoder in the compute dtype, the last layer's class / box heads in float32. As the authors' COCO
post-processing (`PostProcessCocoGrounding`), a class score is the mean of the sigmoid token scores over the class
name's tokens; the best `max_detections` (query, class) pairs are kept if their score is above the threshold (strictly),
each a box in image pixels.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._core.errors import InputError
from fastforks._core.labels import label_set
from fastforks._core.patch import base_type, override
from fastforks._core.runner import host
from fastforks.detection._attention import deformable_sample
from fastforks.detection._runner import Forward, Runner
from fastforks.kernels import msda
from fastforks.shared import mlx_wrap
from fastforks.shared.bert_tokenizer import BertTokenizer
from fastforks.shared.seginfer import pretrained

if TYPE_CHECKING:
    from fastforks.detection._runner import Context

MEAN = (0.485, 0.456, 0.406)  # fixed: ImageNet normalisation (the image processor's)
STD = (0.229, 0.224, 0.225)  # fixed: ImageNet normalisation
SHORT, LONG = 800, 1333  # fixed: the image processor's shortest_edge / longest_edge
SEPARATORS = ("[CLS]", "[SEP]", ".", "?")  # fixed: tokens that end a phrase (the model's SPECIAL_TOKENS)
_TEXT_CACHE = 8  # fixed: cache policy, query lists whose text features are kept


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params` + `[options]`."""

    arch: str  # transformers class
    score_threshold: float
    max_detections: int


@dataclass(frozen=True)
class Text:
    """Text features of one query list (batch 1)."""

    features: torch.Tensor  # [1, T, 256] compute dtype: the projected BERT output
    mask: torch.Tensor  # [1, T] bool, all true (no padding)
    attention: torch.Tensor  # [1, T, T] bool: tokens attend within their phrase (+ themselves)
    positions: torch.Tensor  # [1, T] int64: position ids restarting after each separator
    classes: torch.Tensor  # [C, T] float32: each class's tokens, rows sum to 1


def target_size(h: int, w: int) -> tuple[int, int]:
    """The processor's output size (`get_size_with_aspect_ratio(…, 800, 1333)`)."""
    size, raw = SHORT, None
    lo, hi = float(min(h, w)), float(max(h, w))
    if hi / lo * size > LONG:
        raw = LONG * lo / hi
        size = round(raw)
    if (h <= w and h == size) or (w <= h and w == size):
        return h, w
    if w < h:
        return int((raw if raw is not None else size) * h / w), size
    return size, int((raw if raw is not None else size) * w / h)


def phrase_masks(ids: list[int], separators: set[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """`generate_masks_with_special_tokens_and_transfer_map` for one caption: attention `[T, T]`, position ids `[T]`."""
    t = len(ids)
    idx = torch.arange(t)
    special = torch.tensor([i in separators for i in ids])
    prev = torch.cummax(torch.where(special, idx, -1), 0)[0]
    nxt = torch.flip(torch.cummin(torch.flip(torch.where(special, idx, t), [0]), 0)[0], [0])
    valid = (nxt != 0) & (nxt != t - 1) & (nxt != t)
    att = ((nxt[:, None] == nxt[None, :]) & valid[None, :]) | torch.eye(t, dtype=torch.bool)
    pos = torch.where(valid, idx - prev - 1, 0).clamp(min=0)
    return att, pos


def _msda(value, shapes, shapes_list, start, loc, weights, step):  # noqa: ANN001, ANN202, ARG001, PLR0913, PLR0917 - transformers' signature
    """`MultiScaleDeformableAttention.forward` on `deformable_sample`: `[B, Q, heads * dh]`, no stack of the samples."""
    points = loc.shape[4]
    return deformable_sample(value, loc.flatten(3, 4), weights.flatten(3), shapes_list, points)


def _attention_forward(  # noqa: PLR0913, PLR0917 - transformers' signature
    self: Any,  # noqa: ANN401 - upstream attention module
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    encoder_hidden_states: torch.Tensor | None = None,
    encoder_attention_mask: torch.Tensor | None = None,
    position_embeddings: torch.Tensor | None = None,
    reference_points: torch.Tensor | None = None,
    spatial_shapes: torch.Tensor | None = None,
    spatial_shapes_list: list[tuple[int, int]] | None = None,
    level_start_index: torch.Tensor | None = None,
    output_attentions: bool = False,  # noqa: FBT001, FBT002 - upstream signature
) -> tuple[torch.Tensor, torch.Tensor]:
    """Export the upstream attention without its redundant internal spatial-length assertion.

    The runner constructs both the flattened features and spatial shapes from the same maps. Ordinary eager and
    compiled calls retain upstream's check; this path only changes fixed/dynamic ONNX tracing.
    """
    if not torch.compiler.is_exporting():
        return base_type(self).forward(
            self,
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            position_embeddings,
            reference_points,
            spatial_shapes,
            spatial_shapes_list,
            level_start_index,
            output_attentions,
        )
    if position_embeddings is not None:
        hidden_states = hidden_states + position_embeddings
    assert encoder_hidden_states is not None  # noqa: S101
    assert reference_points is not None  # noqa: S101
    assert spatial_shapes is not None  # noqa: S101
    batch, queries, _ = hidden_states.shape
    length = encoder_hidden_states.shape[1]
    value = self.value_proj(encoder_hidden_states)
    if attention_mask is not None:
        value = value.masked_fill(~attention_mask[..., None], 0.0)
    value = value.view(batch, length, self.n_heads, self.d_model // self.n_heads)
    offsets = self.sampling_offsets(hidden_states).view(batch, queries, self.n_heads, self.n_levels, self.n_points, 2)
    weights = self.attention_weights(hidden_states).view(batch, queries, self.n_heads, self.n_levels * self.n_points)
    weights = F.softmax(weights, -1).view(batch, queries, self.n_heads, self.n_levels, self.n_points)
    if reference_points.shape[-1] == 2:  # noqa: PLR2004 - point coordinates
        normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        locations = reference_points[:, :, None, :, None, :] + offsets / normalizer[None, None, None, :, None, :]
    elif reference_points.shape[-1] == 4:  # noqa: PLR2004 - box coordinates
        locations = (reference_points[:, :, None, :, None, :2] + offsets / self.n_points
                     * reference_points[:, :, None, :, None, 2:] * 0.5)  # fmt: skip
    else:
        msg = f"Last dim of reference_points must be 2 or 4, got {reference_points.shape[-1]}"
        raise ValueError(msg)
    result = self.attn(
        value, spatial_shapes, spatial_shapes_list, level_start_index, locations, weights, self.im2col_step
    )
    return self.output_proj(result), weights


class Portable(Runner):
    """transformers' network in the compute dtype (text tower float32, once per query list), heads float32."""

    dtypes = ("float32", "float16")
    Params = Params
    open_vocabulary = True
    compiled_methods = ("image",)  # `Model.compile()`: backbone, encoder, decoder, heads (not the text cache)

    def __init__(self, ctx: Context) -> None:
        """Build the network from the variant's config and weights; the tokenizer; the default (label set) queries."""
        super().__init__(ctx)
        paths = ctx.weights()
        net: Any = pretrained(ctx, config=0, weights=1)
        self.dt = ctx.torch_dtype
        net.to(ctx.device)
        m = net.model
        for part in (m.backbone, m.input_proj_vision, m.encoder, m.decoder, m.enc_output, m.enc_output_norm):
            part.to(self.dt)
        m.encoder_output_bbox_embed.to(self.dt)
        m.query_position_embeddings.to(self.dt)
        m.level_embed.data = m.level_embed.data.to(self.dt)
        self.net = net
        for mod in net.modules():  # fastforks: the shared per-level deformable sampling (no stack of the samples)
            if type(mod).__name__ == "MultiScaleDeformableAttention":
                mod.forward = _msda
            elif type(mod).__name__ in {
                "GroundingDinoMultiscaleDeformableAttention",
                "MMGroundingDinoMultiscaleDeformableAttention",
            }:
                override(mod, _attention_forward)
        self.head = net.bbox_embed[-1]
        # shared with the decoder (the same module, or tied weights since transformers 5.2x): a float copy
        if any(p.dtype != torch.float32 for p in self.head.parameters()):
            self.head = _float_copy(self.head)
        bias = getattr(net.class_embed[-1], "bias", None)  # MM-Grounding-DINO: logits / sqrt(d) + a learned bias
        self.logit_affine = None if bias is None else (math.sqrt(net.config.d_model), float(bias))
        self.mean = torch.tensor(MEAN, device=ctx.device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=ctx.device).view(1, 3, 1, 1)
        self.tokenizer = BertTokenizer(paths[2])
        self.separators = {self.tokenizer.vocab[s] for s in SEPARATORS}
        self.max_len = net.config.max_text_len
        self.default = tuple(label_set(ctx.variant.output_info["label_set"]))
        self._text: OrderedDict[tuple[str, ...], Text] = OrderedDict()
        self._lock = threading.Lock()

    def caption(self, names: tuple[str, ...]) -> tuple[list[int], torch.Tensor]:
        """Token ids of `" . ".join(names) + " ."` and each name's token map `[C, T]` (rows sum to 1)."""
        tok = self.tokenizer
        dot = tok.vocab["."]
        ids, spans = [tok.cls], []
        for n in names:
            p = tok.encode(n)
            if not p or all(i == dot for i in p):
                msg = f"class name {n!r} has no tokens besides separators"
                raise InputError(msg)
            spans.append((len(ids), len(ids) + len(p)))
            ids += [*p, dot]
        ids.append(tok.sep)
        if len(ids) > self.max_len:
            msg = f"the caption of {len(names)} class names has {len(ids)} tokens, the model reads {self.max_len}"
            raise InputError(msg)
        classes = torch.zeros(len(names), len(ids))
        for c, (a, b) in enumerate(spans):
            classes[c, a:b] = 1 / (b - a)
        return ids, classes

    def queries(self, names: tuple[str, ...]) -> Text:
        """Text features of `names` (cached for the `_TEXT_CACHE` latest lists)."""
        with self._lock:
            hit = self._text.get(names)
            if hit is not None:
                self._text.move_to_end(names)
                return hit
        ids, classes = self.caption(names)
        att, pos = phrase_masks(ids, self.separators)
        dev = self.ctx.device
        m = self.net.model
        i = torch.tensor([ids], device=dev)
        att, pos = att[None].to(dev), pos[None].to(dev)
        with torch.no_grad():
            h = m.text_backbone(i, att[:, None], torch.zeros_like(i), pos).last_hidden_state
            feat = m.text_projection(h).to(self.dt)
        text = Text(feat, torch.ones_like(i, dtype=torch.bool), att, pos, classes.to(dev))
        with self._lock:
            text = self._text.setdefault(names, text)
            self._text.move_to_end(names)
            while len(self._text) > _TEXT_CACHE:
                self._text.popitem(last=False)
        return text

    def pixels(self, image: torch.Tensor) -> torch.Tensor:
        """`[1, 3, H, W]` in [0, 1] -> the normalised network input (module docstring), float32."""
        size = target_size(*image.shape[-2:])
        x = (image.float() * 255).round().clamp(0, 255)
        if size != tuple(x.shape[-2:]):
            if x.device.type == "cpu":  # torchvision's uint8 path: torch's integer antialiasing kernel (CPU only)
                x = F.interpolate(x.to(torch.uint8), size=size, mode="bilinear", align_corners=False, antialias=True)
            else:  # torchvision elsewhere: float, rounded back to uint8 values
                x = (
                    F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)
                    .round()
                    .clamp(0, 255)
                )
        return (x.float() / 255 - self.mean) / self.std

    def image(
        self, x: torch.Tensor, feat: torch.Tensor, att: torch.Tensor, pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalised input `[B, 3, H, W]` + text (batch 1) -> token logits `[B, Q, T]` and boxes `[B, Q, 4]`, float32.

        `GroundingDinoModel.forward` + the last decoder layer's heads of `GroundingDinoForObjectDetection.forward`
        (two-stage, all pixels and tokens valid).
        """
        m = self.net.model
        b = x.shape[0]
        x = x.to(self.dt)
        feat, att, pos = feat.expand(b, -1, -1), att.expand(b, -1, -1), pos.expand(b, -1)
        tmask = torch.ones(feat.shape[:2], dtype=torch.bool, device=x.device)
        pmask = torch.ones((b, *x.shape[-2:]), dtype=torch.long, device=x.device)
        vis, poss = m.backbone(x, pmask)
        del pmask
        maps = [m.input_proj_vision[i](s) for i, (s, _) in enumerate(vis)]
        masks = [k for _, k in vis]
        for lvl in range(len(maps), m.config.num_feature_levels):
            s = m.input_proj_vision[lvl](vis[-1][0] if lvl == len(vis) else maps[-1])
            k = torch.ones((b, *s.shape[-2:]), dtype=torch.bool, device=x.device)
            maps.append(s)
            masks.append(k)
            poss.append(m.backbone.position_embedding(s, k).to(s.dtype))
            del s, k
        del vis
        shapes = [tuple(s.shape[-2:]) for s in maps]
        src = torch.cat([s.flatten(2).transpose(1, 2) for s in maps], 1)
        mflat = torch.cat([k.flatten(1) for k in masks], 1)
        lpos = torch.cat(
            [p.flatten(2).transpose(1, 2) + m.level_embed[i].view(1, 1, -1) for i, p in enumerate(poss)], 1
        )
        del maps, masks, poss
        sh = torch.as_tensor(shapes, dtype=torch.long, device=x.device)
        start = torch.cat((sh.new_zeros((1,)), sh.prod(1).cumsum(0)[:-1]))
        ratios = torch.ones((b, len(shapes), 2), dtype=src.dtype, device=x.device)
        del x
        enc = m.encoder(
            vision_features=src,
            vision_attention_mask=~mflat,
            vision_position_embedding=lpos,
            spatial_shapes=sh,
            spatial_shapes_list=shapes,
            level_start_index=start,
            valid_ratios=ratios,
            text_features=feat,
            text_attention_mask=~tmask,
            text_position_embedding=None,
            text_self_attention_masks=~att,
            text_position_ids=pos,
            return_dict=True,
        )
        vis_h, txt_h = enc.last_hidden_state_vision, enc.last_hidden_state_text
        del enc, src, lpos, feat, att, pos
        obj, proposals = m.generate_encoder_output_proposals(vis_h, ~mflat, shapes)
        enc_cls = m.encoder_output_class_embed(obj, txt_h, tmask)
        coords = m.encoder_output_bbox_embed(obj) + proposals
        del obj, proposals
        top = torch.topk(enc_cls.max(-1)[0], m.config.num_queries, dim=1)[1]
        del enc_cls
        ref = torch.gather(coords, 1, top.unsqueeze(-1).expand(-1, -1, 4)).sigmoid()
        del coords, top
        target = m.query_position_embeddings.weight.unsqueeze(0).expand(b, -1, -1)
        dec = m.decoder(
            inputs_embeds=target,
            vision_encoder_hidden_states=vis_h,
            vision_encoder_attention_mask=mflat,
            text_encoder_hidden_states=txt_h,
            text_encoder_attention_mask=~tmask,
            reference_points=ref,
            spatial_shapes=sh,
            spatial_shapes_list=shapes,
            level_start_index=start,
            valid_ratios=ratios,
            self_attn_mask=None,
            return_dict=True,
        )
        hid = dec.intermediate_hidden_states[:, -1].float()
        last_ref = dec.intermediate_reference_points[:, -2] if len(m.decoder.layers) > 1 else ref
        logits = hid @ txt_h.float().transpose(-1, -2)
        if self.logit_affine is not None:
            logits = logits / self.logit_affine[0] + self.logit_affine[1]
        boxes = (self.head(hid) + torch.special.logit(last_ref.float(), eps=1e-5)).sigmoid()
        return logits, boxes

    def _image(self, x: torch.Tensor, text: Text) -> tuple[torch.Tensor, torch.Tensor]:
        return self.image(x, text.features, text.attention, text.positions)

    def module(self) -> torch.nn.Module:
        """The network for the card's label set, the text features as constants.

        Normalised RGB `[B, 3, H, W]` -> class scores `[B, 900, 80]` (0..1: the mean token probability of each class
        name) and boxes `[B, 900, 4]` (centre x / y, width, height, 0..1 of the image).
        """
        t = self.queries(self.default)
        feat, att, pos, classes = (v.clone() for v in (t.features, t.attention, t.positions, t.classes))

        def fn(net: Any, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN401, ARG001 - Forward's call
            logits, boxes = self.image(x, feat, att, pos)
            return logits.sigmoid() @ classes.T, boxes

        return Forward(self.net, fn, output_names=("scores", "boxes"), query_scores=lambda s: s.amax(-1))

    @host
    def __call__(
        self, image: torch.Tensor, score_threshold: float, text: tuple[str, ...] | None = None
    ) -> dict[str, torch.Tensor]:
        """Detections of `text` (default: the card's label set) in one image."""
        t = self.queries(self.default if text is None else text)
        logits, boxes = self._image(self.pixels(image), t)
        prob = logits[0].sigmoid() @ t.classes.T  # [Q, C]
        n = prob.shape[1]
        k = min(self.ctx.params.max_detections, prob.numel())
        scores, flat = prob.flatten().topk(k)
        keep = scores > score_threshold
        scores, flat = scores[keep], flat[keep]
        cx, cy, bw, bh = boxes[0][flat // n].unbind(-1)
        h, w = image.shape[-2:]
        out = torch.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), -1) * boxes.new_tensor([w, h, w, h])
        return {"boxes": out, "scores": scores, "labels": flat % n}


def _float_copy(mod: torch.nn.Module) -> torch.nn.Module:
    import copy  # noqa: PLC0415

    return copy.deepcopy(mod).float()


class Kernels(Portable):
    """Fuse all levels of bilinear sampling and the weighted reduction in one CUDA launch."""

    module = Runner.module

    def __init__(self, ctx: Context) -> None:
        """Prepare the portable network and replace its multi-scale sampler."""
        super().__init__(ctx)
        msda.load(ctx.device)
        for mod in self.net.modules():
            if type(mod).__name__ == "MultiScaleDeformableAttention":
                mod.forward = msda.sample


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(Portable, attrs=(), methods=("image",)),
}
