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

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._core.errors import InputError
from fastforks._core.labels import label_set
from fastforks.detection._runner import Forward, Runner
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
        self.head = net.bbox_embed[-1]
        if self.head is m.decoder.bbox_embed[-1] and self.dt != torch.float32:  # shared with the decoder: a float copy
            self.head = _float_copy(self.head)
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
        maps = [m.input_proj_vision[i](s) for i, (s, _) in enumerate(vis)]
        masks = [k for _, k in vis]
        for lvl in range(len(maps), m.config.num_feature_levels):
            s = m.input_proj_vision[lvl](vis[-1][0] if lvl == len(vis) else maps[-1])
            k = torch.ones((b, *s.shape[-2:]), dtype=torch.bool, device=x.device)
            maps.append(s)
            masks.append(k)
            poss.append(m.backbone.position_embedding(s, k).to(s.dtype))
        shapes = [tuple(s.shape[-2:]) for s in maps]
        src = torch.cat([s.flatten(2).transpose(1, 2) for s in maps], 1)
        mflat = torch.cat([k.flatten(1) for k in masks], 1)
        lpos = torch.cat(
            [p.flatten(2).transpose(1, 2) + m.level_embed[i].view(1, 1, -1) for i, p in enumerate(poss)], 1
        )
        sh = torch.as_tensor(shapes, dtype=torch.long, device=x.device)
        start = torch.cat((sh.new_zeros((1,)), sh.prod(1).cumsum(0)[:-1]))
        ratios = torch.ones((b, len(shapes), 2), dtype=src.dtype, device=x.device)
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
        obj, proposals = m.generate_encoder_output_proposals(vis_h, ~mflat, shapes)
        enc_cls = m.encoder_output_class_embed(obj, txt_h, tmask)
        coords = m.encoder_output_bbox_embed(obj) + proposals
        top = torch.topk(enc_cls.max(-1)[0], m.config.num_queries, dim=1)[1]
        ref = torch.gather(coords, 1, top.unsqueeze(-1).expand(-1, -1, 4)).sigmoid()
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


RUNNERS = {"portable": Portable}
