"""OWLv2 runners: the transformers `Owlv2ForObjectDetection` network, our image tower forwards, cached text queries.

As the Hugging Face `Owlv2ImageProcessor` + `post_process_object_detection` (transformers 5.19):

- preprocessing: pad the image to a square (bottom / right, value 0), Gaussian anti-aliasing blur (sigma
  `(factor - 1) / 2`, kernel `2 ceil(3 sigma) + 1`, reflect border), bilinear resize to `image_size` without
  antialiasing, CLIP mean / std;
- one box per patch token; its score is the sigmoid of the best query's logit, its label that query; kept if the score
  is above the threshold (strictly); boxes are scaled by the padded square's side (they may leave the image).

The queries are class names (`text`), else the card's label set; each name is tokenized as the `CLIPTokenizer` of the
checkpoint (max 16 tokens) and embedded by the text tower once (the embeddings of the last `_TEXT_CACHE` query lists
are kept). The image tower and the heads run every call: the encoder layers in the compute dtype (portable: merged qkv +
SDPA; kernels: fused fp16 GEMMs, CUDA graph), the final norms and heads in float32.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._core.half import fp16_accumulate
from fastforks._core.labels import label_set
from fastforks.detection._runner import Forward, Runner
from fastforks.shared.clip_tokenizer import ClipTokenizer
from fastforks.shared.cuda_graphs import GraphCache
from fastforks.shared.seginfer import pretrained

if TYPE_CHECKING:
    from fastforks.detection._runner import Context

MEAN = (0.48145466, 0.4578275, 0.40821073)  # fixed: OpenAI CLIP normalisation (the image processor's)
STD = (0.26862954, 0.26130258, 0.27577711)  # fixed: OpenAI CLIP normalisation
CONTEXT = 16  # fixed: the processor's text max_length (OWL-ViT / OWLv2 position table)
QUICK_GELU = 1.702  # fixed: QuickGELU(x) = x * sigmoid(1.702 x) = SiLU(1.702 x) / 1.702
SILU = 3  # fixed: fused-GEMM epilogue code of SiLU (`fastforks._optimize.conv_ops`)
_TEXT_CACHE = 8  # fixed: cache policy, query lists whose text embeddings are kept


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params` + `[options]`."""

    arch: str  # transformers class
    score_threshold: float
    max_detections: int
    cuda_graphs: bool


def gaussian_resize(x: torch.Tensor, size: int) -> torch.Tensor:
    """`[B, 3, S, S]` float32 -> `[B, 3, size, size]`: the processor's anti-aliasing blur, then bilinear (no antialias).

    The blur is torchvision's `gaussian_blur` (reflect border), separated into a row and a column pass.
    """
    factor = x.shape[-1] / size
    sigma = max((factor - 1) / 2, 0.0)
    if sigma > 0:
        k = 2 * math.ceil(3 * sigma) + 1
        t = torch.linspace(-(k // 2), k // 2, k, device=x.device)
        g = torch.exp(-0.5 * (t / sigma) ** 2)
        g = (g / g.sum()).to(x.dtype)
        c = x.shape[1]
        x = F.pad(x, (k // 2,) * 4, mode="reflect")
        x = F.conv2d(x, g.view(1, 1, 1, k).expand(c, 1, 1, k), groups=c)
        x = F.conv2d(x, g.view(1, 1, k, 1).expand(c, 1, k, 1), groups=c)
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=False)


class Portable(Runner):
    """transformers' network; image tower layers in the compute dtype (merged qkv + SDPA), norms and heads float32."""

    dtypes = ("float32", "float16")
    Params = Params
    open_vocabulary = True
    compiled_methods = ("image",)  # `Model.compile()`: the image tower and heads (not the tokenizer / text cache)

    def __init__(self, ctx: Context) -> None:
        """Build the network from the variant's config and weights; the tokenizer; the default (label set) queries."""
        super().__init__(ctx)
        paths = ctx.weights()
        net: Any = pretrained(ctx, config=0, weights=1)
        self.dt = ctx.torch_dtype
        vc = net.config.vision_config
        self.size, self.patch = vc.image_size, vc.patch_size
        net.to(ctx.device)
        vis = net.owlv2.vision_model
        vis.embeddings.to(self.dt)
        vis.pre_layernorm.to(self.dt)
        vis.encoder.to(self.dt)
        net.owlv2.text_model.to(self.dt)
        net.owlv2.text_projection.to(self.dt)
        self.net = net
        self.layers = list(vis.encoder.layers)
        for lay in self.layers:
            a = lay.self_attn
            with torch.no_grad():
                qkv = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight])
                bqkv = torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias])
            lay.register_buffer("ff_wqkv", qkv.detach(), persistent=False)
            lay.register_buffer("ff_bqkv", bqkv.detach(), persistent=False)
        self.mean = torch.tensor(MEAN, device=ctx.device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=ctx.device).view(1, 3, 1, 1)
        self.tokenizer = ClipTokenizer(paths[2], paths[3], context=CONTEXT)
        self.default = tuple(label_set(ctx.variant.output_info["label_set"]))
        self._text: OrderedDict[tuple[str, ...], torch.Tensor] = OrderedDict()
        self._lock = threading.Lock()

    def queries(self, names: tuple[str, ...]) -> torch.Tensor:
        """Normalised text embeddings `[Q, D]` float32 of `names` (cached for the `_TEXT_CACHE` latest lists)."""
        with self._lock:
            hit = self._text.get(names)
            if hit is not None:
                self._text.move_to_end(names)
                return hit
        ids = torch.zeros(len(names), CONTEXT, dtype=torch.int64)  # padding: "!" (id 0), after eot (causal mask)
        for i, n in enumerate(names):
            t = self.tokenizer(n)
            ids[i, : len(t)] = torch.tensor(t)
        o = self.net.owlv2
        with torch.no_grad():
            pooled = o.text_model(input_ids=ids.to(self.ctx.device)).pooler_output
            q = o.text_projection(pooled).float()
        q = q / (torch.linalg.norm(q, dim=-1, keepdim=True) + 1e-6)
        with self._lock:
            q = self._text.setdefault(names, q)
            self._text.move_to_end(names)
            while len(self._text) > _TEXT_CACHE:
                self._text.popitem(last=False)
        return q

    def pixels(self, image: torch.Tensor) -> torch.Tensor:
        """`[1, 3, H, W]` in [0, 1] -> the normalised network input `[1, 3, S, S]` float32 (module docstring)."""
        h, w = image.shape[-2:]
        s = max(h, w)
        x = F.pad(image.float(), (0, s - w, 0, s - h))
        x = gaussian_resize(x, self.size) if s != self.size else x
        return (x - self.mean) / self.std

    def layer(self, lay: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - an Owlv2EncoderLayer
        """`Owlv2EncoderLayer.forward` with one qkv GEMM and SDPA."""
        a, m = lay.self_attn, lay.mlp
        b, n, c = x.shape
        h = a.num_heads
        y = lay.layer_norm1(x)
        q, k, v = F.linear(y, lay.ff_wqkv, lay.ff_bqkv).view(b, n, 3, h, c // h).permute(2, 0, 3, 1, 4).unbind(0)
        o = F.scaled_dot_product_attention(q, k, v, scale=a.scale).transpose(1, 2).reshape(b, n, c)
        x = x + a.out_proj(o)
        y = m.fc1(lay.layer_norm2(x))
        return x + m.fc2(y * torch.sigmoid(QUICK_GELU * y))

    def image(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalised network input -> class embeddings, logit shift, logit scale and boxes (float32).

        Class embeddings `[B, P, D]` normalised, shift and scale `[B, P, 1]`, boxes `[B, P, 4]` centre / size in 0..1;
        `P` patches row-major.
        """
        net = self.net
        e = net.owlv2.vision_model.embeddings
        p = e.patch_embedding(x.to(self.dt)).flatten(2).transpose(1, 2)
        t = torch.cat([e.class_embedding.expand(p.shape[0], 1, -1), p], 1) + e.position_embedding.weight
        t = net.owlv2.vision_model.pre_layernorm(t)
        for lay in self.layers:
            t = self.layer(lay, t)
        return self.heads(t)

    def heads(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The float32 part after the encoder (see `image`)."""
        net = self.net
        t = net.owlv2.vision_model.post_layernorm(t.float())
        f = net.layer_norm(t[:, 1:] * t[:, :1])
        ch = net.class_head
        emb = ch.dense0(f)
        emb = emb / (torch.linalg.norm(emb, dim=-1, keepdim=True) + 1e-6)
        boxes = torch.sigmoid(net.box_head(f) + net.box_bias)
        return emb, ch.logit_shift(f), F.elu(ch.logit_scale(f)) + 1, boxes

    def _image(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.image(x)

    def module(self) -> torch.nn.Module:
        """The network for the card's label set, the text embeddings as constants.

        Normalised RGB `[B, 3, S, S]` (S = the variant's `image_size`) -> class logits `[B, P, 80]` (apply a sigmoid)
        and boxes `[B, P, 4]` (centre x / y, width, height of the padded square, 0..1), one per patch.
        """
        q = self.queries(self.default).clone()  # a plain tensor (the cache may hold an inference-mode one)

        def fn(net: Any, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN401, ARG001 - Forward's call
            emb, shift, scale, boxes = self.image(x)
            return (emb @ q.T + shift) * scale, boxes

        return Forward(
            self.net, fn, output_names=("logits", "boxes"), query_scores=lambda logits: logits.sigmoid().amax(-1)
        )

    def __call__(
        self, image: torch.Tensor, score_threshold: float, text: tuple[str, ...] | None = None
    ) -> dict[str, torch.Tensor]:
        """Detections of `text` (default: the card's label set) in one image."""
        q = self.queries(self.default if text is None else text)
        emb, shift, scale, boxes = self._image(self.pixels(image))
        logits = (emb[0] @ q.T + shift[0]) * scale[0]
        best, labels = logits.max(-1)
        scores = torch.sigmoid(best)
        keep = scores > score_threshold
        scores, order = scores[keep].sort(descending=True, stable=True)
        order = order[: self.ctx.params.max_detections]
        cx, cy, bw, bh = boxes[0][keep][order].unbind(-1)
        s = float(max(image.shape[-2:]))
        out = torch.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), -1) * s
        return {"boxes": out, "scores": scores[: order.shape[0]], "labels": labels[keep][order]}


def _w4(w: torch.Tensor) -> torch.Tensor:
    """A Linear weight `[out, in]` as the fused GEMM's fp16 `[out, in, 1, 1]` channels-last filter."""
    return w.detach().half().reshape(*w.shape, 1, 1).contiguous(memory_format=torch.channels_last)


class Kernels(Portable):
    """Fused fp16 GEMMs, flash attention and the image tower + heads replayed from a CUDA graph.

    GEMM epilogues: bias / SiLU (QuickGELU with folded scales) / residual; fp16-accumulate cuBLAS elsewhere.
    """

    def __init__(self, ctx: Context) -> None:
        """The portable network with the fused-GEMM filters attached; the graph cache."""
        super().__init__(ctx)
        for lay in self.layers:
            a, m = lay.self_attn, lay.mlp
            with torch.no_grad():
                w1, b1 = m.fc1.weight.float() * QUICK_GELU, m.fc1.bias.float() * QUICK_GELU
                w2 = m.fc2.weight.float() / QUICK_GELU
            bufs = {"ff_k_qkv": _w4(lay.ff_wqkv), "ff_k_o": _w4(a.out_proj.weight), "ff_k_1": _w4(w1)}
            bufs["ff_k_b1"] = b1.half()
            for k, v in {**bufs, "ff_k_2": _w4(w2)}.items():
                lay.register_buffer(k, v.detach(), persistent=False)
        self.graphs = GraphCache(self._joined, graphs=1) if ctx.params.cuda_graphs else None

    def layer(self, lay: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - an Owlv2EncoderLayer
        """`Owlv2EncoderLayer.forward` with fused GEMM epilogues."""
        from fastforks._optimize.conv_ops import lin  # noqa: PLC0415 - loads the fvm_conv kernels (kernels mode only)

        a, m = lay.self_attn, lay.mlp
        b, n, c = x.shape
        h = a.num_heads
        q, k, v = (
            lin(lay.layer_norm1(x), lay.ff_k_qkv, lay.ff_bqkv).view(b, n, 3, h, c // h).permute(2, 0, 3, 1, 4).unbind(0)
        )
        o = F.scaled_dot_product_attention(q, k, v, scale=a.scale).transpose(1, 2).reshape(b, n, c)
        x = lin(o, lay.ff_k_o, a.out_proj.bias, res=x)
        y = lin(lay.layer_norm2(x), lay.ff_k_1, lay.ff_k_b1, act=SILU)
        return lin(y, lay.ff_k_2, m.fc2.bias, res=x)

    def _joined(self, x: torch.Tensor) -> torch.Tensor:
        """`image` as one float32 tensor `[B, P, D + 6]` (one graph output)."""
        with fp16_accumulate():
            return torch.cat(self.image(x), -1)

    def _image(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = None
        if self.graphs is not None and x.is_cuda:
            out = self.graphs.run((x.device, tuple(x.shape)), x)
        if out is None:
            out = self._joined(x)
        d = out.shape[-1] - 6
        return out.split([d, 1, 1, 4], -1)

    def release_graphs(self) -> None:
        """Free the CUDA graph and its memory pool."""
        if self.graphs is not None:
            self.graphs.release()


RUNNERS = {"portable": Portable, "kernels": Kernels}
