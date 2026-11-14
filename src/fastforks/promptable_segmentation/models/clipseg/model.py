"""CLIPSeg runners: the transformers `CLIPSegForImageSegmentation` network, our forwards, cached text prompts.

As the Hugging Face `CLIPSegProcessor` + `CLIPSegForImageSegmentation` (transformers 5.19):

- input: the image as uint8, resized to 352 x 352 (bilinear, antialiased; the aspect ratio is not kept), `/ 255`,
  ImageNet mean / std; CPU: the uint8 resize itself (exact), CUDA: a float resize rounded to integers
- image: the CLIP ViT-B/16 vision tower with its 7 x 7 position table bicubically resized to 22 x 22 (once, at load);
  only the layers up to the last extracted one (9 of 12) run, as the outputs of the others are unused
- text: CLIP byte-level BPE (the checkpoint's vocabulary, 77 tokens), the CLIP text tower's end token, projected;
  the embeddings of the last `_TEXT_CACHE` prompts are kept
- decoder: the activations after layers 9, 6, 3, each reduced to 64 channels and added to the running state, FiLM by
  the text embedding before the first decoder layer, 3 post-norm transformer layers, the patch tokens back to a
  22 x 22 map, the transposed-convolution head to 352 x 352 mask logits

The API returns one instance per prompt (CLIPSeg is semantic: one mask for every pixel of the concept): its mask is
the logits > 0, bilinearly resized to the input size; its score the highest pixel probability (sigmoid of the logits),
so it is kept at the default threshold 0.5 exactly when the mask is not empty; its box the mask's bounding box.

Portable runs the vision tower (merged qkv + SDPA), text tower and decoder in the compute dtype; kernels adds fused
fp16 GEMMs to the vision tower and replays the vision tower and the decoder from CUDA graphs.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from fastforks._core.errors import InputError
from fastforks._core.half import fp16_accumulate
from fastforks.promptable_segmentation._runner import Context, Forward, Runner, scale
from fastforks.shared.clip_tokenizer import ClipTokenizer
from fastforks.shared.cuda_graphs import GraphCache
from fastforks.shared.seginfer import pretrained

SIZE = 352  # fixed: the processor's input side (preprocessor_config.json)
CONTEXT = 77  # fixed: CLIP text context (the text tower's position table)
QUICK_GELU = 1.702  # fixed: QuickGELU(x) = x * sigmoid(1.702 x) = SiLU(1.702 x) / 1.702
SILU = 3  # fixed: fused-GEMM epilogue code of SiLU (`fastforks._optimize.conv_ops`)
_TEXT_CACHE = 16  # fixed: cache policy, prompts whose text embeddings are kept


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params` + `[options]`."""

    arch: str  # transformers class
    cuda_graphs: bool


def box(mask: torch.Tensor) -> torch.Tensor:
    """Bounding box `[4]` `(x0, y0, x1, y1)` of a boolean `[h, w]` mask, normalised by its size (zeros if empty)."""
    h, w = mask.shape
    rows, cols = mask.any(1), mask.any(0)
    i = torch.arange(max(h, w), device=mask.device)
    big = max(h, w)
    y0, y1 = torch.where(rows, i[:h], big).amin(), torch.where(rows, i[:h], -1).amax() + 1
    x0, x1 = torch.where(cols, i[:w], big).amin(), torch.where(cols, i[:w], -1).amax() + 1
    b = torch.stack((x0, y0, x1, y1)).float() / torch.tensor([w, h, w, h], device=mask.device, dtype=torch.float32)
    return torch.where(rows.any(), b, 0.0)


class Portable(Runner):
    """transformers' network; vision tower (merged qkv + SDPA), text tower and decoder in the compute dtype."""

    dtypes = ("float32", "float16")
    Params = Params
    prompts = frozenset({"text"})
    mask_size = (SIZE, SIZE)
    compiled_methods = ("embed", "decode")  # `segment_concept` tokenizes on the host (cached), then calls `decode`

    def __init__(self, ctx: Context) -> None:
        """Build the network, keep the vision layers in use, merge q / k / v, resize the position table, tokenizer."""
        super().__init__(ctx)
        paths = ctx.weights()
        net: Any = pretrained(ctx, config=0, weights=1)
        cfg = net.config
        self.dt = dt = ctx.torch_dtype
        self.extract = tuple(cfg.extract_layers)
        vis = net.clip.vision_model
        vis.encoder.layers = vis.encoder.layers[: max(self.extract) + 1]  # the later layers feed nothing we use
        del net.clip.visual_projection, vis.post_layernorm
        e = vis.embeddings
        grid = SIZE // e.patch_size
        with torch.no_grad():  # the processor's size is fixed: the bicubic position table once, not per call
            pos = e.interpolate_pos_encoding(
                torch.zeros(1, grid * grid + 1, e.position_embedding.weight.shape[1]), SIZE, SIZE
            )
        net.to(ctx.device, dt)
        net.register_buffer("ff_pos", pos.to(ctx.device, dt), persistent=False)
        self.net = net
        self.layers = list(vis.encoder.layers)
        for lay in self.layers:
            a = lay.self_attn
            with torch.no_grad():
                qkv = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight])
                bqkv = torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias])
            lay.register_buffer("ff_wqkv", qkv.detach(), persistent=False)
            lay.register_buffer("ff_bqkv", bqkv.detach(), persistent=False)
        self.tokenizer = ClipTokenizer(paths[2], paths[3], context=CONTEXT)
        self._text: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._lock = threading.Lock()

    def condition(self, text: str) -> torch.Tensor:
        """The projected text embedding `[1, 512]` (compute dtype) of `text`, cached for the latest prompts."""
        with self._lock:
            hit = self._text.get(text)
            if hit is not None:
                self._text.move_to_end(text)
                return hit
        ids = torch.tensor([self.tokenizer(text)], device=self.ctx.device)
        clip = self.net.clip
        with torch.no_grad():
            c = clip.text_projection(clip.text_model(input_ids=ids).pooler_output)
        with self._lock:
            c = self._text.setdefault(text, c)
            self._text.move_to_end(text)
            while len(self._text) > _TEXT_CACHE:
                self._text.popitem(last=False)
        return c

    def pixels(self, image: torch.Tensor) -> torch.Tensor:
        """`[1, 3, H, W]` in [0, 1] -> the normalised network input `[1, 3, 352, 352]` float32 (module docstring)."""
        u = (image.float() * 255).round()
        if image.device.type == "cpu":
            x = F.interpolate(u.to(torch.uint8), (SIZE, SIZE), mode="bilinear", antialias=True, align_corners=False)
        else:
            x = F.interpolate(u, (SIZE, SIZE), mode="bilinear", antialias=True, align_corners=False).round()
        return scale(x.float().clamp(0, 255) / 255, self.ctx.io)

    def layer(self, lay: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - a CLIPSegEncoderLayer
        """`CLIPSegEncoderLayer.forward` (pre-norm) with one qkv GEMM and SDPA."""
        a, m = lay.self_attn, lay.mlp
        b, n, c = x.shape
        h = a.num_heads
        y = lay.layer_norm1(x)
        q, k, v = F.linear(y, lay.ff_wqkv, lay.ff_bqkv).view(b, n, 3, h, c // h).permute(2, 0, 3, 1, 4).unbind(0)
        o = F.scaled_dot_product_attention(q, k, v, scale=a.scale).transpose(1, 2).reshape(b, n, c)
        x = x + a.out_proj(o)
        y = m.fc1(lay.layer_norm2(x))
        return x + m.fc2(y * torch.sigmoid(QUICK_GELU * y))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Normalised input -> the vision tower's outputs after the extracted layers (`[1, 485, 768]` each)."""
        vis = self.net.clip.vision_model
        e = vis.embeddings
        p = e.patch_embedding(x.to(self.dt)).flatten(2).transpose(1, 2)
        t = vis.pre_layrnorm(torch.cat([e.class_embedding.expand(p.shape[0], 1, -1), p], 1) + self.net.ff_pos)
        out = []
        for i, lay in enumerate(self.layers):
            t = self.layer(lay, t)
            if i in self.extract:
                out.append(t)
        return tuple(out)

    def embed(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The extracted vision activations `layer3`, `layer6`, `layer9` (compute dtype)."""
        return dict(zip((f"layer{i}" for i in self.extract), self.encode(self.pixels(image)), strict=True))

    def decode(self, acts: tuple[torch.Tensor, ...], cond: torch.Tensor) -> torch.Tensor:
        """`CLIPSegDecoder.forward`: activations (in extraction order) + text embedding -> logits `[1, 352, 352]`."""
        d = self.net.decoder
        out = None
        for i, (act, lay, reduce) in enumerate(zip(acts[::-1], d.layers, d.reduces, strict=True)):
            out = reduce(act) if out is None else reduce(act) + out
            if i == d.conditional_layer:
                out = d.film_mul(cond)[:, None] * out + d.film_add(cond)[:, None]
            a, m = lay.self_attn, lay.mlp
            b, n, c = out.shape
            h = a.num_heads
            q, k, v = (p(out).view(b, n, h, c // h).transpose(1, 2) for p in (a.q_proj, a.k_proj, a.v_proj))
            o = F.scaled_dot_product_attention(q, k, v, scale=a.scale).transpose(1, 2).reshape(b, n, c)
            out = lay.layer_norm1(out + a.out_proj(o))
            out = lay.layer_norm2(out + m.fc2(m.activation_fn(m.fc1(out))))
        assert out is not None  # noqa: S101 - three extracted layers
        b, n, c = out.shape
        g = round((n - 1) ** 0.5)
        y = out[:, 1:].transpose(1, 2).reshape(b, c, g, g)
        return d.transposed_convolution(y)[:, 0].float()

    def _decode(self, features: dict[str, torch.Tensor], cond: torch.Tensor) -> torch.Tensor:
        return self.decode(tuple(features[f"layer{i}"] for i in self.extract), cond)

    def segment_concept(
        self,
        features: dict[str, torch.Tensor],
        text: str | None,
        boxes: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """The one instance of a text prompt: `scores [1]`, `boxes [1, 4]`, `pred_masks [1, 352, 352]` (float32).

        Raises:
            InputError: No text (CLIPSeg takes text prompts only; the core rejects boxes before).
        """
        del labels
        if text is None or boxes is not None:
            msg = "CLIPSeg takes a text prompt (text=...) only"
            raise InputError(msg)
        logits = self._decode(features, self.condition(text))[0]
        return {"scores": logits.amax().sigmoid()[None], "boxes": box(logits > 0)[None], "pred_masks": logits[None]}

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
        """Not used: CLIPSeg takes text prompts (the core sends them to `segment_concept`).

        Raises:
            InputError: Always.
        """
        del features, size, points, labels, boxes, masks, multimask
        msg = "CLIPSeg takes a text prompt (text=...)"
        raise InputError(msg)

    def upsample(self, pred_masks: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Mask logits `[P, M, 352, 352]` -> `[P, M, H, W]` (bilinear)."""
        return F.interpolate(pred_masks, size, mode="bilinear", align_corners=False)

    def module(self) -> torch.nn.Module:
        """Image + projected text embedding `[1, 512]` -> mask logits `[1, 352, 352]`.

        The image is the normalised network input `[1, 3, 352, 352]` (resized and normalised as the module docstring).
        """
        return Forward(
            self.net, lambda _net, x, cond: self.decode(self.encode(x), cond.to(self.dt)), output_names=("logits",)
        )


def _w4(w: torch.Tensor) -> torch.Tensor:
    """A Linear weight `[out, in]` as the fused GEMM's fp16 `[out, in, 1, 1]` channels-last filter."""
    return w.detach().half().reshape(*w.shape, 1, 1).contiguous(memory_format=torch.channels_last)


class Kernels(Portable):
    """Fused fp16 GEMMs in the vision tower, the vision tower and the decoder from CUDA graphs.

    The GEMM epilogues fuse bias, SiLU (QuickGELU's 1.702 folded into the weights) and residuals; cuBLAS runs with
    fp16 accumulation elsewhere.
    """

    def __init__(self, ctx: Context) -> None:
        """The portable network with the fused-GEMM filters attached; the graph caches."""
        super().__init__(ctx)
        for lay in self.layers:
            a, m = lay.self_attn, lay.mlp
            with torch.no_grad():
                w1, b1 = m.fc1.weight.float() * QUICK_GELU, m.fc1.bias.float() * QUICK_GELU
                w2 = m.fc2.weight.float() / QUICK_GELU
            bufs = {"ff_k_qkv": _w4(lay.ff_wqkv), "ff_k_o": _w4(a.out_proj.weight), "ff_k_1": _w4(w1)}
            bufs |= {"ff_k_b1": b1.half(), "ff_k_2": _w4(w2)}
            for k, v in bufs.items():
                lay.register_buffer(k, v.detach(), persistent=False)
            del lay.ff_wqkv  # the fused filter replaces it
        graphs = ctx.params.cuda_graphs
        self.graphs = (GraphCache(self._encode, graphs=1), GraphCache(self._decode16, graphs=1)) if graphs else None

    def layer(self, lay: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401 - a CLIPSegEncoderLayer
        """`CLIPSegEncoderLayer.forward` with fused GEMM epilogues."""
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

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        with fp16_accumulate():
            return Portable.encode(self, x)

    def _decode16(self, *args: torch.Tensor) -> torch.Tensor:
        with fp16_accumulate():
            return self.decode(args[:-1], args[-1])

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """`Portable.encode` with fused GEMMs, from the CUDA graph when on."""
        out = self.graphs[0].run(x.device, x) if self.graphs is not None and x.is_cuda else None
        return self._encode(x) if out is None else tuple(out)

    def _decode(self, features: dict[str, torch.Tensor], cond: torch.Tensor) -> torch.Tensor:
        args = (*(features[f"layer{i}"] for i in self.extract), cond)
        out = self.graphs[1].run(cond.device, *args) if self.graphs is not None and cond.is_cuda else None
        return self._decode16(*args) if out is None else out

    def module(self) -> torch.nn.Module:
        """Not exportable (fused kernels); the portable runner's `module()` is.

        Raises:
            NotImplementedError: Always.
        """
        msg = "kernels has no exportable module; use the portable runner"
        raise NotImplementedError(msg)

    def networks(self) -> tuple[torch.nn.Module, ...]:
        """The network (vision and text towers, decoder)."""
        return (self.net,)

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pools."""
        if self.graphs is not None:
            for g in self.graphs:
                g.release()


RUNNERS = {"portable": Portable, "kernels": Kernels}
