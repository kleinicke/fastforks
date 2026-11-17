"""EfficientViT-SAM input handling (upstream `EfficientViTSam.transform`) and the encoder in a traceable form.

- input: `SamResize` (PIL bilinear on uint8, longest side `input_size`: 512 for L0-L2, 1024 for XL0 / XL1; skipped
  when the longest side already is `input_size`), normalise, zero-pad bottom-right to a square (`SamPad`). On the CPU
  PIL itself resizes (bit-exact); elsewhere `pil_bilinear` (separable antialiased passes with uint8 rounding in
  between, like PIL).
- encoder: upstream `EfficientViTSamImageEncoder` minus its final `LayerNorm2d` (`Trunk`), with the neck's
  `UpSampleLayer`s (shape-dependent Python) swapped for `Upsample` / identity so `torch.fx` can trace it: the input
  is always the `input_size` square, so which stage already is 64 x 64 is known when the network is built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from fastforks._optimize import ops
from fastforks._optimize.graph import attr
from fastforks.promptable_segmentation._runner import scale

from .net.ops import UpSampleLayer

if TYPE_CHECKING:
    from torch import fx

    from fastforks._core.cards import Io

    from .net.sam import EfficientViTSamImageEncoder
    from .net.segment_anything.mask_decoder import MaskDecoder
    from .net.segment_anything.prompt_encoder import PromptEncoder

EMBED = 64  # fixed: SAM's image embedding is 64 x 64 (the mask decoder's dense positional encoding)
WIDE = 1024  # fixed: linear 1x1 convs from at least this many channels accumulate in float32 (see `wide_points`)


def preprocess_shape(h: int, w: int, long_side: int) -> tuple[int, int]:
    """Upstream `SamResize.get_preprocess_shape`: `(h, w)` scaled so the longer side is `long_side`."""
    s = long_side / max(h, w)
    return int(h * s + 0.5), int(w * s + 0.5)


def pil_bilinear(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """PIL `BILINEAR` resize of a float `[B, C, H, W]` 0..255 tensor (horizontal pass, round, vertical pass, round)."""
    (h, w), (nh, nw) = x.shape[-2:], size
    if nw != w:
        x = F.interpolate(x, (h, nw), mode="bilinear", align_corners=False, antialias=True).round_().clamp_(0, 255)
    if nh != h:
        x = F.interpolate(x, (nh, nw), mode="bilinear", align_corners=False, antialias=True).round_().clamp_(0, 255)
    return x


def _pil_resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """PIL itself (CPU, bit-exact with upstream), on a float `[1, 3, H, W]` 0..255 tensor."""
    arr = x[0].permute(1, 2, 0).to(torch.uint8).numpy()
    out = np.asarray(Image.fromarray(arr).resize(size[::-1], Image.Resampling.BILINEAR))
    return torch.from_numpy(out.copy()).permute(2, 0, 1)[None].float()


def prepare(image: torch.Tensor, io: Io, long_side: int, *, exact: bool) -> torch.Tensor:
    """`[1, 3, H, W]` RGB in `[0, 1]` -> the normalised, zero-padded `[1, 3, long_side, long_side]` float32 input.

    Upstream takes uint8 images, so values are rounded to 0..255 first. `exact`: resize with PIL (CPU tensors only).
    """
    size = preprocess_shape(*image.shape[-2:], long_side)
    x = image.mul(255).round_()
    if max(image.shape[-2:]) != long_side:  # upstream SamResize: only when the longest side differs
        x = _pil_resize(x, size) if exact else pil_bilinear(x, size)
    x = scale(x.div_(255), io)
    return F.pad(x, (0, long_side - size[1], 0, long_side - size[0]))


class Upsample(torch.nn.Module):
    """Upstream `UpSampleLayer(size=(64, 64))` as a fixed scale: bicubic in float32, back in the dtype.

    At the encoder's input size the scale gives exactly 64 x 64 (same sampling positions as the fixed size); a scale
    keeps the trunk valid at other sizes, which the BatchNorm-folding probe of `fastforks._optimize` uses.
    """

    def __init__(self, scale: int) -> None:
        """`scale` = 64 / the stage's side at the encoder's input size."""
        super().__init__()
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, C, h, w]` -> `[B, C, scale * h, scale * w]`."""
        return F.interpolate(x.float(), scale_factor=self.scale, mode="bicubic", align_corners=False).to(x.dtype)


class Trunk(torch.nn.Module):
    """Backbone + neck of an upstream image encoder (shared modules): `[B, 3, S, S]` -> `[B, 256, 64, 64]`.

    Without the final `LayerNorm2d`, which the runners apply in float32.
    """

    def __init__(self, enc: EfficientViTSamImageEncoder, input_size: int) -> None:
        """Take `enc`'s modules; swap the neck's `UpSampleLayer`s for their traceable form at `input_size`."""
        super().__init__()
        self.backbone = enc.backbone
        self.neck = enc.neck
        for fid, seq in zip(self.neck.input_keys, self.neck.input_ops, strict=True):
            stride = 2 ** (int(fid.removeprefix("stage")) + 1)  # stage i halves the resolution i + 1 times
            layers = cast("torch.nn.ModuleList", seq.op_list)
            for i, op in enumerate(layers):
                if isinstance(op, UpSampleLayer):
                    side = input_size // stride
                    layers[i] = torch.nn.Identity() if side == EMBED else Upsample(EMBED // side)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The neck's `sam_encoder` output."""
        return self.neck(self.backbone(x))["sam_encoder"]


def point(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None, res: torch.Tensor | None) -> torch.Tensor:
    """Linear 1x1 conv (+ residual) through cuDNN: fp16 in and out, float32 accumulation."""
    y = F.conv2d(x, w, bias)
    return y if res is None else y + res


def tensor_eps(net: torch.nn.Module, device: torch.device) -> int:
    """Each upstream `LiteMLA`'s `eps` (1e-15) as a one-element float32 tensor (same values in PyTorch); how many.

    For the ONNX export: the exporter's optimizer (onnxscript 0.7, no-op rule `x + c` for a scalar `|c| <= 1e-8`)
    drops the scalar `+ eps` of the attention's division, which is then 0 / 0 = NaN wherever a ReLU'd query is all
    zero (onnxruntime and TensorRT alike). It keeps a non-scalar constant.
    """
    n = 0
    for m in net.modules():
        if type(m).__name__ == "LiteMLA" and isinstance(m.eps, float):
            m.eps = torch.full((1,), m.eps, device=device)
            n += 1
    return n


def wide_points(gm: fx.GraphModule, wide: int = WIDE) -> int:
    """Fused linear 1x1 convs from >= `wide` channels  ==>  `point` (cuDNN); returns how many.

    These are the projections of the wide MBConv expansions (folded weights up to 150, biases up to 580): the fused
    kernels accumulate their sum in fp16 and lose it there (L2 mask error 0.20 mean / 0.48 p99 of the golden's
    median, against 0.016 / 0.046 with float32 accumulation; `portable` 0.013 / 0.048).
    """
    n = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target not in {ops.conv_fused, ops.pw}:
            continue
        a: tuple[Any, ...] = node.args
        w = attr(gm, a[1])
        plain = node.target is ops.pw or (list(a[4]) == [0, 0] and list(a[5]) == [1, 1])  # no padding, stride 1
        linear = plain and a[4 if node.target is ops.pw else 6] == 0
        if w is None or w.shape[2:] != (1, 1) or w.shape[1] < wide or not linear:
            continue
        with gm.graph.inserting_before(node):
            new = gm.graph.call_function(point, (a[0], a[1], a[2], a[3]))
        node.replace_all_uses_with(new)
        gm.graph.erase_node(node)
        n += 1
    gm.graph.lint()
    gm.recompile()
    return n


def embed_points(pe: PromptEncoder, points: torch.Tensor, labels: torch.Tensor, *, pad: bool) -> torch.Tensor:
    """Upstream `PromptEncoder._embed_points` with `torch.where` for its boolean-mask writes (same values).

    A masked write reads the mask back on the host, which a CUDA graph capture refuses.
    """
    points = points + 0.5  # shift to the pixel centre
    if pad:
        points = torch.cat([points, points.new_zeros((len(points), 1, 2))], dim=1)
        labels = torch.cat([labels, labels.new_full((len(labels), 1), -1)], dim=1)
    emb = pe.pe_layer.forward_with_coords(points, pe.input_image_size)
    lab = labels[..., None]
    zero = emb.new_zeros(())
    kinds = [pe.not_a_point_embed, *pe.point_embeddings[:2]]  # labels -1, 0, 1
    emb = torch.where(lab == -1, zero, emb)
    for label, kind in zip((-1, 0, 1), kinds, strict=True):
        emb = emb + torch.where(lab == label, cast("torch.nn.Embedding", kind).weight, zero)
    return emb


def decode(  # noqa: PLR0913 - one argument per prompt kind
    pe: PromptEncoder,
    decoder: MaskDecoder,
    embedding: torch.Tensor,
    *,
    points: torch.Tensor | None,
    labels: torch.Tensor | None,
    boxes: torch.Tensor | None,
    masks: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream `PromptEncoder.forward` + `MaskDecoder.predict_masks`: all four masks `[P, 4, h, w]` and scores.

    Prompts in the 1024 frame. Index 0 is the single-mask output, 1-3 the multimask output.
    """
    parts = []
    if points is not None and labels is not None:
        parts.append(embed_points(pe, points, labels, pad=boxes is None))
    if boxes is not None:
        parts.append(pe._embed_boxes(boxes))  # noqa: SLF001 - upstream's own forward calls it
    n = len(parts[0]) if parts else 1 if masks is None else len(masks)
    sparse = torch.cat(parts, dim=1) if parts else embedding.new_empty((n, 0, pe.embed_dim))
    if masks is None:
        dense = pe.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(n, -1, *pe.image_embedding_size)
    else:
        dense = pe._embed_masks(masks)  # noqa: SLF001 - upstream's own forward calls it
    return decoder.predict_masks(
        image_embeddings=embedding,
        image_pe=pe.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
    )
