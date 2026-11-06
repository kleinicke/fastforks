"""Inference fast paths for the transformers Mask2Former / OneFormer networks, patched into their modules in place.

`MaskFormer(model, dtype, kernels=..., task_inputs=...)` takes a `Mask2FormerForUniversalSegmentation` or
`OneFormerForUniversalSegmentation` (Swin backbone, multi-scale deformable attention pixel decoder, masked-attention
transformer decoder) and runs it on one normalised image padded to a multiple of 32. The transformers modules and
weights stay; only how some of them compute changes (the same math in another association or precision):

- Swin layers (one image, any size): LayerNorm + pad + cyclic shift + window partition is one gather through an index
  table (padding reads a zero row: transformers pads after the LayerNorm), window reverse + shift back + crop +
  residual is the inverse gather. Windows are ordered interior first: only the border windows (last window row and
  column) can mix shift regions, so the interior runs SDPA with the relative-position bias alone and only the border
  windows add the shift mask (-100, as transformers). q / k / v are one GEMM. Index tables are tiny and cached for the
  last few feature-map sizes.
- OneFormer: the task token of the fixed task is computed once in float32 (transformers casts the token ids to the
  model dtype first, which rounds them in fp16); the query transformer (on the 1/32 features) stays float32 inside an
  fp16 model, its `nn.MultiheadAttention`s through SDPA; the decoder layers' "unmask fully masked rows" is a
  `masked_fill_` (transformers indexes with `torch.where`, a host sync).
- `kernels` (fp16 on CUDA, `fvm_seg`): the Swin LayerNorm + gather and the inverse gather + residual + LayerNorm each
  in one kernel, and the pixel decoder's multi-scale deformable self-attention after its projections in one kernel
  that takes each query's reference point from its position (transformers' pixel decoders never pad: valid ratios 1),
  softmax and sampling in float32. GEMMs accumulate in fp16.
"""

from __future__ import annotations

import contextlib
import functools
import sys
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
import torch.nn.functional as F

from fastforks._core.errors import BackendUnavailableError
from fastforks._core.half import fp16_accumulate
from fastforks._core.kernels import note_fallback
from fastforks._core.patch import base_type, override
from fastforks.kernels import fvm_seg

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["MaskFormer"]

_Hf = Any  # transformers modules: attribute access the type checker cannot follow


class _Geom(NamedTuple):
    """Window layout of one Swin layer type on one feature-map size."""

    windows: int
    rows: int  # windows x window area
    interior: int  # windows that need no shift mask (they come first)
    idx: torch.Tensor  # [rows] source token per window row, `tokens` (the zero row) for padding
    idx_k: torch.Tensor  # the same as int32, -1 for padding (kernel)
    inv: torch.Tensor  # [tokens] window row per token
    inv_k: torch.Tensor
    mask: torch.Tensor | None  # [border windows, T, T] shift mask (0 / -100), float32


def _interior_first(nh: int, nw: int) -> torch.Tensor:
    """Window indices of an `nh x nw` grid: interior windows first, then the last column, then the last row.

    Each part is row-major: a stable sort by "is border", without a sort op (the ONNX exporter has no stable sort).
    """
    k = torch.arange((nh - 1) * (nw - 1))
    interior = k // max(nw - 1, 1) * nw + k % max(nw - 1, 1)
    last_col = torch.arange(nh - 1) * nw + nw - 1
    last_row = (nh - 1) * nw + torch.arange(nw)
    return torch.cat([interior, last_col, last_row])


@functools.lru_cache(maxsize=16)
def _geom(h: int, w: int, ws: int, s: int, device: torch.device) -> _Geom:
    hp, wp = -(-h // ws) * ws, -(-w // ws) * ws
    nh, nw = hp // ws, wp // ws
    gy, gx = torch.meshgrid(torch.arange(nh), torch.arange(nw), indexing="ij")
    wy, wx = gy.flatten(), gx.flatten()  # no comprehension / nonzero: dynamo compiles this fullgraph
    if s > 0:
        order = _interior_first(nh, nw)
        interior = (nh - 1) * (nw - 1)
    else:
        order, interior = torch.arange(nh * nw), nh * nw
    wy, wx = wy[order], wx[order]
    iy, ix = torch.arange(ws).repeat_interleave(ws), torch.arange(ws).repeat(ws)
    y, x = wy[:, None] * ws + iy[None], wx[:, None] * ws + ix[None]  # shifted padded-grid coordinates [windows, T]
    sy, sx = (y + s) % hp, (x + s) % wp  # torch.roll(-s): out[i] = in[i + s]
    valid = ((sy < h) & (sx < w)).flatten()
    src = (sy * w + sx).flatten()
    n, rows = h * w, nh * nw * ws * ws
    # max(n + 1, rows): TensorRT's ScatterElements wants indices no longer than the data (padded windows: rows > n)
    inv = torch.zeros(max(n + 1, rows), dtype=torch.long)
    inv = inv.scatter_(0, torch.where(valid, src, n), torch.arange(rows))[:n]
    mask = None
    if s > 0 and interior < nh * nw:  # transformers `get_attn_mask` regions, border windows only
        reg = ((y >= hp - ws).long() + (y >= hp - s).long()) * 3 + (x >= wp - ws).long() + (x >= wp - s).long()
        rb = reg[interior:]
        mask = torch.where(rb[:, None, :] != rb[:, :, None], -100.0, 0.0).to(device)
    return _Geom(
        windows=nh * nw,
        rows=rows,
        interior=interior,
        idx=torch.where(valid, src, n).to(device),
        idx_k=torch.where(valid, src, -1).to(device, torch.int32),
        inv=inv.to(device),
        inv_k=inv.to(device, torch.int32),
        mask=mask,
    )


def _grid(n: int, per_block: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return (min(-(-n // per_block), 65535 * 8), 1, 1), (256, 1, 1)


def _swin_layer(  # the signature of transformers' `SwinLayer.forward`
    self: _Hf,
    hidden_states: torch.Tensor,
    input_dimensions: tuple[int, int],
    always_partition: bool = False,  # noqa: FBT001, FBT002
    **kw: _Hf,
) -> tuple[torch.Tensor, None]:
    """`SwinLayer.forward` (one image): gathers instead of pad / roll / partition, SDPA on the windows."""
    h, w = input_dimensions
    ws, s = self.ff_window, self.ff_shift
    b, n, c = hidden_states.shape
    if b != 1 or (min(h, w) <= ws and not always_partition):
        out = base_type(self).forward(self, hidden_states, input_dimensions, always_partition, **kw)
        self.window_size, self.shift_size = ws, s  # transformers clamps them on small maps
        return out
    geom = _geom.__wrapped__ if torch.compiler.is_compiling() else _geom  # dynamo warns on lru_cache
    g = geom(h, w, ws, s, hidden_states.device)
    a = self.attention
    hd = a.head_dim
    nh = c // hd
    t = ws * ws
    ln1, ln2 = self.layernorm_before, self.layernorm_after
    x = hidden_states.reshape(n, c)
    fused = self.ff_kernels and x.is_cuda and x.dtype == torch.float16
    if fused:
        win = x.new_empty((g.rows, c))
        args = (x, g.idx_k, ln1.weight, ln1.bias, win, g.rows, c, float(ln1.eps))
        fvm_seg.load(x.device).launch("ln_gather_h", *_grid(g.rows, 8), *args)
    else:
        xl = F.layer_norm(x, (c,), ln1.weight, ln1.bias, ln1.eps)
        win = torch.cat([xl, xl.new_zeros(1, c)]).index_select(0, g.idx)
    qkv = F.linear(win, a.ff_qkv_w, a.ff_qkv_b).view(g.windows, t, 3, nh, hd).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    rpb, ni = a.ff_rpb, g.interior
    if g.mask is None:
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=rpb.expand(g.windows, -1, -1, -1), scale=a.scaling)
    else:
        bm = rpb + g.mask[:, None].to(x.dtype)
        o = F.scaled_dot_product_attention(q[ni:], k[ni:], v[ni:], attn_mask=bm, scale=a.scaling)
        if ni:
            oi = F.scaled_dot_product_attention(
                q[:ni], k[:ni], v[:ni], attn_mask=rpb.expand(ni, -1, -1, -1), scale=a.scaling
            )
            o = torch.cat([oi, o])
    o = F.linear(o.transpose(1, 2).reshape(g.rows, c), a.o_proj.weight, a.o_proj.bias)
    if fused:
        hs, y = x.new_empty((n, c)), x.new_empty((n, c))
        args = (o, g.inv_k, x, ln2.weight, ln2.bias, hs, y, n, c, float(ln2.eps))
        fvm_seg.load(x.device).launch("gather_add_ln_h", *_grid(n, 8), *args)
    else:
        hs = x + o.index_select(0, g.inv)
        y = F.layer_norm(hs, (c,), ln2.weight, ln2.bias, ln2.eps)
    return (self.mlp(y) + hs).view(b, n, c), None


@functools.lru_cache(maxsize=16)
def _msda_meta(shapes: tuple[tuple[int, int], ...], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    starts = [sum(h * w for h, w in shapes[:i]) for i in range(len(shapes))]
    flat = [v for hw in shapes for v in hw]
    return torch.tensor(flat, dtype=torch.int32, device=device), torch.tensor(starts, dtype=torch.int32, device=device)


def _msda_torch(
    self: _Hf,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None,
    position_embeddings: torch.Tensor | None,
    spatial_shapes_list: Sequence[tuple[int, int]] | None,
    **kw: _Hf,
) -> tuple[torch.Tensor, torch.Tensor]:
    """OneFormer's `MultiscaleDeformableAttention.forward` without its tensor shape check (2-d reference points)."""
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    q = hidden_states if position_embeddings is None else hidden_states + position_embeddings
    b, s, c = encoder_hidden_states.shape
    nh, levels, points = self.n_heads, self.n_levels, self.n_points
    value = self.value_proj(encoder_hidden_states)
    if kw.get("attention_mask") is not None:
        value = value.masked_fill(kw["attention_mask"][..., None], 0.0)
    value = value.view(b, s, nh, c // nh)
    nq = q.shape[1]
    off = self.sampling_offsets(q).view(b, nq, nh, levels, points, 2)
    w = torch.softmax(self.attention_weights(q).view(b, nq, nh, levels * points), -1).view(b, nq, nh, levels, points)
    ref, shapes = kw["reference_points"], kw["spatial_shapes"]
    norm = torch.stack([shapes[..., 1], shapes[..., 0]], -1)
    loc = ref[:, :, None, :, None, :] + off / norm[None, None, None, :, None, :]
    out = self.ff_msda_torch(value, list(spatial_shapes_list or ()), loc, w)
    return self.output_proj(out), w


def _msda(
    self: _Hf,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None = None,
    position_embeddings: torch.Tensor | None = None,
    spatial_shapes_list: Sequence[tuple[int, int]] | None = None,
    **kw: _Hf,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The pixel decoder's `MultiscaleDeformableAttention.forward` (self-attention, no padding) as one kernel."""
    if (
        not self.ff_kernels
        or hidden_states.dtype != torch.float16
        or not hidden_states.is_cuda
        or encoder_hidden_states is None
        or spatial_shapes_list is None
        or encoder_hidden_states.shape != hidden_states.shape
    ):
        if self.ff_kernels:
            note_fallback("fvm_seg", "deformable attention other than unpadded self-attention")
        if type(self).__name__.startswith("OneFormer"):  # its shape check on a tensor breaks the dynamo graph
            return _msda_torch(
                self, hidden_states, encoder_hidden_states, position_embeddings, spatial_shapes_list, **kw
            )
        return base_type(self).forward(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            position_embeddings=position_embeddings,
            spatial_shapes_list=spatial_shapes_list,
            **kw,
        )
    q = hidden_states if position_embeddings is None else hidden_states + position_embeddings
    b, s, c = encoder_hidden_states.shape
    nh, levels, points = self.n_heads, self.n_levels, self.n_points
    value = self.value_proj(encoder_hidden_states)
    off = self.sampling_offsets(q)
    logit = self.attention_weights(q)
    shapes, starts = _msda_meta(tuple((int(h), int(w)) for h, w in spatial_shapes_list), value.device)
    out = torch.empty_like(value)
    n = b * s * nh * (c // nh // 2)
    args = (value, off, logit, shapes, starts, out, b, s, nh, c // nh, levels, points)
    fvm_seg.load(value.device).launch("msda_self_h", *_grid(n, 256), *args)
    return self.output_proj(out), None


def _oneformer_layer(  # noqa: PLR0913, PLR0917 - the signature of transformers' layer
    self: _Hf,
    index: int,
    output: torch.Tensor,
    multi_stage_features: list[torch.Tensor],
    multi_stage_positional_embeddings: list[torch.Tensor],
    attention_mask: torch.Tensor,
    query_embeddings: torch.Tensor | None = None,
    output_attentions: bool | None = False,  # noqa: FBT001, FBT002
) -> tuple[torch.Tensor, ...]:
    """The transformers `OneFormerTransformerDecoderLayer.forward`, the full-row unmasking as `masked_fill_`."""
    level = index % self.num_feature_levels
    attention_mask.masked_fill_((attention_mask.sum(-1) == attention_mask.shape[-1])[..., None], value=False)
    output, cross = self.cross_attn(
        output,
        multi_stage_features[level],
        memory_mask=attention_mask,
        memory_key_padding_mask=None,
        pos=multi_stage_positional_embeddings[level],
        query_pos=query_embeddings,
    )
    output, self_w = self.self_attn(output, output_mask=None, output_key_padding_mask=None, query_pos=query_embeddings)
    output = self.ffn(output)
    return (output, self_w, cross) if output_attentions else (output,)


def _mha_sdpa(
    self: _Hf, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args: _Hf, **kw: _Hf
) -> tuple[torch.Tensor, None]:
    """`nn.MultiheadAttention` (sequence first, no masks) through SDPA: no `[heads, queries, keys]` score matrix."""
    if args or kw.get("key_padding_mask") is not None or kw.get("attn_mask") is not None or self.batch_first:
        return base_type(self).forward(self, query, key, value, *args, **kw)
    lq, b, e = query.shape
    nh = self.num_heads
    d = e // nh
    w, bias = self.in_proj_weight, self.in_proj_bias
    q = F.linear(query, w[:e], bias[:e]).view(lq, b, nh, d).permute(1, 2, 0, 3)
    k = F.linear(key, w[e : 2 * e], bias[e : 2 * e]).view(key.shape[0], b, nh, d).permute(1, 2, 0, 3)
    v = F.linear(value, w[2 * e :], bias[2 * e :]).view(value.shape[0], b, nh, d).permute(1, 2, 0, 3)
    o = F.scaled_dot_product_attention(q, k, v)
    return F.linear(o.permute(2, 0, 1, 3).reshape(lq, b, e), self.out_proj.weight, self.out_proj.bias), None


def _cast(o: _Hf, src: torch.dtype, dst: torch.dtype) -> _Hf:
    if isinstance(o, torch.Tensor):
        return o.to(dst) if o.dtype == src else o
    if isinstance(o, (list, tuple)):
        return type(o)(_cast(v, src, dst) for v in o)
    if isinstance(o, dict):
        return {k: _cast(v, src, dst) for k, v in o.items()}
    return o


def _float32_island(mod: torch.nn.Module, dtype: torch.dtype) -> None:
    """Run `mod` in float32 inside a `dtype` model (inputs cast up, outputs cast back)."""
    mod.float()
    mod.register_forward_pre_hook(
        lambda _m, a, k: (_cast(a, dtype, torch.float32), _cast(k, dtype, torch.float32)), with_kwargs=True
    )
    mod.register_forward_hook(lambda _m, _a, o: _cast(o, torch.float32, dtype))


class _TaskToken(torch.nn.Module):
    """OneFormer's task MLP output for the fixed task (computed once in float32)."""

    token: torch.Tensor

    def __init__(self, token: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("token", token, persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.token.expand(inputs.shape[0], -1)


def _patch(m: _Hf, fn: _Hf, **attrs: _Hf) -> None:
    for k, v in attrs.items():
        setattr(m, k, v)
    override(m, fn)


def _even(m: _Hf, what: str, n: int) -> None:
    """The fvm_seg kernels read fp16 pairs (half2): checked on the host instead of misreading odd sizes."""
    if n % 2:
        msg = f"fvm_seg needs an even {what}, {type(m).__name__} has {n}; use mode='portable'"
        raise BackendUnavailableError(msg)


def _patch_backbone_and_decoder(model: _Hf, *, kernels: bool) -> None:
    for m in model.modules():
        if type(m).__name__ == "SwinLayer":
            if kernels:
                _even(m, "channel count", int(m.layernorm_before.normalized_shape[0]))
            a = m.attention
            qkv_b = [a.q_proj.bias, a.k_proj.bias, a.v_proj.bias]
            bufs = {
                "ff_qkv_w": torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]),
                "ff_qkv_b": torch.cat(qkv_b) if all(v is not None for v in qkv_b) else None,
                "ff_rpb": a.relative_position_bias(),  # [1, heads, T, T], weights only
            }
            for k, v in bufs.items():
                a.register_buffer(k, None if v is None else v.detach(), persistent=False)
            _patch(m, _swin_layer, ff_window=int(m.window_size), ff_shift=int(m.shift_size), ff_kernels=kernels)
        elif type(m).__name__.endswith("PixelDecoderEncoderMultiscaleDeformableAttention"):
            if kernels:
                _even(m, "head dim", int(m.d_model) // int(m.n_heads))
            torch_fn = getattr(sys.modules[base_type(m).__module__], "multi_scale_deformable_attention", None)
            _patch(m, _msda, ff_kernels=kernels, ff_msda_torch=torch_fn)  # the library's own pure-torch kernel


def _patch_oneformer(model: _Hf, dtype: torch.dtype) -> None:
    qt = model.model.transformer_module.decoder.query_transformer
    for m in qt.modules():
        if isinstance(m, torch.nn.MultiheadAttention) and m._qkv_same_embed_dim:  # noqa: SLF001
            _patch(m, _mha_sdpa)
    for m in model.modules():
        if type(m).__name__ == "OneFormerTransformerDecoderLayer":
            _patch(m, _oneformer_layer)
    if dtype != torch.float32:
        _float32_island(qt, dtype)


def _fp16_accumulate(*, enabled: bool) -> contextlib.AbstractContextManager[None]:
    return fp16_accumulate() if enabled else contextlib.nullcontext()


class MaskFormer(torch.nn.Module):
    """A transformers Mask2Former / OneFormer network prepared for inference (the modules are patched in place).

    Args:
        model: `Mask2FormerForUniversalSegmentation` or `OneFormerForUniversalSegmentation`, float32, eval mode.
        dtype: Weights and activations (float32, or float16 on CUDA); the OneFormer query transformer stays float32.
        kernels: Use the `fvm_seg` kernels and fp16-accumulate GEMMs (float16 on CUDA only).
        task_inputs: OneFormer: the task's token ids (CLIP tokenizer, zero-padded to the config's `task_seq_len`).
    """

    task: torch.Tensor

    def __init__(
        self, model: _Hf, dtype: torch.dtype, *, kernels: bool, task_inputs: Sequence[int] | None = None
    ) -> None:
        """Patch `model` and cast it to `dtype` (in place; the weights stay shared)."""
        super().__init__()
        self.model, self.ff_dtype, self.kernels = model, dtype, kernels
        self.oneformer = hasattr(model.model, "task_encoder")
        if self.oneformer:
            if not task_inputs:
                msg = "OneFormer needs task_inputs"
                raise ValueError(msg)
            task = torch.zeros((1, model.config.task_seq_len), dtype=torch.long)
            task[0, : len(task_inputs)] = torch.tensor(list(task_inputs))
            self.register_buffer("task", task, persistent=False)
            with torch.no_grad():
                token = model.model.task_encoder(task.float())
            model.model.task_encoder = _TaskToken(token)
        _patch_backbone_and_decoder(model, kernels=kernels)
        model.to(dtype)
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.to(memory_format=torch.channels_last)
        if self.oneformer:
            _patch_oneformer(model, dtype)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalised image `[1, 3, H, W]` (sides multiples of 32) -> mask and class logits (model dtype).

        Mask logits are `[1, Q, H/4, W/4]`, class logits `[1, Q, classes + 1]` (the last class is "no object").
        """
        x = x.to(self.ff_dtype).contiguous(memory_format=torch.channels_last)
        kw = {"task_inputs": self.task} if self.oneformer else {}
        with _fp16_accumulate(enabled=self.kernels and x.is_cuda and self.ff_dtype == torch.float16):
            o = self.model(pixel_values=x, **kw)
        return o.masks_queries_logits, o.class_queries_logits
