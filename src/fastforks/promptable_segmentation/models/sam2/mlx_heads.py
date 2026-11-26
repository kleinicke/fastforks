"""SAM 2's mask decoder and video memory written in MLX, channels last (`mlx_net` is the image encoder).

- `MlxDecoder` is `MaskDecoder.predict_masks` in float32 (the two-way transformer, the upscaling with the
  high-resolution maps, the hypernetwork product and the score heads). It replaces that method on the upstream module,
  so the prompt encoder, the choice among the masks and the tracker's heads stay the upstream PyTorch code around it.
- `MlxMemoryAttention` / `MlxMemoryEncoder` are the video modules in the run dtype, the tokens' residual stream
  float32 as in the image encoder. `MlxTracker` is `video.Tracker` with them and a memory bank of MLX arrays.
- `mlx_net.Twins` keeps the MLX array of every feature map handed to PyTorch code: when such a map comes back (the
  decoder after the encoder, the heads after memory attention) it is not converted again.
"""

# ruff: noqa: ANN401 - the upstream torch modules

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from fastforks._core.mlx import mx, to_mx, to_torch
from fastforks.shared.mlx_vit import gelu

from .mlx_net import Twins, arr, layer_norm, linear
from .net.sam2_utils import MLP, LayerNorm2d
from .video import FILL_HOLE_AREA, HOLE_SCORE, MEM_BIAS, MEM_DIM, MEM_SCALE, Tracker

if TYPE_CHECKING:
    from collections.abc import Callable

    from .net.mask_decoder import MaskDecoder
    from .net.memory_attention import MemoryAttention
    from .net.memory_encoder import CXBlock, MemoryEncoder
    from .net.prompt_encoder import PromptEncoder
    from .video import VideoModules

F32 = mx.float32
QUERIES = 512  # queries per product with the memory: measured faster than all at once, and a lower peak


def _conv1x1(conv: Any, dtype: mx.Dtype) -> Callable[[mx.array], mx.array]:
    """A 1 x 1 `Conv2d` as one `addmm` on channels-last maps."""
    w, b = arr(conv.weight[:, :, 0, 0].T, dtype), arr(conv.bias, dtype)
    return lambda x: mx.addmm(b, x, w)


def _mlp(mlp: Any, dtype: mx.Dtype) -> Callable[[mx.array], mx.array]:
    """Upstream `MLP` (ReLU between the layers, an optional sigmoid at the end)."""
    if not isinstance(mlp.act, torch.nn.ReLU):
        msg = f"SAM 2 on MLX: MLP activation {type(mlp.act).__name__} is not written"
        raise NotImplementedError(msg)
    layers = [linear(layer, dtype) for layer in mlp.layers]

    def run(x: mx.array) -> mx.array:
        for layer in layers[:-1]:
            x = mx.maximum(layer(x), 0)
        x = layers[-1](x)
        return mx.sigmoid(x) if mlp.sigmoid_output else x

    return run


def _token_norm(norm: Any) -> Callable[[mx.array], mx.array]:
    """A `LayerNorm` of a few tokens in float32 (MLX's own: fast on few rows)."""
    g, b, eps = arr(norm.weight, F32), arr(norm.bias, F32), float(norm.eps)
    return lambda x: mx.fast.layer_norm(x, g, b, eps)


def _attention(attn: Any, dtype: mx.Dtype) -> Callable[[mx.array, mx.array, mx.array], mx.array]:
    """Upstream `Attention.forward(q, k, v)` on `[B, N, C]` tokens."""
    q, k, v, out = (linear(layer, dtype) for layer in (attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj))
    heads = attn.num_heads
    scale = (attn.internal_dim // heads) ** -0.5

    def split(x: mx.array) -> mx.array:
        return x.reshape(x.shape[0], x.shape[1], heads, -1).transpose(0, 2, 1, 3)

    def run(qx: mx.array, kx: mx.array, vx: mx.array) -> mx.array:
        y = mx.fast.scaled_dot_product_attention(split(q(qx)), split(k(kx)), split(v(vx)), scale=scale)
        return out(y.transpose(0, 2, 1, 3).reshape(qx.shape[0], qx.shape[1], -1))

    return run


def _up(conv: Any) -> Callable[[mx.array], mx.array]:
    """`ConvTranspose2d(kernel 2, stride 2)` on `[B, H, W, C]` as one GEMM (each input pixel writes its 2 x 2)."""
    cin, cout = conv.weight.shape[:2]
    w = arr(conv.weight.permute(0, 2, 3, 1).reshape(cin, 4 * cout), F32)
    b = arr(conv.bias, F32)

    def run(x: mx.array) -> mx.array:
        n, h, wd, _ = x.shape
        y = (x @ w).reshape(n, h, wd, 2, 2, cout).transpose(0, 1, 3, 2, 4, 5)
        return y.reshape(n, 2 * h, 2 * wd, cout) + b

    return run


class MlxDecoder:
    """`MaskDecoder.predict_masks` of one loaded SAM 2 on MLX, float32."""

    def __init__(self, dec: MaskDecoder, twins: Twins) -> None:
        """Copy the decoder's weights; `twins` finds the encoder's maps again."""
        ups = dec.output_upscaling
        if not dec.use_high_res_features or not isinstance(ups[2], torch.nn.GELU):
            msg = "SAM 2 on MLX: a mask decoder without high-resolution features or GELU upscaling is not written"
            raise NotImplementedError(msg)
        self.twins = twins
        self.scored = bool(dec.pred_obj_scores)
        tokens = [dec.obj_score_token.weight] if self.scored else []
        self.tokens = arr(torch.cat([*tokens, dec.iou_token.weight, dec.mask_tokens.weight]), F32)
        self.masks = dec.num_mask_tokens
        tr: Any = dec.transformer
        self.layers = [self._layer(layer) for layer in tr.layers]
        self.final = _attention(tr.final_attn_token_to_image, F32)
        self.final_norm = _token_norm(tr.norm_final_attn)
        self.up1, self.up_norm, self.up2 = _up(ups[0]), layer_norm(ups[1], F32), _up(ups[3])
        self.hyper = [_mlp(m, F32) for m in dec.output_hypernetworks_mlps]
        self.iou = _mlp(dec.iou_prediction_head, F32)
        head = dec.pred_obj_score_head if self.scored else None
        self.score = None if head is None else _mlp(head, F32) if isinstance(head, MLP) else linear(head, F32)
        self.pe: mx.array | None = None

    @staticmethod
    def _layer(blk: Any) -> Callable[[mx.array, mx.array, mx.array, mx.array], tuple[mx.array, mx.array]]:
        """Upstream `TwoWayAttentionBlock.forward`."""
        self_attn, to_image, to_token = (
            _attention(a, F32) for a in (blk.self_attn, blk.cross_attn_token_to_image, blk.cross_attn_image_to_token)
        )
        n1, n2, n3 = _token_norm(blk.norm1), _token_norm(blk.norm2), _token_norm(blk.norm3)
        n4, mlp, skip_pe = layer_norm(blk.norm4, F32), _mlp(blk.mlp, F32), blk.skip_first_layer_pe

        def run(q: mx.array, k: mx.array, q_pe: mx.array, k_pe: mx.array) -> tuple[mx.array, mx.array]:
            if skip_pe:
                q = self_attn(q, q, q)
            else:
                qq = q + q_pe
                q = q + self_attn(qq, qq, q)
            q = n1(q)
            q = n2(q + to_image(q + q_pe, k + k_pe, k))
            q = n3(q + mlp(q))
            return q, n4(k + to_token(k + k_pe, q + q_pe, q))

        return run

    def __call__(  # noqa: PLR0913, PLR0917 - upstream `predict_masks` arguments
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,  # noqa: FBT001 - upstream signature
        high_res_features: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Upstream `predict_masks`: all masks `[B, 4, 4h, 4w]`, their scores, the mask tokens, the object score."""
        if high_res_features is None:
            msg = "SAM 2 on MLX: the mask decoder needs the high-resolution features"
            raise ValueError(msg)
        n = sparse_prompt_embeddings.shape[0]
        if not repeat_image and image_embeddings.shape[0] != n:
            msg = f"SAM 2 mask decoder: {image_embeddings.shape[0]} images for {n} prompts"
            raise ValueError(msg)
        if self.pe is None:
            self.pe = to_mx(image_pe.float().permute(0, 2, 3, 1)).reshape(1, -1, image_pe.shape[1])
        src = self.twins.array(image_embeddings)
        _, h, w, c = src.shape
        dense = dense_prompt_embeddings
        if dense.stride(2) == 0 and dense.stride(3) == 0:  # one vector per prompt (no mask prompt): not a map
            src = src + to_mx(dense[:, :, 0, 0].float())[:, None, None, :]
        else:
            src = src + to_mx(dense.float().permute(0, 2, 3, 1))
        src = mx.broadcast_to(src, (n, h, w, c)).reshape(n, h * w, c)
        sparse = to_mx(sparse_prompt_embeddings.float())
        tokens = mx.concatenate([mx.broadcast_to(self.tokens, (n, *self.tokens.shape)), sparse], axis=1)
        q, k = tokens, src
        for layer in self.layers:
            q, k = layer(q, k, tokens, self.pe)
        q = self.final_norm(q + self.final(q + tokens, k + self.pe, k))
        s = int(self.scored)
        mask_tokens = q[:, s + 1 : s + 1 + self.masks]
        s0, s1 = (self.twins.array(f) for f in high_res_features)
        up = gelu(self.up_norm(self.up1(k.reshape(n, h, w, c)) + s1))
        up = gelu(self.up2(up) + s0)
        hyper = mx.stack([mlp(mask_tokens[:, i]) for i, mlp in enumerate(self.hyper)], axis=1)  # [B, 4, C/8]
        masks = up.reshape(n, 16 * h * w, -1) @ hyper.transpose(0, 2, 1)
        masks = masks.reshape(n, 4 * h, 4 * w, self.masks)
        iou = self.iou(q[:, s])
        score = self.score(q[:, 0]) if self.score is not None else mx.full((n, 1), 10.0)
        mx.eval(masks, iou, mask_tokens, score)
        return to_torch(masks).permute(0, 3, 1, 2), to_torch(iou), to_torch(mask_tokens), to_torch(score)


def _rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Upstream `apply_rotary_enc` on `[..., N, D]`: the pairs of `x` turned by the angles `[N, D / 2]`."""
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    return mx.stack([a * cos - b * sin, a * sin + b * cos], axis=-1).reshape(x.shape)


class MlxMemoryAttention:
    """Upstream `MemoryAttention` (RoPE self- and cross-attention layers) on MLX, batch first."""

    def __init__(self, att: MemoryAttention, dtype: mx.Dtype) -> None:
        """Copy the weights; attention and MLP in `dtype`, the tokens' stream float32."""
        self.dtype = dtype
        cis = att.layers[0].self_attn.freqs_cis  # [tokens, D / 2] complex, the same for every layer
        self.cos, self.sin = arr(cis.real, dtype), arr(cis.imag, dtype)
        self.layers = [self._layer(layer, dtype) for layer in att.layers]
        self.norm = layer_norm(att.norm, F32)
        self.at_input = bool(att.pos_enc_at_input)

    def _layer(self, layer: Any, dtype: mx.Dtype) -> Callable[[mx.array, mx.array, mx.array, int], mx.array]:
        """Upstream `MemoryAttentionLayer.forward` (SAM 2's settings: position encoding at the cross keys only)."""
        if layer.pos_enc_at_attn or layer.pos_enc_at_cross_attn_queries or not layer.pos_enc_at_cross_attn_keys:
            msg = "SAM 2 on MLX: this memory-attention position-encoding setting is not written"
            raise NotImplementedError(msg)
        if layer.activation_str != "relu" or layer.self_attn.num_heads != 1 or layer.cross_attn_image.num_heads != 1:
            msg = "SAM 2 on MLX: memory attention is written for one head and ReLU"
            raise NotImplementedError(msg)
        sa, ca = layer.self_attn, layer.cross_attn_image
        sq, sk, sv, so = (linear(m, dtype) for m in (sa.q_proj, sa.k_proj, sa.v_proj, sa.out_proj))
        cq, ck, cv, co = (linear(m, dtype) for m in (ca.q_proj, ca.k_proj, ca.v_proj, ca.out_proj))
        n1, n2, n3 = (layer_norm(n, dtype) for n in (layer.norm1, layer.norm2, layer.norm3))
        fc1, fc2 = linear(layer.linear1, dtype), linear(layer.linear2, dtype)
        scale = sa.internal_dim**-0.5
        cos, sin = self.cos, self.sin

        def attend(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
            return mx.fast.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None], scale=scale)[:, 0]

        def run(x: mx.array, memory: mx.array, keys: mx.array, plain: int) -> mx.array:
            h = n1(x)
            x = x + so(attend(_rotate(sq(h), cos, sin), _rotate(sk(h), cos, sin), sv(h))).astype(F32)
            h = n2(x)
            k = ck(keys)
            b, n, d = k.shape
            turned = n - plain  # whole frames of tokens; the last `plain` keys (object pointers) are not rotated
            rot = _rotate(k[:, :turned].reshape(b, -1, cos.shape[0], d), cos, sin).reshape(b, turned, d)
            k = mx.concatenate([rot, k[:, turned:]], axis=1) if plain else rot
            # The values are a linear map of the 64-wide memory and the weights of a query sum to 1: the weighted
            # sum is taken over the memory itself (a quarter of the product) and projected after.
            q, kt = _rotate(cq(h), cos, sin) * scale, k.transpose(0, 2, 1)
            parts = [mx.softmax(q[:, i : i + QUERIES] @ kt, axis=-1) @ memory for i in range(0, q.shape[1], QUERIES)]
            x = x + co(cv(mx.concatenate(parts, axis=1))).astype(F32)
            return x + fc2(mx.maximum(fc1(n3(x)), 0)).astype(F32)

        return run

    def __call__(self, curr: mx.array, curr_pos: mx.array, memory: mx.array, memory_pos: mx.array, plain: int):  # noqa: ANN204
        """Float32 `curr` `[P, N, 256]` conditioned on `memory` `[P, M, 64]` (its last `plain` tokens not rotated).

        `curr_pos` `[N, 256]`, `memory_pos` `[M, 64]`; returns float32 `[P, N, 256]`.
        """
        if curr.shape[1] != self.cos.shape[0]:
            msg = f"SAM 2 on MLX: memory attention is built for {self.cos.shape[0]} tokens, got {curr.shape[1]}"
            raise ValueError(msg)
        x = curr + 0.1 * curr_pos if self.at_input else curr
        memory = memory.astype(self.dtype)
        keys = memory + memory_pos.astype(self.dtype)
        for layer in self.layers:
            x = layer(x, memory, keys, plain)
            mx.eval(x)  # a layer's attention weights (tokens x memory) do not outlive it
        return self.norm(x)


class MlxMemoryEncoder:
    """Upstream `MemoryEncoder` on MLX: the mask downsampler, the fuser's ConvNeXt blocks, the output projection."""

    def __init__(self, enc: MemoryEncoder, dtype: mx.Dtype) -> None:
        """Copy the weights (all in `dtype`)."""
        self.dtype = dtype
        mods = list(enc.mask_downsampler.encoder)
        self.down = []
        for conv, norm, act in zip(mods[0:-1:3], mods[1:-1:3], mods[2:-1:3], strict=True):
            if not isinstance(norm, LayerNorm2d) or not isinstance(act, torch.nn.GELU):
                msg = "SAM 2 on MLX: this mask downsampler is not written"
                raise NotImplementedError(msg)
            self.down.append(
                (
                    arr(conv.weight.permute(0, 2, 3, 1), dtype),
                    arr(conv.bias, dtype),
                    conv.stride[0],
                    conv.padding[0],
                    layer_norm(norm, dtype),
                )
            )
        self.down_out = _conv1x1(mods[-1], dtype)
        self.pix = _conv1x1(enc.pix_feat_proj, dtype)
        self.fuser = [self._cx(blk, dtype) for blk in enc.fuser.layers]
        if not isinstance(enc.fuser.proj, torch.nn.Identity) or isinstance(enc.out_proj, torch.nn.Identity):
            msg = "SAM 2 on MLX: this memory encoder's projections are not written"
            raise NotImplementedError(msg)
        self.out = _conv1x1(enc.out_proj, dtype)

    @staticmethod
    def _cx(blk: CXBlock, dtype: mx.Dtype) -> Callable[[mx.array], mx.array]:
        """Upstream `CXBlock.forward` (depthwise conv, norm, two linears, layer scale, residual)."""
        dw = blk.dwconv
        if dw.groups != dw.in_channels or blk.gamma is None:
            msg = "SAM 2 on MLX: a ConvNeXt block without depthwise conv or layer scale is not written"
            raise NotImplementedError(msg)
        w, pad = arr(dw.weight.permute(0, 2, 3, 1), dtype), dw.padding[0]
        b = arr(dw.bias, dtype) if dw.bias is not None else mx.zeros(1, dtype)
        norm, fc1 = layer_norm(blk.norm, dtype), linear(blk.pwconv1, dtype)
        # the layer scale folded into the second linear
        w2 = arr((blk.pwconv2.weight * blk.gamma[:, None]).T, dtype)
        b2 = arr(blk.pwconv2.bias * blk.gamma, dtype)
        groups = dw.in_channels

        def run(x: mx.array) -> mx.array:
            h = norm(mx.conv2d(x, w, padding=pad, groups=groups) + b)
            return x + mx.addmm(b2, gelu(fc1(h)), w2)

        return run

    def __call__(self, pix: mx.array, mask: mx.array) -> mx.array:
        """Features `[P, t, t, 256]` and the mask `[P, S, S, 1]` (already scaled) -> memory `[P, t, t, 64]`."""
        m = mask.astype(self.dtype)
        for w, b, stride, pad, norm in self.down:
            m = gelu(norm(mx.conv2d(m, w, stride=stride, padding=pad) + b))
        x = self.pix(pix.astype(self.dtype)) + self.down_out(m)
        for blk in self.fuser:
            x = blk(x)
        return self.out(x)


def fill_holes(low: mx.array, max_area: int = FILL_HOLE_AREA) -> mx.array:
    """`video.fill_holes` on `[P, h, w]` logits: the same label propagation, exact (integer labels in float32)."""
    p, h, w = low.shape
    bg = low <= 0
    big = float(h * w)

    def spread(x: mx.array, pad: float, pick: Callable[[mx.array, mx.array], mx.array]) -> mx.array:
        """3 x 3 minimum / maximum (`pick`) of every pixel's neighbourhood, the border padded with `pad`."""
        x = mx.pad(x, ((0, 0), (0, 0), (1, 1)), constant_values=pad)
        x = pick(pick(x[:, :, :-2], x[:, :, 1:-1]), x[:, :, 2:])
        x = mx.pad(x, ((0, 0), (1, 1), (0, 0)), constant_values=pad)
        return pick(pick(x[:, :-2], x[:, 1:-1]), x[:, 2:])

    lab = mx.where(bg, mx.arange(h * w, dtype=F32).reshape(1, h, w), big)
    for _ in range(max_area):
        lab = mx.where(bg, spread(lab, big, mx.minimum), big)
    lo = spread(lab, big, mx.minimum)
    hi = spread(mx.where(bg, lab, -1.0), -1.0, mx.maximum)
    n = p * h * w
    key = mx.arange(p, dtype=mx.int32).reshape(p, 1, 1) * (h * w) + mx.minimum(lab, big - 1).astype(mx.int32)
    key = mx.where(bg, key, n).reshape(-1)  # background pixels -> their label; the rest -> a dump slot
    count = mx.zeros(n + 1).at[key].add(mx.ones(n))
    touch = ((lo != lab) | (hi != lab)).reshape(-1).astype(F32)
    opened = mx.zeros(n + 1).at[key].add(touch) > 0
    hole = bg.reshape(-1) & (count[key] <= max_area) & ~opened[key]
    return mx.where(hole.reshape(low.shape), HOLE_SCORE, low)


class MlxTracker(Tracker):
    """`video.Tracker` with memory attention and the memory encoder on MLX; the bank holds MLX arrays `[P, N, 64]`."""

    def __init__(  # noqa: PLR0913, PLR0917 - the shared modules
        self,
        mods: VideoModules,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        no_mem: torch.Tensor,
        dtype_name: str,
        image_size: int,
        twins: Twins,
    ) -> None:
        """As `Tracker` (float32 PyTorch around the heads); the memory modules of `mods` move to MLX in `dtype_name`."""
        super().__init__(mods, prompt_encoder, mask_decoder, no_mem, torch.float32, image_size)
        dtype = getattr(mx, dtype_name)
        self.mx_dtype, self.twins = dtype, twins
        self.attention = MlxMemoryAttention(mods.memory_attention, dtype)
        self.encoder = MlxMemoryEncoder(mods.memory_encoder, dtype)
        del mods.memory_attention, mods.memory_encoder  # their weights live in MLX now
        self.mx_pos = arr(self.pos[:, 0], F32)  # [N, 256]
        self.mx_mem_pos = arr(self.mem_pos[:, 0], dtype)  # [N, 64]
        self.mx_tpos = arr(mods.maskmem_tpos_enc.reshape(-1, MEM_DIM), dtype)
        self.mx_no_obj = arr(mods.no_obj_embed_spatial.reshape(-1), dtype)

    def fill(self, low: torch.Tensor) -> torch.Tensor:
        """`fill_holes` of the low-resolution logits `[P, 1, h, w]` on MLX."""
        return to_torch(fill_holes(to_mx(low[:, 0].float())))[:, None]

    def encode(self, raw: torch.Tensor, high: torch.Tensor, obj: torch.Tensor, *, binarize: bool) -> Any:
        """Upstream `_encode_new_memory`: bfloat16 memory tokens `[P, N, 64]` (an MLX array)."""
        m = to_mx(high.float().permute(0, 2, 3, 1))
        m = (m > 0).astype(F32) if binarize else mx.sigmoid(m)
        feats = self.encoder(self.twins.array(raw), m * MEM_SCALE + MEM_BIAS)
        absent = to_mx((obj <= 0).float()).astype(feats.dtype)[:, :, None, None]  # [P, 1, 1, 1]
        feats = (feats + absent * self.mx_no_obj).reshape(feats.shape[0], -1, MEM_DIM).astype(mx.bfloat16)
        mx.async_eval(feats)
        return feats

    def condition(  # noqa: PLR0913, PLR0917 - the memory frames and the objects that use them
        self,
        raw: torch.Tensor,
        t: int,
        n_frames: int,
        bank: dict[int, tuple[Any, torch.Tensor]],
        memory: list[int] | None = None,
        objects: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Upstream `_prepare_memory_conditioned_features` (see `Tracker.condition`) on MLX."""
        p, c, h, w = raw.shape
        slots, ptr_frames, max_ptrs = self.plan(t, n_frames, memory)
        rows = None if objects is None else mx.array(objects.cpu().numpy())
        feats, poss = [], []
        for k, age in slots:
            encoded = bank[k][0]
            if encoded is None:
                msg = f"SAM 2 memory for frame {k} was released before it was read"
                raise RuntimeError(msg)
            feats.append((encoded if rows is None else encoded[rows]).astype(self.mx_dtype))
            poss.append(self.mx_mem_pos + self.mx_tpos[age - 1])
        ptrs, pos = self.pointers(bank, ptr_frames, t, max_ptrs, slice(None) if objects is None else objects)
        feats.append(to_mx(ptrs.float().permute(1, 0, 2)).astype(self.mx_dtype))
        poss.append(to_mx(pos.float()).astype(self.mx_dtype))
        curr = self.twins.array(raw)
        curr = mx.broadcast_to(curr, (p, h, w, c)).reshape(p, h * w, c)
        out = self.attention(curr, self.mx_pos, mx.concatenate(feats, axis=1), mx.concatenate(poss), ptrs.shape[0])
        return self.twins.tensor(out.reshape(p, h, w, c))
