"""SAM 2's prompt encoder, mask decoder and video memory written in MLX, channels last (`mlx_net`: the encoder).

No PyTorch in a call: the builders read the upstream modules' weights through `mlx_net.arr` (so a load replays them
from the conversion cache, `fastforks.shared.mlx_port`), the calls take and return MLX arrays.

- `MlxPrompts` is `PromptEncoder`: points and box corners are encoded on the host in numpy (a few tokens), a mask
  prompt by its two strided convolutions on MLX.
- `MlxDecoder` is `MaskDecoder` in float32 (the two-way transformer, the upscaling with the high-resolution maps, the
  hypernetworks and the score heads). The masks asked for are chosen before the hypernetwork product: the three
  multimask outputs, or mask 0 and the best other one for the single output (upstream's stability fallback), not
  all four.
- `MlxMemoryAttention` / `MlxMemoryEncoder` are the video modules in the run dtype, the tokens' residual stream
  float32 as in the image encoder (`mlx_video.MlxTracker` runs them).
"""

# ruff: noqa: ANN401 - the upstream torch modules

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.shared.mlx_vit import gelu

from .mlx_net import arr, layer_norm, linear

if TYPE_CHECKING:
    from collections.abc import Callable

    from .net.mask_decoder import MaskDecoder
    from .net.memory_attention import MemoryAttention
    from .net.memory_encoder import CXBlock, MemoryEncoder

F32 = mx.float32
QUERIES = 512  # queries per product with the memory: measured faster than all at once, and a lower peak
MEM_DIM = 64  # fixed: memory feature width (`video.MEM_DIM`)
FILL_HOLE_AREA, HOLE_SCORE = 8, 0.1  # fixed: `video.FILL_HOLE_AREA`, `video.HOLE_SCORE`
# The cached layouts of the builders (`mlx_port.recorded`): the image network, and the video modules loaded with the
# first `video` call. The files below join the cache key by hash.
LAYOUT, VIDEO_LAYOUT = "sam2-mlx-v1", "sam2-mlx-video-v1"
SOURCES = tuple(Path(__file__).with_name(n) for n in ("mlx_net.py", "mlx_heads.py", "mlx_video.py", "fast.py"))


def _is(module: Any, name: str) -> bool:
    """`module` is a `torch.nn` class called `name` (by name: this file does not import PyTorch)."""
    return type(module).__name__ == name


def _conv1x1(conv: Any, dtype: mx.Dtype) -> Callable[[mx.array], mx.array]:
    """A 1 x 1 `Conv2d` as one `addmm` on channels-last maps."""
    w, b = arr(conv.weight[:, :, 0, 0].T, dtype), arr(conv.bias, dtype)
    return lambda x: mx.addmm(b, x, w)


def _mlp(mlp: Any, dtype: mx.Dtype) -> Callable[[mx.array], mx.array]:
    """Upstream `MLP` (ReLU between the layers, an optional sigmoid at the end)."""
    if not _is(mlp.act, "ReLU"):
        msg = f"SAM 2 on MLX: MLP activation {type(mlp.act).__name__} is not written"
        raise NotImplementedError(msg)
    layers = [linear(layer, dtype) for layer in mlp.layers]
    sigmoid = bool(mlp.sigmoid_output)  # a plain value: the closure must not keep the torch module alive

    def run(x: mx.array) -> mx.array:
        for layer in layers[:-1]:
            x = mx.maximum(layer(x), 0)
        x = layers[-1](x)
        return mx.sigmoid(x) if sigmoid else x

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


class MlxPrompts:
    """Upstream `PromptEncoder` of one loaded SAM 2: sparse prompts on the host, the mask prompt on MLX (float32)."""

    def __init__(self, pe: Any) -> None:
        """Copy the tables (the point encodings to numpy) and the mask downscaling's weights."""
        down = pe.mask_downscaling
        kinds = [type(m).__name__ for m in down]
        if kinds != ["Conv2d", "LayerNorm2d", "GELU", "Conv2d", "LayerNorm2d", "GELU", "Conv2d"]:
            msg = f"SAM 2 on MLX: this mask downscaling is not written ({kinds})"
            raise NotImplementedError(msg)
        self.size = np.array([pe.input_image_size[1], pe.input_image_size[0]], dtype=np.float32)  # (w, h)
        th, tw = pe.image_embedding_size
        self.gauss = np.array(arr(pe.pe_layer.positional_encoding_gaussian_matrix, F32))  # [2, 128]
        self.points = np.concatenate([np.array(arr(e.weight, F32)) for e in pe.point_embeddings])  # [4, 256]
        self.not_a_point = np.array(arr(pe.not_a_point_embed.weight, F32))  # [1, 256]
        self.no_mask = arr(pe.no_mask_embed.weight, F32).reshape(1, 1, -1)
        self.down = [
            (arr(conv.weight.permute(0, 2, 3, 1), F32), arr(conv.bias, F32), conv.stride[0], layer_norm(norm, F32))
            for conv, norm in ((down[0], down[1]), (down[3], down[4]))
        ]
        self.down_out = _conv1x1(down[6], F32)
        # upstream `get_dense_pe`: the encoding of the token centres, (x, y) in [0, 1]
        x = np.broadcast_to(((np.arange(tw, dtype=np.float32) + 0.5) / tw)[None, :], (th, tw))
        y = np.broadcast_to(((np.arange(th, dtype=np.float32) + 0.5) / th)[:, None], (th, tw))
        self.dense_pe = mx.array(self._fourier(np.stack([x, y], axis=-1)).reshape(1, th * tw, -1))
        mx.eval(self.dense_pe)

    def _fourier(self, coords: np.ndarray) -> np.ndarray:
        """Upstream `PositionEmbeddingRandom._pe_encoding`: `[..., 2]` in `[0, 1]` -> `[..., 256]` float32."""
        x = (2 * coords.astype(np.float32) - 1) @ self.gauss
        x = np.float32(2 * math.pi) * x
        return np.concatenate([np.sin(x), np.cos(x)], axis=-1).astype(np.float32)

    def sparse(self, coords: np.ndarray, labels: np.ndarray) -> mx.array:
        """Upstream `_embed_points` (padded): `[P, N, 2]` input-pixel coordinates, labels -> `[P, N + 1, 256]`.

        Labels: 0 / 1 a negative / positive click, 2 / 3 a box's corners, -1 not a point.
        """
        points = np.pad(coords.astype(np.float32) + np.float32(0.5), ((0, 0), (0, 1), (0, 0)))
        labs = np.pad(labels.astype(np.int64), ((0, 0), (0, 1)), constant_values=-1)[..., None]
        emb = self._fourier(points / self.size)
        emb = np.where(labs == -1, self.not_a_point, emb)
        emb = emb + self.points[np.clip(labs[..., 0], 0, len(self.points) - 1)] * (labs >= 0)
        return mx.array(emb.astype(np.float32))

    def empty(self, p: int) -> mx.array:
        """`sparse` of upstream's one not-a-point click per object (the tracker's frames without a prompt)."""
        return mx.broadcast_to(mx.array(self.not_a_point)[None], (p, 2, self.not_a_point.shape[1]))

    def dense(self, mask: mx.array) -> mx.array:
        """Upstream `_embed_masks`: mask logits `[P, 4h, 4w]` -> the dense prompt `[P, h * w, 256]`."""
        x = mask[..., None].astype(F32)
        for w, b, stride, norm in self.down:
            x = gelu(norm(mx.conv2d(x, w, stride=stride) + b))
        x = self.down_out(x)
        return x.reshape(x.shape[0], -1, x.shape[-1])


class MlxDecoder:
    """Upstream `MaskDecoder` of one loaded SAM 2 on MLX, float32."""

    def __init__(self, dec: MaskDecoder, pe: mx.array) -> None:
        """Copy the decoder's weights; `pe` is the token grid's encoding `[1, h * w, 256]` (`MlxPrompts.dense_pe`)."""
        ups = dec.output_upscaling
        if not dec.use_high_res_features or not _is(ups[2], "GELU"):
            msg = "SAM 2 on MLX: a mask decoder without high-resolution features or GELU upscaling is not written"
            raise NotImplementedError(msg)
        self.pe = pe
        self.scored = bool(dec.pred_obj_scores)
        tokens = [dec.obj_score_token.weight] if self.scored else []
        self.tokens = mx.concatenate([arr(t, F32) for t in (*tokens, dec.iou_token.weight, dec.mask_tokens.weight)])
        self.n_masks = dec.num_mask_tokens
        tr: Any = dec.transformer
        self.layers = [self._layer(layer) for layer in tr.layers]
        self.final = _attention(tr.final_attn_token_to_image, F32)
        self.final_norm = _token_norm(tr.norm_final_attn)
        self.up1, self.up_norm, self.up2 = _up(ups[0]), layer_norm(ups[1], F32), _up(ups[3])
        self.hyper = [_mlp(m, F32) for m in dec.output_hypernetworks_mlps]
        self.iou = _mlp(dec.iou_prediction_head, F32)
        head = dec.pred_obj_score_head if self.scored else None
        self.score = None if head is None else _mlp(head, F32) if _is(head, "MLP") else linear(head, F32)
        # upstream `forward`'s choices
        self.dynamic = bool(dec.dynamic_multimask_via_stability)
        self.delta = float(dec.dynamic_multimask_stability_delta)
        self.thresh = float(dec.dynamic_multimask_stability_thresh)
        self.token_per_mask = bool(dec.use_multimask_token_for_obj_ptr)

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

    def run(  # noqa: PLR0913, PLR0917 - upstream `predict_masks` arguments
        self, image: mx.array, dense: mx.array, sparse: mx.array | None, s0: mx.array, s1: mx.array, p: int
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        """Upstream `predict_masks` up to the hypernetwork product, for `p` prompts.

        `image` `[1 or P, h, w, 256]` (float32, channels last as `s0` `[.., 4h, 4w, 32]` and `s1` `[.., 2h, 2w, 64]`),
        `dense` `[1, 1, 256]` or `[P, h * w, 256]`, `sparse` `[P, T, 256]` or None (no point tokens).

        Returns:
            The upscaled map `[P, 16 h w, 32]`, the hypernetwork rows `[P, 4, 32]` (a mask is their product, `masks`),
            the predicted IoUs `[P, 4]`, the mask tokens `[P, 4, 256]` and the object score `[P, 1]`.
        """
        _, h, w, c = image.shape
        k = mx.broadcast_to(image.reshape(-1, h * w, c) + dense, (p, h * w, c))
        tokens = mx.broadcast_to(self.tokens, (p, *self.tokens.shape))
        if sparse is not None:
            tokens = mx.concatenate([tokens, sparse], axis=1)
        q = tokens
        for layer in self.layers:
            q, k = layer(q, k, tokens, self.pe)
        q = self.final_norm(q + self.final(q + tokens, k + self.pe, k))
        s = int(self.scored)
        mask_tokens = q[:, s + 1 : s + 1 + self.n_masks]
        up = gelu(self.up_norm(self.up1(k.reshape(p, h, w, c)) + s1))
        up = gelu(self.up2(up) + s0)
        hyper = mx.stack([mlp(mask_tokens[:, i]) for i, mlp in enumerate(self.hyper)], axis=1)  # [P, 4, C / 8]
        score = self.score(q[:, 0]) if self.score is not None else mx.full((p, 1), 10.0)
        return up.reshape(p, 16 * h * w, -1), hyper, self.iou(q[:, s]), mask_tokens, score

    @staticmethod
    def masks(up: mx.array, rows: mx.array) -> mx.array:
        """The masks `[P, R, 4h, 4w]` of the hypernetwork rows `rows` `[P, R, 32]` (square maps)."""
        p, r = rows.shape[:2]
        side = math.isqrt(up.shape[1])
        return (up @ rows.transpose(0, 2, 1)).transpose(0, 2, 1).reshape(p, r, side, side)

    def single(self, up: mx.array, hyper: mx.array, iou: mx.array) -> tuple[mx.array, mx.array]:
        """Upstream `forward` without `multimask_output`: the mask `[P, 4h, 4w]` and its predicted IoU `[P]`.

        Mask 0, or (upstream `_dynamic_multimask_via_stability`) the best of the others where mask 0 is unstable.
        """
        if not self.dynamic:
            return self.masks(up, hyper[:, :1])[:, 0], iou[:, 0]
        best = mx.argmax(iou[:, 1:], axis=-1)[:, None] + 1  # [P, 1]
        both = self.masks(
            up, mx.concatenate([hyper[:, :1], mx.take_along_axis(hyper, best[:, :, None], axis=1)], axis=1)
        )
        first = both[:, 0]
        area_i = (first > self.delta).sum(axis=(1, 2)).astype(F32)
        area_u = (first > -self.delta).sum(axis=(1, 2)).astype(F32)
        stable = mx.where(area_u > 0, area_i / mx.maximum(area_u, 1), 1.0) >= self.thresh
        score = mx.where(stable, iou[:, 0], mx.take_along_axis(iou, best, axis=1)[:, 0])
        return mx.where(stable[:, None, None], first, both[:, 1]), score

    def predict(  # noqa: PLR0913, PLR0917 - upstream `forward` arguments
        self,
        image: mx.array,
        dense: mx.array,
        sparse: mx.array | None,
        s0: mx.array,
        s1: mx.array,
        p: int,
        *,
        multimask: bool,
    ) -> tuple[mx.array, mx.array]:
        """Upstream `forward` (arguments as `run`): logits `[P, M, 4h, 4w]` and predicted IoUs `[P, M]`, M 3 or 1."""
        up, hyper, iou, _, _ = self.run(image, dense, sparse, s0, s1, p)
        if multimask:
            return self.masks(up, hyper[:, 1:]), iou[:, 1:]
        low, score = self.single(up, hyper, iou)
        return low[:, None], score[:, None]


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
            if not _is(norm, "LayerNorm2d") or not _is(act, "GELU"):
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
        if not _is(enc.fuser.proj, "Identity") or _is(enc.out_proj, "Identity"):
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
