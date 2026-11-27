"""Restormer on Apple MLX, written in `mlx.core` operations (`MlxRestormer(net, dtype)`): image -> restored image.

`Restormer.forward` of the vendored upstream network (RGB, no dual-pixel input), channels last. Written out instead
of replayed from an exported graph: the export was ~3 s of a 3.5-4.4 s first call at 512 x 512.

- each LayerNorm runs on the shared row kernel without its affine, which is folded into the 1 x 1 conv after it
  (`qkv`, `project_in`; BiasFree: the row divided by its standard deviation, not centred, as upstream);
- the 1 x 1 convolutions are matrix products over the channels, the `qkv` depthwise conv the shared depthwise kernel;
- MDTA: `q^T k` and the q / k norms over the pixels are one Metal kernel (`token_gram`, float32), the temperature,
  the head mask and the softmax run in float32, and `project_out` is multiplied into the `[C, C]` attention so the
  output is one GEMM of the v tokens;
- GDFN: the depthwise conv, the gelu and the product are one Metal kernel (`dw_gate`); its four channels per thread
  and MLX's GEMM tiles take the hidden width padded to a multiple of 8 (127 -> 128, ...; zero filters);
- the 3 x 3 convolutions go through `mlx_graph._conv2d` (its channel padding onto MLX's fast tiles), as the graph
  translation ran them.

Each block's attention and FFN halves are evaluated one by one, one in flight (`_step`): MLX frees a command buffer's
transients only when it completes, and one lazy graph of the whole U-Net holds far more than its live set.

Every weight-derived array goes through `mlx_port.tensor` by its final shape (`_arr`), so `model._MlxPortable` loads
the folded, padded arrays from the conversion cache (`mlx_port.recorded`, layout `LAYOUT`) without the checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_graph as mg
from fastforks.shared import mlx_port

from .net.restormer_arch import WithBias_LayerNorm

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["LAYOUT", "SOURCES", "MlxRestormer"]

LAYOUT = "restormer-v1"
SOURCES = (Path(__file__),)

ONE = (1, 1)
_EPS = 1e-5  # upstream's LayerNorm epsilon (hard-coded in restormer_arch.py)


def _np(t: Any) -> np.ndarray:  # noqa: ANN401 - a torch tensor
    return t.detach().float().numpy().astype(np.float64)


def _arr(shape: tuple[int, ...], dtype: mx.Dtype, make: Callable[[], np.ndarray]) -> mx.array:
    """The array of `shape` that `make()` computes from weight values, in `dtype` (cached: `make` not run on a hit)."""

    def convert() -> mx.array:
        a = make()
        if a.shape != shape:
            msg = f"Restormer MLX: built {a.shape}, declared {shape}"
            raise AssertionError(msg)
        return mx.array(np.ascontiguousarray(a, np.float32)).astype(dtype)

    return mlx_port.tensor(SimpleNamespace(shape=shape), dtype, convert)


def _unshuffle(x: mx.array) -> mx.array:
    """`PixelUnshuffle(2)` of an NHWC map (output channel `c * 4 + i * 2 + j`)."""
    n, h, w, c = x.shape
    x = x.reshape(n, h // 2, 2, w // 2, 2, c).transpose(0, 1, 3, 5, 2, 4)
    return x.reshape(n, h // 2, w // 2, c * 4)


def _shuffle(x: mx.array) -> mx.array:
    """`PixelShuffle(2)` of an NHWC map."""
    n, h, w, c = x.shape
    x = x.reshape(n, h, w, c // 4, 2, 2).transpose(0, 1, 4, 2, 5, 3)
    return x.reshape(n, h * 2, w * 2, c // 4)


@mx.compile
def _attention(gram: mx.array, sq: mx.array, scale: mx.array, mask: mx.array, proj: mx.array) -> mx.array:
    """`project_out @ softmax(temperature * q^T k / (|q| |k|))` as `[B, C in, C out]` (float32)."""
    c = gram.shape[-1]
    norm = mx.maximum(mx.sqrt(sq), 1e-12)  # F.normalize over the pixels
    g = gram * (scale / norm[:, :c, None]) / norm[:, None, c:] + mask
    return mx.swapaxes(proj @ mx.softmax(g, axis=-1), 1, 2)


class _Block:
    """`TransformerBlock.forward` on NHWC maps."""

    def __init__(self, m: Any, dtype: mx.Dtype) -> None:  # noqa: ANN401 - a `TransformerBlock`
        at, ff = m.attn, m.ffn
        c, heads = at.qkv.in_channels, at.num_heads
        convs = (at.qkv, at.qkv_dwconv, at.project_out, ff.project_in, ff.dwconv, ff.project_out)
        if any(conv.bias is not None for conv in convs):
            msg = "Restormer MLX: convolution biases (bias=True) are not supported (upstream's models are bias-free)"
            raise NotImplementedError(msg)
        self.c, self.dtype = c, dtype
        self.center = isinstance(m.norm1.body, WithBias_LayerNorm)
        self.qkv, self.qkv_b = self._fold(m.norm1.body, lambda: _np(at.qkv.weight)[:, :, 0, 0], (3 * c, c), dtype)
        self.dw = _arr((3, 3, 3 * c), dtype, lambda: _np(at.qkv_dwconv.weight)[:, 0].transpose(1, 2, 0))
        self.dw_b = mx.zeros((3 * c,), dtype)
        rep = c // heads
        self.scale = _arr((1, c, 1), mx.float32, lambda: np.repeat(_np(at.temperature).reshape(-1), rep)[None, :, None])
        head = np.arange(c) // rep
        self.mask = mx.array(np.where(head[:, None] == head[None, :], 0.0, -np.inf).astype(np.float32))
        self.proj = _arr((c, c), mx.float32, lambda: _np(at.project_out.weight)[:, :, 0, 0])  # [C out, C in]
        hidden = ff.project_out.in_channels
        wide = -(-hidden // 8) * 8
        grow = wide - hidden

        def w_in() -> np.ndarray:  # [2 wide, C]
            w = _np(ff.project_in.weight)[:, :, 0, 0].reshape(2, hidden, c)
            return np.pad(w, [(0, 0), (0, grow), (0, 0)]).reshape(2 * wide, c)

        def w_dw() -> np.ndarray:  # [3, 3, 2 wide]
            w = _np(ff.dwconv.weight)[:, 0].transpose(1, 2, 0).reshape(3, 3, 2, hidden)
            return np.pad(w, [(0, 0), (0, 0), (0, 0), (0, grow)]).reshape(3, 3, 2 * wide)

        self.w_in, self.b_in = self._fold(m.norm2.body, w_in, (2 * wide, c), dtype)
        self.w_dw = _arr((3, 3, 2 * wide), dtype, w_dw)
        out = lambda: np.pad(_np(ff.project_out.weight)[:, :, 0, 0].T, [(0, grow), (0, 0)])  # noqa: E731
        self.w_out = _arr((wide, c), dtype, out)

    @staticmethod
    def _fold(
        body: Any,  # noqa: ANN401 - the norm's torch module
        w: Callable[[], np.ndarray],
        shape: tuple[int, int],
        dtype: mx.Dtype,
    ) -> tuple[mx.array, mx.array | None]:
        """The bias-free 1 x 1 conv `w()` `[O, C]` after the norm `body` with its affine folded in: `[C, O]`, bias."""
        o, c = shape
        b = _arr((o,), dtype, lambda: w() @ _np(body.bias)) if isinstance(body, WithBias_LayerNorm) else None
        return _arr((c, o), dtype, lambda: (w() * _np(body.weight)[None, :]).T), b

    def _linear(self, x: mx.array, w: mx.array, b: mx.array | None) -> mx.array:
        return x @ w if b is None else mx.addmm(b, x, w)

    def attn(self, x: mx.array) -> mx.array:
        """The MDTA half of the block: `x` plus the attention of `norm1(x)`."""
        n, h, w, c = x.shape
        t = self._linear(ops.row_layer_norm(x, None, None, _EPS, center=self.center), self.qkv, self.qkv_b)
        t = ops.depthwise(t, self.dw, self.dw_b, ONE, ONE, ONE).reshape(n, h * w, 3 * c)
        gram, sq = ops.token_gram(t, c)
        mat = _attention(gram, sq, self.scale, self.mask, self.proj).astype(self.dtype)
        return x + (t[..., 2 * c :] @ mat).reshape(n, h, w, c)

    def ffn(self, x: mx.array) -> mx.array:
        """The GDFN half of the block: `x` plus the gated feed-forward of `norm2(x)`."""
        y = self._linear(ops.row_layer_norm(x, None, None, _EPS, center=self.center), self.w_in, self.b_in)
        return x + ops.dw_gate(y, self.w_dw) @ self.w_out


_PENDING: list[mx.array] = []  # the step in flight (`_step`)


def _step(x: mx.array) -> mx.array:
    """Schedule `x` and wait for the step before it (pipeline depth 1: the GPU stays busy, the transients bounded).

    restormer-sidd float32, 512 x 512, footprint of a call: the blocks as unbounded `async_eval` steps 3466-3546 MB
    (836 ms), as depth-1 steps 3464-3512 MB (811 ms), each block's attention and FFN halves as depth-1 steps 2982 MB
    (808-874 ms), synchronous 2600-2829 MB (1047-1060 ms); the exported port 2443-2500 MB (1330 ms).
    """
    mx.async_eval(x)
    if _PENDING:
        mx.eval(_PENDING.pop())
    _PENDING.append(x)
    return x


class MlxRestormer:
    """The network: `[B, 3, H, W]` float32 (H, W multiples of 8) -> the restored `[B, 3, H, W]` float32, unclamped."""

    def __init__(self, net: Any, dtype: str) -> None:  # noqa: ANN401 - a `Restormer`
        """The weights of `net` (float32, upstream layout) in `dtype`."""
        if net.dual_pixel_task:
            msg = "Restormer MLX: the dual-pixel network is not supported"
            raise NotImplementedError(msg)
        self.dtype = getattr(mx, dtype)
        self.convs = {
            name: self._weight(_conv(net, name).weight, lambda w: w.transpose(0, 2, 3, 1))
            for name in ("patch_embed", "down1_2", "down2_3", "down3_4", "up4_3", "up3_2", "up2_1", "output")
        }  # OHWI
        self.reduce = {
            name: self._weight(getattr(net, name).weight, lambda w: w[:, :, 0, 0].T)
            for name in ("reduce_chan_level3", "reduce_chan_level2")
        }
        self.stages = {
            name: [_Block(m, self.dtype) for m in getattr(net, name)]
            for name in (
                "encoder_level1", "encoder_level2", "encoder_level3", "latent",
                "decoder_level3", "decoder_level2", "decoder_level1", "refinement",
            )
        }  # fmt: skip

    def _weight(self, t: Any, view: Callable[[np.ndarray], np.ndarray]) -> mx.array:  # noqa: ANN401 - a torch tensor
        """`view` of the conv weight `t` (a transpose; its shape read off a broadcast stand-in), in the run dtype."""
        shape = view(np.broadcast_to(np.float32(0), tuple(int(n) for n in t.shape))).shape
        return _arr(shape, self.dtype, lambda: view(_np(t)))

    def _conv(self, name: str, x: mx.array) -> mx.array:
        return mg._conv2d(x, self.convs[name], ONE, ONE, ONE, 1)  # noqa: SLF001 - the translator's conv dispatch

    def _stage(self, name: str, x: mx.array) -> mx.array:
        for block in self.stages[name]:
            x = _step(block.ffn(_step(block.attn(x))))
        return x

    def __call__(self, image: mx.array) -> mx.array:
        """`Restormer.forward`."""
        mg._KERNELS, mg._GEMM1X1 = False, True  # noqa: SLF001 - the translator's conv dispatch, as the export ran
        mg._HALF_WINO_F32 = mg._HALF_WINO_PAD = False  # noqa: SLF001
        try:
            return self._forward(mx.transpose(image, (0, 2, 3, 1)).astype(self.dtype))
        finally:
            _PENDING.clear()

    def _forward(self, img: mx.array) -> mx.array:
        e1 = self._stage("encoder_level1", self._conv("patch_embed", img))
        e2 = self._stage("encoder_level2", _unshuffle(self._conv("down1_2", e1)))
        e3 = self._stage("encoder_level3", _unshuffle(self._conv("down2_3", e2)))
        x = self._stage("latent", _unshuffle(self._conv("down3_4", e3)))
        x = mx.concatenate([_shuffle(self._conv("up4_3", x)), e3], axis=-1) @ self.reduce["reduce_chan_level3"]
        del e3
        x = self._stage("decoder_level3", x)
        x = mx.concatenate([_shuffle(self._conv("up3_2", x)), e2], axis=-1) @ self.reduce["reduce_chan_level2"]
        del e2
        x = self._stage("decoder_level2", x)
        x = mx.concatenate([_shuffle(self._conv("up2_1", x)), e1], axis=-1)
        del e1
        x = self._stage("refinement", self._stage("decoder_level1", x))
        out = self._conv("output", x) + img
        return mx.transpose(out, (0, 3, 1, 2)).astype(mx.float32)


def _conv(net: Any, name: str) -> Any:  # noqa: ANN401 - torch modules
    """The bias-free 3 x 3 conv `name` (`output`), of a patch embedding (`.proj`) or a resampling (`.body[0]`)."""
    m = getattr(net, name)
    m = m.proj if hasattr(m, "proj") else m.body[0] if hasattr(m, "body") else m
    if m.bias is not None:
        msg = f"Restormer MLX: {name} has a bias (upstream's models are bias-free)"
        raise NotImplementedError(msg)
    return m
