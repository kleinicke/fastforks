"""The Swin (v1) detection backbone on Apple MLX, hand-written: NHWC, the run dtype, each layer one step.

`shared.swin_v1.SwinTransformer.forward` (BiRefNet's and InSPyReNet's encoder): the patch embedding (a reshape of
the NHWC image times the conv as a matrix, its LayerNorm), the layers on `shared.mlx_swin` (LayerNorm + window
shuffles as gathers, the shared window attention kernel), each stage's normed map and `PatchMerging` in one step.
The arrays come through `shared.mlx_port.tensor` (the conversion cache: only a miss reads the torch module).
"""

# ruff: noqa: ANN401 - the network's torch modules

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.shared import mlx_port, mlx_swin, mlx_vit
from fastforks.shared.cspnext import mlx_net as csp

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["SOURCES", "Encoder", "arr", "lin", "norm", "np64"]

SOURCES = (Path(__file__), Path(mlx_swin.__file__))  # the conversion layout depends on these


def np64(t: Any) -> np.ndarray:
    """A torch tensor as float64 numpy."""
    return t.detach().double().numpy()


def arr(shape: tuple[int, ...], dtype: mx.Dtype, make: Callable[[], np.ndarray]) -> mx.array:
    """An array of `shape` made by `make()` in numpy (only on a conversion-cache miss)."""

    def convert() -> mx.array:
        out = mx.array(np.ascontiguousarray(make(), dtype=np.float32)).astype(dtype)
        mx.eval(out)
        return out

    return mlx_port.tensor(SimpleNamespace(shape=tuple(int(d) for d in shape)), dtype, convert)


def lin(layer: Any, dtype: mx.Dtype) -> tuple[mx.array, mx.array | None]:
    """An `nn.Linear` as (`[in, out]`, bias)."""
    o, i = (int(d) for d in layer.weight.shape)
    w = arr((i, o), dtype, lambda: np64(layer.weight).T)
    return w, None if layer.bias is None else arr((o,), dtype, lambda: np64(layer.bias))


def norm(m: Any, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
    """A LayerNorm's (weight, bias)."""
    c = int(m.weight.shape[0])
    return arr((c,), dtype, lambda: np64(m.weight)), arr((c,), dtype, lambda: np64(m.bias))


def _block(blk: Any, dtype: mx.Dtype) -> mlx_swin.Block:
    """A `SwinTransformerBlock` (its relative position bias gathered to `[1, heads, T, T]`)."""
    a = blk.attn
    b = mlx_swin.Block()
    b.window, b.shift = int(blk.window_size), int(blk.shift_size)
    b.heads = int(a.num_heads)
    b.head_dim = int(a.dim) // b.heads
    b.scale = float(a.scale)
    b.qkv, b.o = lin(a.qkv, dtype), lin(a.proj, dtype)
    b.fc1, b.fc2 = lin(blk.mlp.fc1, dtype), lin(blk.mlp.fc2, dtype)
    t, heads = b.window * b.window, b.heads

    def rpb() -> np.ndarray:
        idx = a.relative_position_index.numpy().reshape(-1)
        return np64(a.relative_position_bias_table)[idx].reshape(t, t, heads).transpose(2, 0, 1)[None]

    b.rpb = arr((1, heads, t, t), dtype, rpb)
    b.ln1, b.ln2 = norm(blk.norm1, dtype), norm(blk.norm2, dtype)
    return b


class Encoder:
    """Normalised NHWC `[1, H, W, 3]` (H, W multiples of the patch) -> the four normed stage maps `[1, h, w, C]`.

    `embedding=True`: the normed patch embedding map in front (InSPyReNet's backbone).
    """

    def __init__(self, sw: Any, dtype: mx.Dtype, *, embedding: bool = False) -> None:
        """Convert `sw` (a `SwinTransformer` with `patch_norm` and no absolute position embedding)."""
        proj = sw.patch_embed.proj
        c, _, p, _ = (int(d) for d in proj.weight.shape)
        self.patch, self.dtype, self.embedding = p, dtype, embedding
        self.embed = (
            arr((p * p * 3, c), dtype, lambda: np64(proj.weight).transpose(2, 3, 1, 0).reshape(-1, c)),
            arr((c,), dtype, lambda: np64(proj.bias)),
        )
        self.embed_norm = norm(sw.patch_embed.norm, dtype)
        self.stages: list[tuple[list[Callable[..., mx.array]], Any]] = []
        for layer in sw.layers:
            blocks = [self._step(_block(blk, dtype)) for blk in layer.blocks]
            ds = layer.downsample
            merge = None if ds is None else (norm(ds.norm, dtype), lin(ds.reduction, dtype)[0])
            self.stages.append((blocks, merge))
        self.norms = [norm(getattr(sw, f"norm{i}"), dtype) for i in range(len(self.stages))]
        self.embed_step = csp.stepped(self._embed)
        self.merge_steps = [
            None if m is None else csp.stepped(lambda t, n, m=m: self._merge(t, n, m)) for _, m in self.stages
        ]

    @staticmethod
    def _step(blk: mlx_swin.Block) -> Callable[..., mx.array]:
        """One layer on tokens shaped `[h, w, C]` (the shape keys the compiled graph)."""
        return csp.stepped(lambda t: blk(t.reshape(-1, t.shape[-1]), t.shape[:2]).reshape(t.shape))

    def _embed(self, x: mx.array) -> mx.array:
        h, w = x.shape[1] // self.patch, x.shape[2] // self.patch
        t = mlx_vit.patches(x.transpose(0, 3, 1, 2), self.patch)[0]
        t = mlx_swin.ln(mx.addmm(self.embed[1], t, self.embed[0]), *self.embed_norm)
        return t.reshape(h, w, -1)

    @staticmethod
    def _merge(t: mx.array, n: tuple[mx.array, mx.array], merge: Any) -> tuple[mx.array, mx.array]:
        """The stage map (normed) and `PatchMerging` of the stage's tokens `[h, w, C]`."""
        h, w, c = t.shape
        out = mlx_swin.ln(t, *n)[None]
        t = t.reshape(h // 2, 2, w // 2, 2, c).transpose(0, 2, 3, 1, 4).reshape(h // 2, w // 2, 4 * c)
        return out, mlx_swin.ln(t, *merge[0]) @ merge[1]

    def __call__(self, x: mx.array) -> list[mx.array]:
        """The maps of `x`, shallow first."""
        t = self.embed_step(x)
        feats = [t[None]] if self.embedding else []
        for (blocks, merge), n, step in zip(self.stages, self.norms, self.merge_steps, strict=True):
            for blk in blocks:
                t = blk(t)
            if merge is None or step is None:
                feats.append(mlx_swin.ln(t, *n)[None])
            else:
                f, t = step(t, n)
                feats.append(f)
        return feats
