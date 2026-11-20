"""Safetensors checkpoint weights straight into MLX, one group of tensors at a time: no PyTorch twin, no float32 copy.

`mx.load` of a safetensors file is lazy: every array it returns is a pending read of its own byte range of the file,
so opening a 16 GB checkpoint costs nothing and evaluating one array reads only that array. `Weights` builds on that
for networks too big for the usual MLX port (`mlx_wrap.port` first builds the PyTorch runner on the CPU in float32,
then copies its parameters into MLX: a 3.4 B-parameter network needs 13.6 GB for that twin alone):

- `take(keys, dtype)` converts a group (a transformer block, a VAE stage) to the run dtype, evaluates it and drops the
  lazy source, so the checkpoint dtype and the run dtype never coexist for more than that group. Resident memory is
  then the weights in the run dtype plus one group in flight.
- `stream(keys, dtype)` returns the group as fresh lazy reads that are not kept: the bytes come from the file (the
  page cache) on every evaluation, for weights that do not fit resident (a 7 B transformer in bf16 is 16.5 GB).
- `peek(key)` is one lazy array for load-time arithmetic (embeddings computed once from the checkpoint).
- `nf4_quantize` / `Nf4.expand`: a linear weight stored in 4 bit (`fastforks.shared.nf4`'s format, bit-exact with its
  PyTorch quantiser and expansion), for a transformer that does not fit resident in bf16 (SeedVR2 7B: 4.3 GB of NF4
  blocks instead of 16.3 GB).

A family's hand-written MLX network (`mx` operations, as `seedvr2/net/mlx_net.py`) uses it directly. For a
safetensors checkpoint nothing is converted or cached; a pickled checkpoint must first be written as safetensors
(`fastforks._core.mlx.converted` caches such a conversion under `FASTFORKS_CACHE`).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

from fastforks._core.mlx import mx
from fastforks.shared.nf4 import BLOCK, NF4_CODE

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

__all__ = ["DTYPES", "Nf4", "Weights", "nbytes", "nf4_quantize"]

DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}
"""Run dtype names (the `ctx.dtype` strings) -> MLX dtypes."""


def nbytes(arrays: Iterable[mx.array]) -> int:
    """Total bytes of `arrays` (evaluated or not)."""
    return sum(a.nbytes for a in arrays)


def _cast(a: mx.array, dtype: mx.Dtype | None) -> mx.array:
    """`a` in `dtype` if it is a floating array of another floating dtype (integer tensors keep theirs)."""
    if dtype is None or a.dtype == dtype or not mx.issubdtype(a.dtype, mx.floating):
        return a
    return a.astype(dtype)


def _open(path: str) -> dict[str, mx.array]:
    """The lazy arrays of the safetensors file `path` (cached files may carry no extension, e.g. Hub blobs)."""
    return mx.load(path, format="safetensors")


class Weights:
    """The tensors of one or more safetensors files as lazy MLX arrays, handed out group by group.

    Args:
        paths: The files; a key in a later file replaces the same key of an earlier one.
    """

    def __init__(self, *paths: str | Path) -> None:
        """Open the files lazily (nothing is read here)."""
        self.paths = tuple(str(p) for p in paths)
        self._lazy: dict[str, mx.array] = {}
        for p in self.paths:
            self._lazy.update(_open(p))

    def keys(self) -> list[str]:
        """The keys not yet taken."""
        return list(self._lazy)

    def __contains__(self, key: str) -> bool:
        """`key` is in the files and not yet taken."""
        return key in self._lazy

    def prefixed(self, prefix: str) -> list[str]:
        """The keys not yet taken that start with `prefix`."""
        return [k for k in self._lazy if k.startswith(prefix)]

    def peek(self, key: str) -> mx.array:
        """`key` as a lazy array in its checkpoint dtype (kept available for `take` / `stream`)."""
        return self._lazy[key]

    def take(
        self,
        keys: Iterable[str],
        dtype: mx.Dtype | None = None,
        transform: Callable[[str, mx.array], mx.array] | None = None,
    ) -> dict[str, mx.array]:
        """`keys` read, `transform`ed, cast to `dtype`, evaluated together, and forgotten here.

        `transform` sees the checkpoint dtype; only floating tensors are cast. Only this group's checkpoint copy and
        its converted copy coexist; the lazy source of each array is released by the evaluation.
        """
        out = {}
        for k in list(keys):
            a = self._lazy.pop(k)
            out[k] = _cast(transform(k, a) if transform is not None else a, dtype)
        mx.eval(out)
        return out

    def stream(
        self,
        keys: Iterable[str],
        dtype: mx.Dtype | None = None,
        transform: Callable[[str, mx.array], mx.array] | None = None,
    ) -> dict[str, mx.array]:
        """`keys` as fresh lazy reads (`transform`ed, cast), not kept: every evaluation reads the file again."""
        fresh: dict[str, mx.array] = {}
        for p in self.paths:
            fresh.update(_open(p))
        out = {}
        for k in keys:
            a = fresh[k]
            out[k] = _cast(transform(k, a) if transform is not None else a, dtype)
        return out


_TINY = 1.1754943508222875e-38  # torch.finfo(torch.float32).tiny


@dataclass(frozen=True)
class Nf4:
    """A linear weight `[n, k]` in NF4: packed bytes `q` `[n * k / 2]` (high nibble first), bf16 `absmax`."""

    q: mx.array
    absmax: mx.array

    @property
    def nbytes(self) -> int:
        """Bytes held."""
        return self.q.nbytes + self.absmax.nbytes

    def expand(self, dtype: mx.Dtype) -> mx.array:
        """The weight in `dtype`: `bf16(code * absmax)` (float32 product, as bitsandbytes), then cast."""
        return _nf4_expand(self.q, self.absmax).astype(dtype)


@mx.compile
def _nf4_expand(q: mx.array, absmax: mx.array) -> mx.array:
    code = mx.array(NF4_CODE, dtype=mx.float32)
    vals = mx.stack([code[(q >> 4).astype(mx.uint32)], code[(q & 15).astype(mx.uint32)]], axis=-1)
    w = vals.reshape(-1, BLOCK) * absmax.reshape(-1, 1).astype(mx.float32)
    return w.astype(mx.bfloat16).reshape(absmax.shape[0], -1)


@mx.compile
def _nf4_codes(w: mx.array) -> tuple[mx.array, mx.array]:
    n, k = w.shape
    blk = w.astype(mx.float32).reshape(n, k // BLOCK, BLOCK)
    am = mx.abs(blk).max(axis=-1).astype(mx.bfloat16)
    x = blk / mx.maximum(am.astype(mx.float32), _TINY)[..., None]
    idx = mx.zeros(x.shape, dtype=mx.uint8)
    for a, b in pairwise(NF4_CODE):  # torch.bucketize(x, mids): the mids below x
        idx = idx + (x > (a + b) / 2).astype(mx.uint8)
    pairs = idx.reshape(-1, 2)
    return (pairs[:, 0] << 4) | pairs[:, 1], am


def nf4_quantize(w: mx.array) -> Nf4:
    """Weight `[n, k]` (k a multiple of 64) -> `Nf4`, as `fastforks.shared.nf4.quantize` (evaluated).

    Per block of 64 weights the bf16 absmax, each weight the nearest code value of `w / absmax` (ties to the lower).
    """
    if w.shape[1] % BLOCK:
        msg = f"NF4 needs the input width to be a multiple of {BLOCK}, got {w.shape[1]}"
        raise ValueError(msg)
    q, am = _nf4_codes(w)
    mx.eval(q, am)
    return Nf4(q, am)
