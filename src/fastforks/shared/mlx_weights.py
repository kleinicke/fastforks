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

A family's hand-written MLX network (`mx` operations, as `seedvr2/net/mlx_net.py`) uses it directly. For a
safetensors checkpoint nothing is converted or cached; a pickled checkpoint must first be written as safetensors
(`fastforks._core.mlx.converted` caches such a conversion under `FASTFORKS_CACHE`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastforks._core.mlx import mx

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

__all__ = ["DTYPES", "Weights", "nbytes"]

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
