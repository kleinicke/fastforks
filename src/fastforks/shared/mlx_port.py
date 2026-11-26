"""A hand-written MLX network built straight from the conversion cache: no PyTorch runner, no checkpoint on a hit.

For port authors. A family's MLX network (`mlx_net.py`, `mx` operations) is usually built from the PyTorch network:
its builder walks the modules and turns each weight into an `mx.array` (folded, transposed, cast). `recorded` keeps
that builder as it is and caches what it produced:

- the first load builds the real network (`real`: the checkpoint loaded, under `mlx_wrap.lazy_build`), runs the
  builder (`make`) on it and records every array the builder asks `tensor` for, in call order;
- those arrays go to the conversion cache (`fastforks._core.mlx.convert_cached`): one safetensors file per variant,
  revision, layout and run dtype, already in the run dtype (the arrays the builder asked for in float32 stay float32:
  the file's `fp32` metadata), with its provenance;
- every later load builds the network's skeleton on the meta device (`skeleton`: module structure and shapes, no
  weights, no checkpoint read) and runs the same builder on it: each `tensor` call takes the next cached array
  instead of converting (its shape and dtype checked against the skeleton's tensor), and every array must be taken.

So a builder is cache-ready when it reads weights only through `tensor` (`fastforks.shared.cspnext.mlx_net.arr` does)
and everything else it reads from the modules is structure (shapes, strides, flags), which the meta skeleton has.
What a builder computes from weight values elsewhere (a `.item()`, a numpy view) is no longer seen on a hit: route it
through `tensor` too.

The layout tag names the builder's version, and `sources` (the builder's files and the checkpoint conversion's) join
it by hash: a changed builder never replays an old file (`fastforks cache mlx --prune` removes the superseded one).
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING, Any, TypeVar

from fastforks._core.mlx import convert_cached, mx
from fastforks.shared.mlx_wrap import lazy_build

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from fastforks._core.runner import Context

__all__ = ["recorded", "tensor"]

_T = TypeVar("_T")


class _Tape:
    """The arrays of one build: appended to (`replay` None) or handed out in order."""

    def __init__(self, replay: dict[str, mx.array] | None) -> None:
        self.replay = replay
        self.arrays: list[mx.array] = []
        self.taken = 0


_TAPE: contextvars.ContextVar[_Tape | None] = contextvars.ContextVar("mlx_port_tape", default=None)


def _key(i: int) -> str:
    return f"{i:05d}"


def tensor(t: Any, dtype: mx.Dtype, convert: Callable[[], mx.array]) -> mx.array:  # noqa: ANN401 - a torch tensor
    """The MLX array a builder makes of the torch tensor `t` in `dtype`: `convert()`, or the cached one in a replay.

    Outside `recorded` this is `convert()`. Recording, `convert()` is kept for the cache; replaying, `t` is the
    meta skeleton's tensor and only its shape is read.

    Raises:
        RuntimeError: A replay asks for more arrays than the cache holds, or for another shape or dtype (the builder
            and the cached layout disagree: change the layout tag).
    """
    tape = _TAPE.get()
    if tape is None:
        return convert()
    if tape.replay is None:
        a = convert()
        tape.arrays.append(a)
        return a
    key = _key(tape.taken)
    a = tape.replay.get(key)
    shape = tuple(int(n) for n in t.shape)
    if a is None or tuple(a.shape) != shape or a.dtype != dtype:
        found = "nothing" if a is None else f"{tuple(a.shape)} {a.dtype}"
        msg = f"MLX port cache: array {key} is {found}, the builder wants {shape} {dtype} (bump the layout tag)"
        raise RuntimeError(msg)
    tape.taken += 1
    return a


def _run(tape: _Tape, make: Callable[[Any], _T], net: Any) -> _T:  # noqa: ANN401 - a torch module
    token = _TAPE.set(tape)
    try:
        return make(net)
    finally:
        _TAPE.reset(token)


def recorded(  # noqa: PLR0913 - the builder, both networks and the cache key
    ctx: Context,
    layout: str,
    make: Callable[[Any], _T],
    *,
    real: Callable[[], Any],
    skeleton: Callable[[], Any],
    sources: tuple[Path, ...] = (),
) -> _T:
    """`make(network)` in the run dtype `ctx.dtype`, its arrays from the conversion cache (see the module docstring).

    Args:
        ctx: The runner's context (its variant, card and dtype key the cache).
        layout: The builder's layout tag (name and version, plus every option that changes its arrays).
        make: The MLX builder: a torch network -> the MLX network, every weight read through `tensor`.
        real: Builds the PyTorch network with the checkpoint (a miss, or `load(weights=...)`): float32 on the CPU.
        skeleton: Builds the same network without weights; called under the meta device.
        sources: The files whose code shapes the arrays (the builder, the checkpoint conversion).

    Raises:
        RuntimeError: The cached arrays and the builder disagree (see `tensor`).
    """
    built: list[_T] = []
    islands: set[str] = set()  # what the builder asked for in float32 stays float32

    def build() -> tuple[dict[str, mx.array], dict[str, Any]]:
        tape = _Tape(None)
        built.append(_run(tape, make, lazy_build(real)))
        arrays = {_key(i): a for i, a in enumerate(tape.arrays)}
        islands.update(k for k, a in arrays.items() if a.dtype == mx.float32)
        return arrays, {"arrays": len(arrays)}

    found = convert_cached(ctx, layout, build, dtype=ctx.dtype, fp32=islands.__contains__, sources=sources)
    if not found.hit:
        return built[0]
    import torch  # noqa: PLC0415 - the skeleton is a torch module on the meta device

    mx.eval(list(found.arrays.values()))  # read the file now, not in the first call
    with torch.device("meta"):
        net = skeleton()
    tape = _Tape(found.arrays)
    out = _run(tape, make, net)
    if tape.taken != len(found.arrays):
        n = len(found.arrays)
        msg = f"MLX port cache: the builder took {tape.taken} of {n} cached arrays (bump the layout tag)"
        raise RuntimeError(msg)
    return out
