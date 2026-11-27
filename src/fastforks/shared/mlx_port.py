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

Taps, for debugging a port layer by layer (`tools/mlx_layer_diff.py`): a hand-written forward marks what it computed
with `x = tap(name, x)` at its block boundaries, `name` being the PyTorch module whose output `x` corresponds to (its
dotted path in the torch network, or a suffix of it: `blocks.3`, `layer2.1`, `scratch.refinenet4`). `tap` returns `x`
unchanged and costs a context-variable read; only inside `with taps() as rows:` does it record. Recording evaluates `x`
(`mx.eval`), so the tool runs the port with `mx.disable_compile()`; `rows` gets one `Tap` per call, in call order, with
its value (float32 numpy) and the milliseconds since the previous tap (the work up to this boundary). A tap's layout
may be MLX's (`[B, H, W, C]`, `[B, D, H, W, C]`, tokens `[B, N, C]` of a `[B, C, h, w]` map): the tool aligns it.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from fastforks._core.mlx import convert_cached, mx

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    import numpy as np

    from fastforks._core.runner import Context

__all__ = ["Tap", "header_skeleton", "recorded", "tap", "taps", "tensor"]

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
        from fastforks.shared.mlx_wrap import lazy_build  # noqa: PLC0415 - torch: only for a first build

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


def header_skeleton(path: str | Path) -> dict[str, Any]:
    """A safetensors file's tensors as empty ones of their shapes, from its header only: a state-dict skeleton.

    For a builder that reads the checkpoint's tensors by name rather than a module (`recorded`'s `skeleton`, called
    under the meta device): no checkpoint read and no network class to import on a hit.
    """
    import torch  # noqa: PLC0415 - the skeleton's tensors

    with Path(path).open("rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    return {k: torch.empty(v["shape"]) for k, v in header.items() if k != "__metadata__"}


@dataclass
class Tap:
    """One recorded `tap` call: the torch module `name`, its value (float32, None unless kept) and the time to it."""

    name: str
    shape: tuple[int, ...]
    value: np.ndarray | None
    ms: float


class _Recorder:
    def __init__(self, *, keep: bool) -> None:
        self.keep = keep
        self.rows: list[Tap] = []
        self.last = time.perf_counter()


_TAPS: contextvars.ContextVar[_Recorder | None] = contextvars.ContextVar("mlx_port_taps", default=None)


def tap(name: str, x: mx.array) -> mx.array:
    """`x`, recorded under the torch module `name` inside `taps()` (see the module docstring); else just `x`."""
    rec = _TAPS.get()
    if rec is not None:
        import numpy as np  # noqa: PLC0415 - only while recording

        mx.eval(x)
        now = time.perf_counter()
        value = np.array(x.astype(mx.float32)) if rec.keep else None
        rec.rows.append(Tap(name, tuple(x.shape), value, 1e3 * (now - rec.last)))
        rec.last = time.perf_counter()  # the copy to numpy is not the next block's time
    return x


@contextlib.contextmanager
def taps(*, keep: bool = True) -> Generator[list[Tap]]:
    """Record every `tap` of the block (`keep`: with its value; without, the times only)."""
    rec = _Recorder(keep=keep)
    token = _TAPS.set(rec)
    try:
        yield rec.rows
    finally:
        _TAPS.reset(token)
