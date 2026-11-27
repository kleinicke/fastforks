"""Layer-by-layer diff of a hand-written MLX port against its PyTorch reference, and its per-block time.

    python tools/mlx_layer_diff.py REF [--dtype f16] [--size HxW] [--time] [--runs 5] [--tol 0.02] [--mode portable]

Runs the port (`device="mlx"`, the dtype given) with its `mlx_port.tap` calls recording, then the same inputs
(`fastforks._core.bench.bench_inputs` at `--size`) through the PyTorch portable network on the CPU in float32 with
forward hooks on the modules the taps name. Prints, per tap in call order: the shape, max / mean abs diff, the max diff
relative to the reference's largest magnitude, and with `--time` the median milliseconds since the previous tap; then
the call's outputs and the first tap whose relative diff exceeds `--tol` (the first diverging layer).

A tap names a torch module by its dotted path or a suffix of it (`blocks.3` matches `pretrained.blocks.3`); `name[i]`
is element `i` of a module's tuple output. A module called several times (blocks of a loop, the iterations of a
refinement) is matched call by call. MLX layouts are aligned to torch's where the shapes say how: `[B, H, W, C]` and
`[B, D, H, W, C]` against channels-first, tokens `[B, N, C]` against `[B, C, h, w]`. A tap without a torch module (no
match, or one the reference does not call) is listed with its time only.

The relative diff is against the tap's own largest magnitude, so a residual that shrinks over a refinement loop (the
flow update of iteration 20) shows fp16 noise as a large share of a tiny value: judge those by `max|d|` and the final
output, or set `--atol` so the first-diverging line skips them.

The port runs with `mx.disable_compile()` and every tap evaluates its value, so the times are what each block costs
when run alone: use them to rank blocks, not as the shipped call time (`tools/mlx_core_ab.py` measures that).

A GPU job: run it under the GPU lock (`gpu.sh` / `gpu2.sh --short`).
"""

from __future__ import annotations

import argparse
import gc
import importlib
import statistics
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import nn

DTYPES = {"f16": "float16", "fp16": "float16", "bf16": "bfloat16", "f32": "float32", "fp32": "float32"}
TOL = {"float16": 2e-2, "bfloat16": 5e-2, "float32": 1e-3}


@dataclass
class Row:
    """One tap (or output) compared: `rel` None when there is no reference value to compare with."""

    name: str
    shape: tuple[int, ...]
    torch: str | None  # the module matched, or why not
    max: float | None = None
    mean: float | None = None
    rel: float | None = None
    ms: float | None = None


def split(name: str) -> tuple[str, int | None]:
    """`"mod[1]"` -> `("mod", 1)`, `"mod"` -> `("mod", None)`."""
    if name.endswith("]") and "[" in name:
        base, idx = name[:-1].rsplit("[", 1)
        return base, int(idx)
    return name, None


def align(ref: np.ndarray, got: np.ndarray) -> np.ndarray | None:
    """`got` (an MLX tap) in `ref`'s layout, or None when no known layout change matches the shapes."""
    cands = []
    if got.ndim == 4:  # noqa: PLR2004 - NHWC
        cands.append(got.transpose(0, 3, 1, 2))
    if got.ndim == 5:  # noqa: PLR2004 - NDHWC
        cands.append(got.transpose(0, 4, 1, 2, 3))
    cands.append(got)
    if got.ndim == 3 and ref.ndim == 4 and got.size == ref.size:  # noqa: PLR2004 - tokens [B, N, C] of a map
        cands.append(got.transpose(0, 2, 1).reshape(ref.shape))
    if got.ndim == 4 and ref.ndim == 3 and got.size == ref.size:  # noqa: PLR2004 - NHWC map as tokens [B, N, C]
        cands.append(got.reshape(ref.shape))
    if got.size == ref.size and np.squeeze(got).shape == np.squeeze(ref).shape:
        cands.append(got.reshape(ref.shape))
    return next((c for c in cands if c.shape == ref.shape), None)


def diff(ref: np.ndarray, got: np.ndarray) -> tuple[float, float, float]:
    """Max and mean abs difference, and the max relative to the largest reference magnitude."""
    d = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    scale = float(np.max(np.abs(ref))) if ref.size else 0.0
    m = float(np.max(d)) if d.size else 0.0
    return m, float(np.mean(d)) if d.size else 0.0, m / max(scale, 1e-12)


def compare(
    taps: Iterable[Any],
    recorded: dict[str, list[Any]],
    matched: dict[str, str | None],
    times: list[float] | None = None,
) -> list[Row]:
    """The rows of `taps` (`mlx_port.Tap`s) against the hooked torch outputs `recorded` (module -> outputs per call).

    `matched`: tap module name -> the torch module it hooked (None: none found); `times`: median ms per tap index.
    """
    seen: dict[str, int] = {}
    rows = []
    for i, t in enumerate(taps):
        base, idx = split(t.name)
        k = seen[t.name] = seen.get(t.name, -1) + 1
        row = Row(t.name, t.shape, matched.get(base), ms=times[i] if times and i < len(times) else None)
        rows.append(row)
        mod = matched.get(base)
        if mod is None:
            row.torch = "no torch module"
            continue
        calls = recorded.get(mod, [])
        if k >= len(calls):
            row.torch = f"{mod}: called {len(calls)}x"
            continue
        out = calls[k]
        if idx is not None:
            out = out[idx] if isinstance(out, (list, tuple)) and idx < len(out) else None
        elif isinstance(out, (list, tuple)):
            out = out[0] if out else None
        if out is None or t.value is None:
            continue
        got = align(out, t.value)
        if got is None:
            row.torch = f"{mod}: shape {list(out.shape)}"
            continue
        row.max, row.mean, row.rel = diff(out, got)
    return rows


def first_diverging(rows: list[Row], tol: float, atol: float = 0.0) -> Row | None:
    """The first compared row whose relative diff exceeds `tol` and whose max abs diff exceeds `atol`."""
    return next((r for r in rows if r.rel is not None and r.rel > tol and (r.max or 0.0) > atol), None)


def _num(v: float | None, fmt: str) -> str:
    return "-" if v is None else format(v, fmt)


def table(rows: list[Row], tol: float) -> str:
    """The rows as a fixed-width table (a `!` marks a relative diff over `tol`)."""
    width = max([len(r.name) for r in rows] + [4])
    lines = [f"{'#':>3}  {'tap':<{width}}  {'shape':<18} {'max|d|':>9} {'mean|d|':>9} {'rel':>8}  {'ms':>7}  torch"]
    for i, r in enumerate(rows):
        mark = "!" if r.rel is not None and r.rel > tol else " "
        shape = "x".join(map(str, r.shape))
        lines.append(
            f"{i:>3}  {r.name:<{width}}  {shape:<18} {_num(r.max, '9.2e')} {_num(r.mean, '9.2e')} "
            f"{_num(r.rel, '8.1e')}{mark} {_num(r.ms, '7.2f')}  {r.torch or ''}"
        )
    return "\n".join(lines)


def roots(obj: Any, depth: int = 4) -> list[tuple[str, nn.Module]]:  # noqa: ANN401 - a runner
    """The torch modules a runner holds (attributes, lists, dicts, nested objects), outermost first, by path."""
    from torch import Tensor, nn  # noqa: PLC0415

    out: list[tuple[str, nn.Module]] = []
    seen: set[int] = set()
    todo: list[tuple[str, Any, int]] = [("", obj, 0)]
    while todo:
        path, o, d = todo.pop(0)
        if id(o) in seen or isinstance(o, (Tensor, str, bytes, int, float, type)) or o is None:
            continue
        seen.add(id(o))
        if isinstance(o, nn.Module):
            out.append((path, o))
            continue
        if d >= depth:
            continue
        if isinstance(o, dict):
            items = [(str(k), v) for k, v in o.items()]
        elif isinstance(o, (list, tuple)):
            items = [(str(i), v) for i, v in enumerate(o)]
        elif hasattr(o, "__dict__"):
            items = list(vars(o).items())
        else:
            continue
        todo += [(f"{path}.{k}" if path else k, v, d + 1) for k, v in items]
    return out


def match(names: Iterable[str], mods: list[tuple[str, nn.Module]]) -> dict[str, tuple[str, list[nn.Module]] | None]:
    """Each tap module name -> `(label, modules)`: the exact path, else the shortest suffix match.

    A path relative to a root is taken in every root that has it (volume's part networks run one
    after another), so the k-th tapped call meets the k-th call across those roots in call order.
    """
    every = [(f"{p}.{n}" if n else p, n, m) for p, root in mods for n, m in root.named_modules()]
    out: dict[str, tuple[str, list[nn.Module]] | None] = {}
    for name in names:
        full = [(q, m) for q, _, m in every if q == name]
        rel = [n for _, n, _ in every if n == name] or sorted(
            {n for q, n, _ in every if n.endswith("." + name) or q.endswith("." + name)}, key=len
        )
        if full and not rel:
            out[name] = (full[0][0], [full[0][1]])
        elif rel:
            hit = [(q, m) for q, n, m in every if n == rel[0]]
            label = hit[0][0] + (f" (+{len(hit) - 1} roots)" if len(hit) > 1 else "")
            out[name] = (label, [m for _, m in hit])
        else:
            out[name] = None
    return out


def _to_np(o: Any) -> Any:  # noqa: ANN401 - a module output
    import torch  # noqa: PLC0415

    if isinstance(o, torch.Tensor):
        return o.detach().float().cpu().numpy()
    if isinstance(o, (list, tuple)):
        return tuple(_to_np(x) for x in o)
    return None


def flat(tree: Any, prefix: str = "") -> dict[str, np.ndarray]:  # noqa: ANN401 - a call's result
    """The tensor leaves of a result as float32 numpy arrays, by dotted key."""
    import torch  # noqa: PLC0415

    if isinstance(tree, torch.Tensor):
        return {prefix or "out": tree.detach().cpu().float().numpy()}
    if isinstance(tree, np.ndarray) and tree.dtype != object:
        return {prefix or "out": tree.astype(np.float32)}
    if hasattr(tree, "__dataclass_fields__"):
        tree = {k: getattr(tree, k) for k in tree.__dataclass_fields__}
    if isinstance(tree, (list, tuple)):
        tree = dict(enumerate(tree))
    out: dict[str, np.ndarray] = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            out |= flat(v, f"{prefix}.{k}" if prefix else str(k))
    return out


def _times(model: Any, args: tuple, kw: dict, runs: int) -> list[float]:  # noqa: ANN401
    import torch  # noqa: PLC0415

    from fastforks.shared import mlx_port  # noqa: PLC0415

    each: list[list[float]] = []
    for _ in range(runs):
        with torch.inference_mode(), mlx_port.taps(keep=False) as rows:
            model(*args, **kw)
        each.append([r.ms for r in rows])
    n = min(len(e) for e in each)
    return [statistics.median(e[i] for e in each) for i in range(n)]


def run(a: argparse.Namespace) -> int:
    """Load, tap, hook, compare, print."""
    import torch  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415
    from fastforks._core.bench import BenchSize, bench_inputs  # noqa: PLC0415
    from fastforks._core.mlx import mx  # noqa: PLC0415
    from fastforks.shared import mlx_port  # noqa: PLC0415

    dtype = DTYPES.get(a.dtype, a.dtype)
    tol = a.tol if a.tol is not None else TOL.get(dtype, 2e-2)
    h, w = (int(v) for v in a.size.lower().split("x"))
    card, _ = registry.resolve_any(a.ref)
    task = importlib.import_module(f"fastforks.{card.task}")
    torch.manual_seed(0)
    mx.disable_compile()  # a tap evaluates its value: not inside a compiled function

    model = task.load(a.ref, mode=a.mode, device="mlx", dtype=dtype)
    args, kw = bench_inputs(model, BenchSize(h, w, a.views))
    with torch.inference_mode(), mlx_port.taps() as taps:
        out = flat(model(*args, **kw))
    times = _times(model, args, kw, a.runs) if a.time else None
    runner = type(model._runner).__name__  # noqa: SLF001 - which runner the switch built
    del model
    gc.collect()
    mx.clear_cache()
    print(f"{a.ref}  mlx {a.mode} {dtype} ({runner})  vs  torch portable float32 cpu  input {h}x{w}  tol {tol:g}")
    if not taps:
        print("the port records no taps (`mlx_port.tap` at its block boundaries): comparing the outputs only")

    ref = task.load(a.ref, mode="portable", device="cpu", dtype="float32")
    names = sorted({split(t.name)[0] for t in taps})
    found = match(names, roots(ref._runner))  # noqa: SLF001 - the reference's torch networks
    recorded: dict[str, list[Any]] = {}
    handles = []
    hooked = dict(v for v in found.values() if v is not None)
    for q, ms in hooked.items():

        def hook(_m: Any, _i: Any, o: Any, q: str = q) -> None:  # noqa: ANN401
            recorded.setdefault(q, []).append(_to_np(o))

        handles += [m.register_forward_hook(hook) for m in ms]
    with torch.inference_mode():
        want = flat(ref(*args, **kw))
    for hd in handles:
        hd.remove()

    matched = {k: (v[0] if v else None) for k, v in found.items()}
    rows = compare(taps, recorded, matched, times)
    for k, v in out.items():
        if k in want and want[k].shape == v.shape:
            rows.append(Row(f"output {k}", v.shape, "result", *diff(want[k], v)))
        else:
            rows.append(Row(f"output {k}", v.shape, "result: no matching reference"))
    print(table(rows, tol))
    if times:
        total = sum(times)
        top = sorted(range(len(times)), key=lambda i: -times[i])[:5]
        print(f"tapped time {total:.1f} ms; slowest: " + ", ".join(f"#{i} {taps[i].name} {times[i]:.1f}" for i in top))
    bad = first_diverging(rows, tol, a.atol)
    print("first diverging: " + (f"{bad.name} (rel {bad.rel:.2e})" if bad and bad.rel is not None else "none"))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Command line (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ref")
    ap.add_argument("--dtype", default="f16", help="the port's run dtype: f16, bf16, f32 (or the full name)")
    ap.add_argument("--size", default="384x512", help="input HxW (bench_inputs)")
    ap.add_argument("--views", type=int, default=2, help="views per scene (multi-view tasks)")
    ap.add_argument("--mode", default="portable", help="the MLX mode (portable / kernels)")
    ap.add_argument("--time", action="store_true", help="per-block time: the median of --runs tapped calls")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--tol", type=float, default=None, help="relative diff counted as diverging (default per dtype)")
    ap.add_argument("--atol", type=float, default=0.0, help="... and only above this max abs diff (small residuals)")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
