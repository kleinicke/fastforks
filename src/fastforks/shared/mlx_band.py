"""Row-banded execution of the tail of a translated graph (`fastforks.shared.mlx_graph`).

A super-resolution net ends in convolutions at the output resolution (Real-ESRGAN x4: three 64-channel convolutions
on 2048x2048 maps, 0.5 GB per map, with Winograd transients several times that). Evaluated whole, the tail sets the
peak memory of the forward at several times the rest of the net. When the tail is a *chain* of row-local operations
(convolutions, elementwise operations, integer nearest-neighbour up-sampling) from one value to the output, the
output rows depend on a bounded window of input rows: the tail then runs per band of rows (with the halo the
convolutions need), each band evaluated before the next one starts, and the bands are joined at the end.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Any, cast

import torch

from fastforks._core.mlx import mx

aten = torch.ops.aten

BAND_MB = int(os.environ.get("FF_MLX_BAND_MB", "48"))  # the largest slab (one value's rows of a band) of a band
BAND_MIN_MB = 128  # a tail whose largest value is smaller than this runs whole
_UP = (aten.upsample_nearest2d.default, aten.upsample_nearest2d.vec)
# channels-last ops whose output rows are not a function of the same input rows: the statistics of an instance norm are
# over the whole map (per band they made BEN2's last 1024x1024 block wrong by 0.1 of the alpha range)
_WHOLE_MAP = (aten.instance_norm.default,)


def band_tail(  # noqa: C901, PLR0911, PLR0912, PLR0915
    nodes: list[Any], plan: list[tuple[Any, ...]], values: dict[str, Any], cl_values: set[str], itemsize: int
) -> list[tuple[Any, ...]]:
    """`plan` with its row-local tail chain replaced by one step that runs it per band (or `plan` itself)."""
    from fastforks.shared import mlx_graph as g  # noqa: PLC0415 - mlx_graph imports this module lazily

    if os.environ.get("FF_MLX_NO_BAND") or not plan or plan[-1][0] != "output":
        return plan
    out_refs = list(g._refs(plan[-1][1]))  # noqa: SLF001
    if len(out_refs) != 1:  # one output value
        return plan
    bynode = {n.name: n for n in nodes}
    step_of = {p[1]: p for p in plan if p[0] in ("call", "alias")}
    alias = {p[1]: p[2] for p in plan if p[0] == "alias"}

    def real(name: str) -> str:
        while name in alias:
            name = alias[name]
        return name

    cons: dict[str, set[str]] = defaultdict(set)
    for p in plan:
        if p[0] == "call":
            for a in (*p[3], *p[4].values()):
                for r in g._refs(a):  # noqa: SLF001
                    cons[real(r)].add(p[1])
    cons[real(out_refs[0])].add("\0output")

    def shape(name: str) -> tuple[int, ...] | None:
        node = bynode.get(name)
        v = node.meta.get("val") if node is not None else None
        return tuple(int(d) for d in v.shape) if isinstance(v, torch.Tensor) else None

    chain: list[str] = []  # real value names, output first
    kinds: dict[str, str] = {}
    cur = real(out_refs[0])
    while True:
        p = step_of.get(cur)
        node = bynode.get(cur)
        shp = shape(cur)
        if p is None or p[0] != "call" or node is None or shp is None or len(shp) != 4:  # noqa: PLR2004
            break
        if not cons[cur] <= {*chain, "\0output"}:
            break
        tensor_refs = {real(r) for a in (*p[3], *p[4].values()) for r in g._refs(a)}  # noqa: SLF001
        data = {r for r in tensor_refs if r not in values}
        if len(data) != 1:
            break
        (src,) = data
        sshp = shape(src)
        if sshp is None or len(sshp) != 4 or sshp[0] != shp[0]:  # noqa: PLR2004
            break
        kind = None
        if node.target is aten.convolution.default:
            w = values.get(next(g._refs(p[3][1]), ""))  # noqa: SLF001
            if (
                w is not None
                and not p[3][6]
                and list(p[3][3]) == [1, 1]
                and list(p[3][5]) == [1, 1]
                and w.ndim == 4  # noqa: PLR2004
                and w.shape[2] % 2 == 1
                and list(p[3][4]) == [w.shape[2] // 2, w.shape[3] // 2]
                and shp[2:] == sshp[2:]
            ):
                kind = "conv"
        elif node.target in _WHOLE_MAP:
            break
        elif node.target in g.CL_OPS and node.target is not aten.constant_pad_nd.default and node.target not in _UP:
            ok = shp == sshp
            for r in tensor_refs - {src}:
                c = values.get(r)
                ok = ok and c is not None and (c.ndim < 4 or c.shape[2] == 1)  # noqa: PLR2004
            kind = "ew" if ok else None
        elif node.target in _UP and shp[2] % sshp[2] == 0 and shp[3] % sshp[3] == 0:
            kind = "up"
        if kind is None:
            break
        chain.append(cur)
        kinds[cur] = kind
        cur = src
    if not chain:
        return plan
    entry = cur
    h0 = cast("tuple[int, ...]", shape(entry))[2]
    heights = {n: cast("tuple[int, ...]", shape(n))[2] for n in chain}
    if any(h % h0 for h in heights.values()):
        return plan
    chain.reverse()  # input first
    top = {n: math.prod(cast("tuple[int, ...]", shape(n))) * itemsize for n in chain}
    if max(top.values()) < BAND_MIN_MB << 20:
        return plan
    scale = {n: heights[n] // h0 for n in chain}
    per_row = {n: top[n] // heights[n] for n in chain}  # bytes of one row of a value
    rows = max(1, (BAND_MB << 20) // max(per_row[n] * scale[n] for n in chain))  # entry rows per band
    if rows >= h0:
        return plan
    steps = [step_of[n] for n in chain]
    out_name = out_refs[0]
    cl_out = out_name in cl_values

    def plan_band(a: int, b: int) -> list[tuple[int, int]]:
        """Row range `[lo, hi)` of the entry and of every chain value (its own resolution) for entry rows `[a, b)`."""
        s_out = scale[chain[-1]]
        r = [(a * s_out, b * s_out)]
        for i in range(len(chain) - 1, -1, -1):
            n = chain[i]
            lo, hi = r[0]
            h_in = heights[chain[i - 1]] if i else h0
            if kinds[n] == "conv":
                halo = values[next(iter(g._refs(steps[i][3][1])))].shape[2] // 2  # noqa: SLF001
                r.insert(0, (max(lo - halo, 0), min(hi + halo, h_in)))
            elif kinds[n] == "up":
                f = heights[n] // h_in
                r.insert(0, (lo // f, -(-hi // f)))
            else:
                r.insert(0, (lo, hi))
        return r

    names = {n: [a for a, src in alias.items() if real(src) == n] for n in chain}
    bands = [plan_band(a, min(a + rows, h0)) for a in range(0, h0, rows)]

    def get(env: dict[str, Any], a: Any) -> Any:  # noqa: ANN401
        if isinstance(a, g._Name):  # noqa: SLF001
            return env[a.name]
        if isinstance(a, (list, tuple)):
            return type(a)(get(env, x) for x in a)
        return a

    def run_band(x: mx.array, ranges: list[tuple[int, int]]) -> mx.array:
        env: dict[str, Any] = dict(values)
        env[entry] = x[:, :, ranges[0][0] : ranges[0][1]]
        for i, n in enumerate(chain):
            _, name, f, args, kwargs = steps[i]
            (l_in, h_in_rows), (lo, hi) = ranges[i], ranges[i + 1]
            args = list(args)
            if kinds[n] == "up":
                slab = (h_in_rows - l_in) * (heights[n] // (heights[chain[i - 1]] if i else h0))
                if args[1] is not None:
                    args[1] = [slab, args[1][1]]
                off = lo - l_in * (heights[n] // (heights[chain[i - 1]] if i else h0))
            else:
                off = lo - l_in
            y = f(*[get(env, a) for a in args], **{k: get(env, v) for k, v in kwargs.items()})
            if off or y.shape[2] != hi - lo:
                y = y[:, :, off : off + hi - lo]
            for an in (name, n, *names[n]):  # the step's own name, and every alias of the value
                env[an] = y
        res = env[chain[-1]]
        mx.eval(res)
        return res

    def banded(x: mx.array) -> mx.array:
        outs = [run_band(x, r) for r in bands]
        if cl_out:
            return g._uncl(mx.concatenate([g._cl(o) for o in outs], axis=1))  # noqa: SLF001
        return mx.concatenate(outs, axis=2)

    banded.uncompilable = True  # ty: ignore[unresolved-attribute]  # it evaluates band by band: no `mx.compile`
    cut = {*chain, *(a for n in chain for a in names[n])}
    kept = [p for p in plan if not (p[0] in ("call", "alias") and p[1] in cut)]
    new = ("call", out_name, banded, (g._Name(entry),), {})  # noqa: SLF001
    kept.insert(len(kept) - 1, new)
    return kept
