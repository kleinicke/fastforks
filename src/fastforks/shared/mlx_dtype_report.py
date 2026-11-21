"""`FF_MLX_DTYPE_REPORT=1`: which share of a half-precision replay's heavy compute silently runs in float32.

The MLX port replays a float32 twin, so a float32 table the forward builds (RoPE cos / sin, a resized position
table, a `new_empty` buffer, a literal `.float()`) promotes every op downstream of it. This report runs the replay
eagerly (lazy MLX arrays: no extra evaluation) and, per `GraphNet` input signature, prints once:

- the float32 share of the estimated FLOPs of matmul / addmm / convolution / SDPA / einsum ops, and of the
  activation bytes of all ops;
- the modules whose heavy ops run in float32, by FLOPs;
- the origins: the first float32 values (inputs all non-float32 or none) that feed a promotion of a half-precision
  value, with the module that built them. `fastforks::float32` origins are deliberate float32 islands.

`tools/mlx_dtype_report.py` runs one quick forward of a ref with this on.
"""

from __future__ import annotations

import os
import re
from typing import Any

from fastforks._core.mlx import mx

ON = bool(os.environ.get("FF_MLX_DTYPE_REPORT"))
_HEAVY = ("aten.mm", "aten.bmm", "aten.addmm", "aten.matmul", "aten.baddbmm", "aten.convolution", "aten.einsum")
_SDPA = "scaled_dot_product"
_HALF = (mx.float16, mx.bfloat16)


def meta(nodes: list[Any]) -> dict[str, tuple[str, str]]:
    """Target and innermost module path of each call node (from the export's `nn_module_stack`)."""
    out = {}
    for n in nodes:
        if n.op != "call_function":
            continue
        stack = n.meta.get("nn_module_stack") or {}
        path = next(reversed(stack.values()))[0] if stack else ""
        out[n.name] = (str(n.target), path)
    return out


def _group(path: str) -> str:
    return re.sub(r"\.\d+(?=\.|$)", ".*", path)


def _arrays(a: object) -> list[mx.array]:
    if isinstance(a, mx.array):
        return [a]
    if isinstance(a, (list, tuple)):
        return [y for x in a for y in _arrays(x)]
    return []


def _first(v: object) -> mx.array | None:
    """`v`, or the first element of a tuple-valued op (native_layer_norm, max.dim), if a float array."""
    if isinstance(v, (list, tuple)) and v:
        v = v[0]
    return v if isinstance(v, mx.array) and mx.issubdtype(v.dtype, mx.floating) else None


def _flops(target: str, xs: list[mx.array], out: mx.array) -> float:
    size = out.size
    if _SDPA in target:
        lk = xs[1].shape[-2] if len(xs) > 1 and xs[1].ndim > 1 else out.shape[-2]
        return 4.0 * size * lk
    if "convolution" in target and len(xs) > 1:
        w = xs[1]
        return 2.0 * size * (w.size / max(w.shape[0], 1))
    if ("addmm" in target or "baddbmm" in target) and len(xs) > 2:  # noqa: PLR2004 - (bias, x, w)
        return 2.0 * size * xs[1].shape[-1]
    if xs and xs[0].ndim:
        return 2.0 * size * xs[0].shape[-1]
    return float(size)


class Report:
    """One replay's dtype record (filled by `note`, printed once by `emit`)."""

    def __init__(self, meta: dict[str, tuple[str, str]], dtype: mx.Dtype) -> None:
        """`meta`: from `meta(nodes)`; `dtype`: the replay's."""
        self.meta = meta
        self.dtype = dtype
        self.flops = {"f32": 0.0, "half": 0.0}
        self.bytes = {"f32": 0, "half": 0}
        self.by_module: dict[str, float] = {}
        self.f32_from: dict[str, list[str]] = {}  # float32 value -> its float32 float inputs (empty: an origin)
        self.promotions: list[tuple[str, list[str]]] = []  # ops that mixed half and float32 inputs
        self.first_heavy: str | None = None
        self.done = False

    def note(
        self, name: str, args: list[object], kwargs: dict[str, object], refs: list[str], env: dict[str, Any]
    ) -> None:
        """Record the plan step `name` just run on `args` / `kwargs` (`refs`: its graph inputs' names)."""
        out = env.get(name)
        first = out[0] if isinstance(out, (list, tuple)) and out else out
        if not isinstance(first, mx.array) or not mx.issubdtype(first.dtype, mx.floating):
            return
        xs = [x for x in _arrays(list(args)) + _arrays(list(kwargs.values())) if mx.issubdtype(x.dtype, mx.floating)]
        f32 = first.dtype == mx.float32
        self.bytes["f32" if f32 else "half"] += first.nbytes
        target, path = self.meta.get(name, ("?", ""))
        if any(h in target for h in _HEAVY) or _SDPA in target:
            fl = _flops(target, xs, first)
            ins32 = any(x.dtype == mx.float32 for x in xs)
            key = "f32" if ins32 or f32 else "half"
            self.flops[key] += fl
            if key == "f32":
                self.by_module[path] = self.by_module.get(path, 0.0) + fl
                if self.first_heavy is None:
                    self.first_heavy = name
        if f32:
            src = [n for n in refs if (v := _first(env.get(n))) is not None and v.dtype == mx.float32]
            self.f32_from[name] = src
            if any(x.dtype in _HALF for x in xs) and src:
                self.promotions.append((name, src))

    def _origins(self, name: str, seen: set[str]) -> list[str]:
        stack, found = [name], []
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            src = self.f32_from.get(n)
            if src is None:  # a float32 graph input / constant, or a value the plan did not record
                found.append(n)
                continue
            if not src:
                found.append(n)
            stack.extend(src)
        return found

    def emit(self, label: str = "") -> None:
        """Print the report (once)."""
        if self.done:
            return
        self.done = True
        tot = sum(self.flops.values()) or 1.0
        btot = sum(self.bytes.values()) or 1
        lines = [
            (
                f"mlx dtype report{label} ({self.dtype}): heavy FLOPs float32 {100 * self.flops['f32'] / tot:.1f} %"
                f" of {tot / 1e9:.1f} GFLOP; activation bytes float32 {100 * self.bytes['f32'] / btot:.1f} %"
                f" of {btot / 2**20:.0f} MB"
            ),
        ]
        if self.first_heavy:
            t, p = self.meta.get(self.first_heavy, ("?", ""))
            lines.append(f"  first float32 heavy op: {self.first_heavy} {t} in {p or '?'}")
        by: dict[str, float] = {}
        for path, fl in self.by_module.items():  # repeated blocks together ("blocks.*.attn")
            by[_group(path)] = by.get(_group(path), 0.0) + fl
        for path, fl in sorted(by.items(), key=lambda kv: -kv[1])[:12]:
            lines.append(f"  float32 heavy {100 * fl / tot:5.1f} %  {path or '?'}")
        seen: set[str] = set()
        origins: dict[tuple[str, str], list[Any]] = {}  # (target, module group) -> [first value, count]
        for _, src in self.promotions:
            for s in src:
                for o in self._origins(s, seen):
                    t, path = self.meta.get(o, ("?", ""))
                    origins.setdefault((t, _group(path)), [o, 0])[1] += 1
        lines.append(f"  {len(self.promotions)} ops promote a half-precision value to float32; origins:")
        for (t, path), (o, n) in list(origins.items())[:15]:
            tag = "  (deliberate island)" if "fastforks.float32" in t else ""
            tag = tag or ("  (folded constant / input)" if o not in self.f32_from else "")
            lines.append(f"    {o} {t} in {path or '?'} x{n}{tag}")
        print("\n".join(lines), flush=True)  # noqa: T201 - FF_MLX_DTYPE_REPORT=1
