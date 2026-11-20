"""Generate `src/fastforks/_core/auto_portable.json`: the models for which `mode="auto"` stays on `portable`.

`auto` takes the kernels backend when it can load. That is a loss where the kernels are not faster. This tool reads the
measured speeds (the fixed-size table of docs/benchmarks.md and docs/data/speed.jsonl) and lists a model when the median
over its measured inputs of `portable ms / kernels ms` is below `MIN_GAIN`. Load time stays free: `auto` reads one small
file. An explicit `mode="kernels"` is never affected.

The Apple table selects among passing, comparable MPS and MLX survey runs. Per model revision:

- `auto` and `dtypes` select MLX if it beats the fastest MPS path by `MIN_GAIN` and does not use more memory
  (peak footprint, within `MEM_SLACK`) than that path. It is also selected when at most `TRADE_SLOWDOWN` slower if it
  needs at most `TRADE_MEMORY` of that path's peak memory on every workload.
- `kernels` and `kernels_default` select MLX if it beats the native MPS kernel by `MIN_GAIN`, likewise in memory.
- `mlx` and `mlx_default` select the fastest passing MLX path where it is the only optimized backend.
- `mps_auto` records which native MPS kernel requests beat portable MPS by `MIN_GAIN`.

    uv run python tools/gen_auto_mode.py          # write
    uv run python tools/gen_auto_mode.py --check  # exit 1 if a file is out of date

tests/test_auto_mode.py runs the check.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import gen_docs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "fastforks" / "_core" / "auto_portable.json"
APPLE_OUT = OUT.with_name("apple_mlx.json")
APPLE_DEVICE = "apple-m1-pro-16gb"  # the Mac whose MPS and MLX survey rows decide apple_mlx.json
APPLE_MLX_DEVICE = "apple-m1-pro-mlx-16gb"
MIN_GAIN = 1.05  # kernels must be at least 5 % faster, else the lighter portable backend is chosen
MEM_SLACK = 1.02  # an MLX route may use at most 2 % more peak memory than the MPS path it replaces (measurement noise)
TRADE_SLOWDOWN = 1.01  # ... or be up to 1 % slower than that path ...
TRADE_MEMORY = 0.90  # ... if it needs at most 90 % of its peak memory (a saving of more than 10 %)
KERNEL_MODES = ("kernels",)  # the mode column of the data files


def _ratios() -> dict[str, list[float]]:
    """`{model id: [portable ms / kernels ms, ...]}` over every measured input."""
    found: dict[str, list[float]] = {}
    fixed = gen_docs._fixed_sizes()  # noqa: SLF001 - sibling tool
    for (ref, mode), cells in fixed.items():
        if mode != "portable":
            continue
        kernels = next((fixed[ref, m] for m in KERNEL_MODES if (ref, m) in fixed), {})
        for size, cell in cells.items():
            other = kernels.get(size)
            if isinstance(cell, tuple) and isinstance(other, tuple):
                found.setdefault(ref.split("@")[0], []).append(cell[0] / other[0])
    pairs: dict[tuple[str, str], dict[str, float]] = {}
    for ref, rows in gen_docs._speed_rows().items():  # noqa: SLF001
        for row in rows:
            if row["mode"] in ("portable", *KERNEL_MODES):
                pairs.setdefault((ref, row["input"]), {})[row["mode"] if row["mode"] == "portable" else "k"] = row["ms"]
    for (ref, _), ms in pairs.items():
        if "portable" in ms and "k" in ms:
            found.setdefault(ref.split("@")[0], []).append(ms["portable"] / ms["k"])
    return found


def render() -> str:
    slow = sorted(ref for ref, r in _ratios().items() if statistics.median(r) < MIN_GAIN)
    return json.dumps({"min_gain": MIN_GAIN, "portable": slow}, indent=1) + "\n"


Option = tuple[str, str]  # (mode, dtype)
Workload = tuple[str, str, str]  # (input, scene, protocol)
Run = tuple[float, float | None]  # (ms, peak footprint MB; None: not recorded)


def _apple_options(device: str) -> dict[str, dict[Option, dict[Workload, Run]]]:
    """Passing survey rows keyed by revision, implementation and workload."""
    models: dict[str, dict[Option, dict[Workload, Run]]] = {}
    for line in (ROOT / "docs" / "data" / "speed" / f"{device}.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("status") != "ok" or row.get("golden") != "pass" or row["mode"] not in ("portable", "kernels"):
            continue
        dtype = (row.get("backend") or {}).get("dtype") or row["dtype"]
        ref = row["ref"]
        if "@" not in ref:
            ref += "@1"  # historical unversioned rows predate checkpoint revisions
        workload = (row["input"], row.get("scene", ""), row.get("protocol_id", ""))
        models.setdefault(ref, {}).setdefault((row["mode"], dtype), {})[workload] = (
            row["ms"],
            row.get("peak_footprint_mb"),
        )
    return models


def _route_score(ratios: list[float], savings: list[float | None], *, memory: bool) -> float | None:
    """Median `ms / baseline ms` when the route qualifies, else None.

    It qualifies when at least `MIN_GAIN` faster, or (with the memory rule) at most `TRADE_SLOWDOWN` slower while
    needing at most `TRADE_MEMORY` of the baseline's peak memory on every workload.
    """
    if not ratios:
        return None
    score = statistics.median(ratios)
    if score * MIN_GAIN <= 1:
        return score
    lean = memory and score <= TRADE_SLOWDOWN and all(r is not None and r <= TRADE_MEMORY for r in savings)
    return score if lean else None


def _winners(
    candidates: dict[Option, dict[Workload, Run]],
    alternatives: dict[Option, dict[Workload, Run]],
    *,
    memory: bool = False,
) -> dict[str, Option]:
    """Winning implementation for each requested dtype, plus the unconstrained request under the empty key.

    `memory`: a candidate must also stay within `MEM_SLACK` of the fastest alternative's peak footprint on every
    workload.
    """
    selected = {}
    for dtype in ("", *sorted({d for _, d in candidates})):
        timings = []
        for option, rows in candidates.items():
            if dtype and option[1] != dtype:
                continue
            if not alternatives:
                score = statistics.median(ms for ms, _ in rows.values())
            else:
                ratios, savings = [], []
                for key, (ms, mem) in rows.items():
                    baseline = [
                        other[key]
                        for (_, d), other in alternatives.items()
                        if (not dtype or d == dtype) and key in other
                    ]
                    if baseline:
                        fastest_ms, fastest_mem = min(baseline)  # the path the MLX route would replace
                        if memory and (mem is None or fastest_mem is None or mem > fastest_mem * MEM_SLACK):
                            ratios = []  # more memory (or unknown) on one workload: no route
                            break
                        ratios.append(ms / fastest_ms)
                        savings.append(mem / fastest_mem if mem is not None and fastest_mem else None)
                score = _route_score(ratios, savings, memory=memory)
                if score is None:
                    continue
            timings.append((score, option))
        if timings:
            selected[dtype] = min(timings)[1]
    return selected


def render_apple() -> str:
    mps, mlx = _apple_options(APPLE_DEVICE), _apple_options(APPLE_MLX_DEVICE)
    models: dict[str, object] = {}
    for ref in sorted(mps.keys() | mlx.keys()):
        native, accelerated = mps.get(ref, {}), mlx.get(ref, {})
        kernels = {option: rows for option, rows in native.items() if option[0] == "kernels"}
        baseline = {option: rows for option, rows in native.items() if option[0] == "portable"}
        # Require a comparable MPS run when automatically promoting an MLX route.
        auto = _winners(accelerated, native, memory=True) if native else {}
        optimized = _winners(accelerated, kernels, memory=True) if kernels else {}
        only_mlx = _winners(accelerated, {})
        metal = _winners(kernels, baseline) if baseline else {}
        entry: dict[str, object] = {}
        if "" in auto:
            entry["auto"] = list(auto.pop(""))
        if auto:
            entry["dtypes"] = {dtype: option[0] for dtype, option in auto.items()}
        for key, winners in (("kernels", optimized), ("mlx", only_mlx)):
            if "" in winners:
                entry[f"{key}_default"] = list(winners.pop(""))
            if winners:
                entry[key] = {dtype: option[0] for dtype, option in winners.items()}
        if metal:
            entry["mps_auto"] = list(metal)
        if entry:
            models[ref] = entry
    data = {"min_gain": MIN_GAIN, "measured_on": [APPLE_DEVICE, APPLE_MLX_DEVICE], "models": models}
    return json.dumps(data, indent=1) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if the file is out of date")
    files = {OUT: render(), APPLE_OUT: render_apple()}
    if ap.parse_args().check:
        stale = [p.name for p, text in files.items() if not p.is_file() or p.read_text() != text]
        if stale:
            sys.exit(f"out of date: {', '.join(stale)} (run tools/gen_auto_mode.py)")
    else:
        for p, text in files.items():
            p.write_text(text)


if __name__ == "__main__":
    main()
