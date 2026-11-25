"""Generate `src/fastforks/_core/auto_portable.json`: the models for which `mode="auto"` stays on `portable`.

`auto` takes the kernels backend when it can load. That is a loss where the kernels are not faster. This tool reads the
measured speeds (the fixed-size table of docs/benchmarks.md and docs/data/speed.jsonl) and lists a model when the median
over its measured inputs of `portable ms / kernels ms` is below `MIN_GAIN`. Load time stays free: `auto` reads one small
file. An explicit `mode="kernels"` is never affected. On Windows `auto` reads the `windows` list instead: the same rule
over the rows of `WINDOWS_DEVICE` (the reference card under WDDM, where every launch costs more) for the models that
survey measured, over the reference rows for the others.

The Apple table selects among passing, comparable MPS and MLX survey runs (only in a dtype the card still declares:
`mps_dtypes` for portable MPS, the MLX dtypes for its MLX port). A run whose golden passed is as good as any other, so
float16 is taken like on CUDA wherever it is validated and pays. One rule decides every step (`_route_score`): a
candidate replaces the path chosen so far if it is at least `MIN_GAIN` faster, or about as fast (at most
`TRADE_SLOWDOWN` slower) while needing at most `TRADE_MEMORY` of its peak memory (peak footprint) on every workload. A
faster candidate may use up to `MEM_SLACK` more memory; beyond that the speed must repay the memory (time ratio times
memory ratio still `MIN_GAIN` better). Without a `dtype`, `auto` goes through three steps per model revision:

- `mps_dtype`: portable MPS computes in float16 if that run beats the float32 run by the rule, or if float32 runs out
  of memory on every workload float16 was measured on.
- `mps_auto`: the native Metal kernels replace the portable path chosen above (the `""` request). With a `dtype`,
  they must beat portable MPS in that dtype.
- `auto`: the MLX port replaces the MPS path chosen above. MLX compiles per input shape, so its first call is slow: the
  route also needs its first call to take at most `COLD_MPS_COLD` first calls plus `COLD_MPS_WARM` warm runs of the
  MPS path it replaces, on every workload (`device="mlx"` always runs MLX). `dtypes`: the same for a requested
  `dtype`, against the fastest MPS path in that dtype.
- `mlx` and `mlx_default` select the fastest passing MLX path (`device="mlx"` without a `dtype`, and a `dtype` MPS
  has no implementation for).
- `passed` lists, per MLX mode, the dtypes with a passing MLX run (golden checked): what `contract()` reports as
  validated on MLX, and what tests/test_mlx_validated.py holds the card declarations against.

Only `auto` crosses from PyTorch MPS to MLX, so the table holds no route for an explicit `mode="kernels"`.

    uv run python tools/gen_auto_mode.py          # write
    uv run python tools/gen_auto_mode.py --check  # exit 1 if a file is out of date

tests/test_auto_mode.py runs the check.
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
import sys
from pathlib import Path

import gen_docs
from fastforks._core import registry
from fastforks._core.errors import ModelNotFoundError

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "fastforks" / "_core" / "auto_portable.json"
APPLE_OUT = OUT.with_name("apple_mlx.json")
APPLE_DEVICE = "apple-m1-pro-16gb"  # the Mac whose MPS and MLX survey rows decide apple_mlx.json
APPLE_MLX_DEVICE = "apple-m1-pro-mlx-16gb"
WINDOWS_DEVICE = "rtx-5060-ti-windows-16gb"  # the reference card under Windows: its rows decide `auto` there
MIN_GAIN = 1.05  # kernels must be at least 5 % faster, else the lighter portable backend is chosen
MEM_SLACK = 1.02  # a faster Apple route may use 2 % more peak memory than the path it replaces (measurement noise)
TRADE_SLOWDOWN = 1.05  # a route may be up to 5 % slower than that path (the same speed, within noise) ...
TRADE_MEMORY = 0.90  # ... if it needs at most 90 % of its peak memory (a saving of at least 10 %)
COLD_MPS_COLD = 1.0  # `auto` takes MLX only if its first call is at most this many first calls of the MPS path ...
COLD_MPS_WARM = 2.0  # ... plus this many warm MPS runs
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


def _survey_ratios(device: str) -> dict[str, list[float]]:
    """`{model id: [portable ms / kernels ms, ...]}` over the inputs a survey device measured in both modes."""
    pairs: dict[tuple[str, str], dict[str, float]] = {}
    path = ROOT / "docs" / "data" / "speed" / f"{device}.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else ():
        row = json.loads(line)
        if row.get("status") == "ok" and row.get("golden") != "fail" and row["mode"] in ("portable", *KERNEL_MODES):
            ms = pairs.setdefault((row["ref"].split("@")[0], row["input"]), {})
            mode = "portable" if row["mode"] == "portable" else "k"
            ms[mode] = min(ms.get(mode, row["ms"]), row["ms"])
    found: dict[str, list[float]] = {}
    for (model_id, _), ms in pairs.items():
        if "portable" in ms and "k" in ms:
            found.setdefault(model_id, []).append(ms["portable"] / ms["k"])
    return found


def render() -> str:
    reference = _ratios()
    slow = sorted(ref for ref, r in reference.items() if statistics.median(r) < MIN_GAIN)
    # on Windows a model the Windows survey measured is decided by those rows, every other one by the reference rows
    windows = {**reference, **_survey_ratios(WINDOWS_DEVICE)}
    slow_windows = sorted(ref for ref, r in windows.items() if statistics.median(r) < MIN_GAIN)
    data = {
        "min_gain": MIN_GAIN,
        "portable": slow,
        "windows": {"measured_on": WINDOWS_DEVICE, "portable": slow_windows},
    }
    return json.dumps(data, indent=1) + "\n"


Option = tuple[str, str]  # (mode, dtype)
Workload = tuple[str, str, str]  # (input, scene, protocol)
Run = tuple[float, float | None, float]  # (ms, peak footprint MB; None: not recorded, first call s)


@functools.cache
def _declared_dtypes(ref: str, mode: str, device: str) -> tuple[str, ...] | None:
    """The dtypes the card of `ref` declares for `mode` on the survey `device`; None: no card, or no such list.

    The MLX port's dtypes of the mode, and the portable `mps_dtypes` (the Metal kernels have one dtype, not a list).
    """
    try:
        card, _ = registry.resolve_any(ref.split("@", maxsplit=1)[0])
    except ModelNotFoundError:
        return None
    if device == APPLE_MLX_DEVICE:
        return card.mlx_dtypes(mode)
    return card.portable_mps_dtypes or None if mode == "portable" else None  # a family not validated on MPS: no list


def _apple_options(device: str) -> dict[str, dict[Option, dict[Workload, Run]]]:
    """Passing survey rows keyed by revision, implementation and workload."""
    models: dict[str, dict[Option, dict[Workload, Run]]] = {}
    for line in (ROOT / "docs" / "data" / "speed" / f"{device}.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("status") != "ok" or row.get("golden") != "pass" or row["mode"] not in ("portable", "kernels"):
            continue
        dtype = (row.get("backend") or {}).get("dtype") or row["dtype"]
        ref = row["ref"]
        if "@" not in ref:
            ref += "@1"  # historical unversioned rows predate checkpoint revisions
        declared = _declared_dtypes(ref, row["mode"], device)
        if declared is not None and dtype not in declared:
            continue  # a dtype the card does not (or no longer) declare here: not validated, whatever the row says
        workload = (row["input"], row.get("scene", ""), row.get("protocol_id", ""))
        models.setdefault(ref, {}).setdefault((row["mode"], dtype), {})[workload] = (
            row["ms"],
            row.get("peak_footprint_mb"),
            max(row.get("first_call_s") or 0, row["ms"] / 1000),
        )
    return models


def _mps_oom() -> dict[str, dict[str, set[Workload]]]:
    """Workloads per ref and dtype where the portable MPS path ran out of memory (and so has no row to beat)."""
    oom: dict[str, dict[str, set[Workload]]] = {}
    for line in (ROOT / "docs" / "data" / "speed" / f"{APPLE_DEVICE}.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("status") != "oom" or row["mode"] != "portable":
            continue
        ref = row["ref"] if "@" in row["ref"] else row["ref"] + "@1"
        workload = (row["input"], row.get("scene", ""), row.get("protocol_id", ""))
        oom.setdefault(ref, {}).setdefault(row["dtype"], set()).add(workload)
    return oom


def _oom_routes(
    candidates: dict[Option, dict[Workload, Run]],
    native: dict[Option, dict[Workload, Run]],
    oom: dict[str, set[Workload]],
) -> dict[str, Option]:
    """MLX routes for the dtypes MPS cannot run: every MLX workload of that dtype is an MPS out-of-memory one.

    MPS has no row there to compare against (so `_winners` finds none), but MLX running at all beats an OOM error.
    """
    routes = {}
    for dtype, failed in oom.items():
        options = [(o, rows) for o, rows in candidates.items() if o[1] == dtype and rows]
        if not options or any(o[1] == dtype and rows.keys() & failed for o, rows in native.items()):
            continue
        if all(rows.keys() <= failed for _, rows in options):
            routes[dtype] = min(options, key=lambda item: statistics.median(r[0] for r in item[1].values()))[0]
    return routes


def _route_score(ratios: list[float], costs: list[float | None], *, memory: bool) -> float | None:
    """Median `ms / baseline ms` when the route qualifies, else None.

    `costs`: `peak memory / baseline peak memory` per workload (None: not recorded). Without the memory rule the route
    qualifies when at least `MIN_GAIN` faster. With it, it also needs a known memory on every workload, and then:

    - at most `MEM_SLACK` of the baseline's memory: at least `MIN_GAIN` faster, or at most `TRADE_SLOWDOWN` slower
      while needing at most `TRADE_MEMORY` of the baseline's memory on every workload;
    - more memory: the speed repays it, i.e. time ratio times the largest memory ratio is still `MIN_GAIN` better.
    """
    if not ratios:
        return None
    score = statistics.median(ratios)
    if not memory:
        return score if score * MIN_GAIN <= 1 else None
    known = [c for c in costs if c is not None]
    if len(known) < len(costs):
        return None
    cost = max(known)
    if cost > MEM_SLACK:
        return score if score * cost * MIN_GAIN <= 1 else None
    return score if score * MIN_GAIN <= 1 or (score <= TRADE_SLOWDOWN and cost <= TRADE_MEMORY) else None


def _winners(
    candidates: dict[Option, dict[Workload, Run]],
    alternatives: dict[Option, dict[Workload, Run]],
    *,
    memory: bool = False,
    cold: bool = False,
) -> dict[str, Option]:
    """Winning implementation for each requested dtype, plus the unconstrained request under the empty key.

    A candidate is compared per workload with the fastest alternative (of the requested dtype) and must qualify by
    `_route_score`; `memory` turns its memory rule on. `cold`: and its first call must take at most `COLD_MPS_COLD`
    first calls plus `COLD_MPS_WARM` warm runs of that alternative on every workload.
    """
    selected = {}
    for dtype in ("", *sorted({d for _, d in candidates})):
        timings = []
        for option, rows in candidates.items():
            if dtype and option[1] != dtype:
                continue
            if not alternatives:
                score = statistics.median(run[0] for run in rows.values())
            else:
                ratios, costs = [], []
                for key, (ms, mem, first_s) in rows.items():
                    baseline = [
                        other[key]
                        for (_, d), other in alternatives.items()
                        if (not dtype or d == dtype) and key in other
                    ]
                    if not baseline:
                        continue
                    fastest_ms, fastest_mem, fastest_first = min(baseline)  # the path the route would replace
                    if cold and first_s > COLD_MPS_COLD * fastest_first + COLD_MPS_WARM * fastest_ms / 1000:
                        ratios = []  # the first call costs too many MPS runs
                        break
                    ratios.append(ms / fastest_ms)
                    costs.append(mem / fastest_mem if mem is not None and fastest_mem else None)
                score = _route_score(ratios, costs, memory=memory)
                if score is None:
                    continue
            timings.append((score, option))
        if timings:
            selected[dtype] = min(timings)[1]
    return selected


def _mps_choice(
    native: dict[Option, dict[Workload, Run]], oom: dict[str, set[Workload]]
) -> tuple[Option | None, dict[str, Option]]:
    """What `auto` runs on PyTorch MPS: the portable path without a `dtype`, and the Metal requests that beat portable.

    Returns the portable `(mode, dtype)` (None without a passing portable run) and `{request: kernels option}` with
    the empty key for the request without a dtype (the Metal kernels against that portable path) and one key per
    dtype (against portable in that dtype).
    """
    kernels = {option: rows for option, rows in native.items() if option[0] == "kernels"}
    portable = {option: rows for option, rows in native.items() if option[0] == "portable"}
    full = {o: rows for o, rows in portable.items() if o[1] == "float32"}
    half = {o: rows for o, rows in portable.items() if o[1] == "float16"}
    chosen = full
    if half and full and "" in _winners(half, full, memory=True):
        chosen = half
    elif half and not full and all(rows.keys() <= oom.get("float32", set()) for rows in half.values()):
        chosen = half  # float32 runs out of memory wherever float16 was measured
    metal = {d: o for d, o in _winners(kernels, portable, memory=True).items() if d} if portable else {}
    if chosen and (best := _winners(kernels, chosen, memory=True).get("")):
        metal[""] = best
    return next(iter(chosen), None), metal


def _mlx_auto(
    accelerated: dict[Option, dict[Workload, Run]],
    native: dict[Option, dict[Workload, Run]],
    chosen: Option | None,
    oom: dict[str, set[Workload]],
) -> dict[str, Option]:
    """The MLX routes of `auto`: per requested dtype, and under the empty key against the MPS path `chosen`.

    A route needs a comparable MPS run, except where MPS runs out of memory (`_oom_routes`).
    """
    auto = {d: o for d, o in _winners(accelerated, native, memory=True, cold=True).items() if d} if native else {}
    if chosen and (best := _winners(accelerated, {chosen: native[chosen]}, memory=True, cold=True).get("")):
        auto[""] = best
    return auto | {d: o for d, o in _oom_routes(accelerated, native, oom).items() if d not in auto}


def render_apple() -> str:
    mps, mlx, oom = _apple_options(APPLE_DEVICE), _apple_options(APPLE_MLX_DEVICE), _mps_oom()
    models: dict[str, object] = {}
    for ref in sorted(mps.keys() | mlx.keys()):
        native, accelerated = mps.get(ref, {}), mlx.get(ref, {})
        kernels = {option: rows for option, rows in native.items() if option[0] == "kernels"}
        portable, metal = _mps_choice(native, oom.get(ref, {}))
        chosen = metal.get("") or portable or next(iter(kernels), None)  # what `auto` runs on MPS without a dtype
        auto = _mlx_auto(accelerated, native, chosen, oom.get(ref, {}))
        only_mlx = _winners(accelerated, {})
        entry: dict[str, object] = {}
        if "" in auto:
            entry["auto"] = list(auto.pop(""))
        if auto:
            entry["dtypes"] = {dtype: option[0] for dtype, option in auto.items()}
        if "" in only_mlx:
            entry["mlx_default"] = list(only_mlx.pop(""))
        if only_mlx:
            entry["mlx"] = {dtype: option[0] for dtype, option in only_mlx.items()}
        if metal:
            entry["mps_auto"] = sorted(metal)
        if portable and portable[1] != "float32":
            entry["mps_dtype"] = portable[1]  # also where Metal or MLX is taken: what runs if that cannot load
        if accelerated:
            entry["passed"] = {
                mode: sorted(d for m, d in accelerated if m == mode) for mode in sorted({m for m, _ in accelerated})
            }
        if entry:
            models[ref] = entry
    data = {"min_gain": MIN_GAIN, "measured_on": [APPLE_DEVICE, APPLE_MLX_DEVICE], "models": models}
    return json.dumps(data, indent=1) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if the file is out of date")
    files = {OUT: render(), APPLE_OUT: render_apple()}
    if ap.parse_args().check:
        stale = [p.name for p, text in files.items() if not p.is_file() or p.read_text(encoding="utf-8") != text]
        if stale:
            sys.exit(f"out of date: {', '.join(stale)} (run tools/gen_auto_mode.py)")
    else:
        for p, text in files.items():
            p.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
