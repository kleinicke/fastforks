"""Generate `src/fastforks/_core/auto_portable.json`: the models for which `mode="auto"` stays on `portable`.

`auto` takes the kernels backend when it can load. That is a loss where the kernels are not faster. This tool reads the
measured speeds (the fixed-size table of docs/benchmarks.md and docs/data/speed.jsonl) and lists a model when the median
over its measured inputs of `portable ms / kernels ms` is below `MIN_GAIN`. Load time stays free: `auto` reads one small
file. An explicit `mode="kernels"` is never affected.

It also writes `src/fastforks/_core/apple_mlx.json`: where a load on Apple MPS takes the family's MLX port instead
(`load(device="mps")` with `mode="auto"` or `"kernels"`), from the MPS and MLX survey rows of `APPLE_DEVICE`. Per
model id:

- `auto`: `[mode, dtype]` of the fastest MLX option, if it beats the fastest MPS option (any mode and dtype) by
  `MIN_GAIN`; `mode="auto"` without a dtype takes it.
- `dtypes`: `{dtype: mode}`, the fastest MLX option at a dtype that beats MPS at that dtype by `MIN_GAIN`, or that
  MPS does not run; `mode="auto"` with that dtype takes it.
- `kernels`: dtypes at which the MLX kernels beat the PyTorch Metal kernels by `MIN_GAIN` (only for families with
  both); `mode="kernels"` takes MLX at those. Without Metal kernels at the dtype, `kernels` always takes MLX's.

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


def _apple_options(device: str) -> dict[str, dict[Option, float]]:
    """`{model id: {(mode, dtype): median ms}}` of a device's passing survey rows (dtype: what actually ran)."""
    by: dict[str, dict[Option, list[float]]] = {}
    for line in (ROOT / "docs" / "data" / "speed" / f"{device}.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("status") != "ok" or r.get("golden") == "fail" or r["mode"] not in ("portable", "kernels"):
            continue
        dtype = (r.get("backend") or {}).get("dtype") or r["dtype"]
        by.setdefault(r["ref"].split("@")[0], {}).setdefault((r["mode"], dtype), []).append(r["ms"])
    return {ref: {o: statistics.median(v) for o, v in opts.items()} for ref, opts in by.items()}


def _fastest(opts: dict[Option, float], *, mode: str | None = None, dtype: str | None = None) -> tuple[Option, float]:
    """The fastest option, optionally of one mode / dtype; `(("", ""), inf)` if there is none."""
    found = [(ms, o) for o, ms in opts.items() if (mode is None or o[0] == mode) and (dtype is None or o[1] == dtype)]
    return (min(found)[1], min(found)[0]) if found else (("", ""), float("inf"))


def render_apple() -> str:
    mps, mlx = _apple_options(APPLE_DEVICE), _apple_options(APPLE_MLX_DEVICE)
    models: dict[str, dict[str, object]] = {}
    for ref in sorted(mlx):
        mine, theirs = mlx[ref], mps.get(ref, {})
        entry: dict[str, object] = {}
        best, ms = _fastest(mine)
        if ms * MIN_GAIN <= _fastest(theirs)[1]:
            entry["auto"] = list(best)
        dtypes = {}
        for dtype in sorted({d for _, d in mine}):
            (mode, _), ms = _fastest(mine, dtype=dtype)
            if ms * MIN_GAIN <= _fastest(theirs, dtype=dtype)[1]:
                dtypes[dtype] = mode
        if dtypes:
            entry["dtypes"] = dtypes
        kernels = [
            d
            for d in sorted({d for m, d in mine if m == "kernels"})
            if (k := _fastest(theirs, mode="kernels", dtype=d)[1]) < float("inf")
            and _fastest(mine, mode="kernels", dtype=d)[1] * MIN_GAIN <= k
        ]
        if kernels:
            entry["kernels"] = kernels
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
