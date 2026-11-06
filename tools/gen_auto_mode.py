"""Generate `src/fastforks/_core/auto_portable.json`: the models for which `mode="auto"` stays on `portable`.

`auto` takes the kernels backend when it can load. That is a loss where the kernels are not faster. This tool reads the
measured speeds (the fixed-size table of docs/benchmarks.md and docs/data/speed.jsonl) and lists a model when the median
over its measured inputs of `portable ms / kernels ms` is below `MIN_GAIN`. Load time stays free: `auto` reads one small
file. An explicit `mode="kernels"` is never affected.

    uv run python tools/gen_auto_mode.py          # write
    uv run python tools/gen_auto_mode.py --check  # exit 1 if the file is out of date

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if the file is out of date")
    text = render()
    if ap.parse_args().check:
        if not OUT.is_file() or OUT.read_text() != text:
            sys.exit("out of date (run tools/gen_auto_mode.py)")
    else:
        OUT.write_text(text)


if __name__ == "__main__":
    main()
