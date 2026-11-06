# ruff: noqa: E501,PTH123,SIM115  # benchmark driver
"""Markdown table from `bench_cpu.py` JSONL files: median ms per (model, threads, side, variant, kind) with load flags."""

from __future__ import annotations

import json
import sys
from collections import defaultdict

rows = [json.loads(line) for f in sys.argv[1:] for line in open(f)]
cells: dict = defaultdict(list)
for r in rows:
    cells[(r["model"], r["threads"], r["kind"], r["side"], r["variant"])].append(r)
print("| model | thr | kind | side | variant | median ms | first ms | maxRSS MB | err_rel | foreign cores | n |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for k in sorted(cells):
    r = cells[k][-1]  # last = redo
    err = f"{r['err_rel']:.1e}" if "err_rel" in r else ""
    print(
        f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} | {r['median_s'] * 1e3:.1f} | {r['first_s'] * 1e3:.0f} | "
        f"{r['maxrss_mb']:.0f} | {err} | {r['foreign_cores']} | {r['n']} |"
    )
