"""Rank kernel work from the primary RTX 5060 Ti benchmark tables. Run with no arguments for one TSV row per variant.

Compare the newest successful row for each matching measurement setup. A variant with multiple setups is represented
by the largest historical kernels/compile ratio. These rows describe the revisions measured on their dates.
The GPU survey files for other devices use separate protocols and are outside this inventory.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from fastforks._core import registry

DATA = Path(__file__).resolve().parents[1] / "docs" / "data"
SETUP = ("gpu", "torch", "protocol", "scene", "input")
WORKLOAD = ("frames", "views", "queries", "people", "batch", "scope", "part", "timed", "options")


def timings(file: str, mode: str) -> tuple[dict, set[str]]:
    """Latest matching measurements and ids with any successful timing in the file."""
    newest = {}
    measured = set()
    for line in (DATA / file).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        ms = row.get("ms")
        if row.get("mode") != mode or row.get("side", "lib") != "lib" or not isinstance(ms, (int, float)):
            continue
        if (
            row.get("fails")
            or row.get("error")
            or row.get("status") not in (None, "ok")
            or not math.isfinite(ms)
            or ms <= 0
        ):
            continue
        ref = row["ref"]
        measured.add(ref)
        dtype = (row.get("backend") or {}).get("dtype")
        setup = (*tuple(row.get(k) for k in SETUP), dtype)
        if not all(setup):
            continue
        workload = {k: row[k] for k in WORKLOAD if row.get(k) is not None and row[k] != {}}
        key = (ref, setup, json.dumps(workload, sort_keys=True))
        if key not in newest or row["date"] >= newest[key]["date"]:
            newest[key] = row
    return newest, measured


def main() -> None:
    """Write a tab-separated inventory, with directly comparable slower kernels first."""
    compiled, _ = timings("speed_compile.jsonl", "compile")
    kernels, measured = timings("speed.jsonl", "kernels")
    portable, _ = timings("speed.jsonl", "portable")
    pairs = defaultdict(list)
    for key in kernels.keys() & compiled.keys():
        k, c = kernels[key], compiled[key]
        pairs[key[0]].append((k["ms"] / c["ms"], key, k, c))

    rows = []
    for task in registry.TASKS:
        for ref, (card, v) in registry.cards(task).items():
            enabled = "kernels" in registry.variant_card(card, v).modes
            common = [task, card.family, ref, ",".join(card.kernel_libs) if enabled else ""]
            if not enabled or not pairs[ref]:
                status = "unavailable" if not enabled else "unmatched" if ref in measured else "unmeasured"
                rows.append((0, [*common, status, *([""] * 21)]))
                continue
            ratio, key, k, c = max(pairs[ref], key=lambda p: (p[0], p[1]))
            _, setup, workload = key
            gpu, torch_version, protocol, scene, resolution, dtype = setup
            port = portable.get(key)
            rows.append(
                (
                    ratio,
                    [
                        *common,
                        "matched",
                        f"{ratio:.3f}",
                        k["ms"],
                        c["ms"],
                        resolution,
                        scene,
                        dtype,
                        gpu,
                        torch_version,
                        k["date"],
                        c["date"],
                        protocol,
                        workload,
                        port["ms"] if port else "",
                        f"{k['ms'] / port['ms']:.3f}" if port else "",
                        k.get("peak_alloc_mb", ""),
                        c.get("peak_alloc_mb", ""),
                        (port or {}).get("peak_alloc_mb", ""),
                        k.get("peak_reserved_mb", ""),
                        c.get("peak_reserved_mb", ""),
                        (port or {}).get("peak_reserved_mb", ""),
                        str(k["commit"] == c.get("commit") == (port or {}).get("commit"))
                        if k.get("commit") and c.get("commit") and (port or {}).get("commit")
                        else "",
                    ],
                )
            )
    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    writer.writerow(
        [
            "task",
            "family",
            "variant",
            "libraries",
            "status",
            "kernels_over_compile",
            "kernels_ms",
            "compile_ms",
            "input",
            "scene",
            "dtype",
            "gpu",
            "torch",
            "kernels_date",
            "compile_date",
            "protocol",
            "workload",
            "portable_ms",
            "kernels_over_portable",
            "kernels_peak_alloc_mib",
            "compile_peak_alloc_mib",
            "portable_peak_alloc_mib",
            "kernels_peak_reserved_mib",
            "compile_peak_reserved_mib",
            "portable_peak_reserved_mib",
            "same_commit",
        ]
    )
    for _, row in sorted(rows, key=lambda item: (-item[0], item[1][:3])):
        writer.writerow(row)


if __name__ == "__main__":
    main()
