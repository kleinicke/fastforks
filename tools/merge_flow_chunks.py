"""Merge bounded KITTI flow runs after verifying complete, non-overlapping dataset coverage."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from _data import DATA

SAME = (
    "ref",
    "task",
    "tier",
    "mode",
    "dtype",
    "how",
    "dataset",
    "column",
    "commit",
    "source_id",
    "gpu",
    "torch",
    "backend",
    "note",
)
KITTI_TRAIN_PAIRS = 200


def _collect_samples(rows: list[dict]) -> dict[str, dict]:
    """Check chunk provenance and per-pair statistics before combining them."""
    first = rows[0]
    samples: dict[str, dict] = {}
    for row in rows:
        if any(row.get(k) != first.get(k) for k in SAME):
            msg = "chunks have incompatible model, backend, or environment provenance"
            raise ValueError(msg)
        pairs = row.get("pair_stats", [])
        if len(pairs) != row.get("n"):
            msg = "chunk has missing per-pair statistics"
            raise ValueError(msg)
        for sample in pairs:
            key = sample["id"]
            if key in samples:
                msg = f"duplicate pair {key}"
                raise ValueError(msg)
            if (
                not math.isfinite(sample["epe"])
                or not math.isfinite(sample["ms"])
                or sample["valid"] <= 0
                or not 0 <= sample["outliers"] <= sample["valid"]
            ):
                msg = f"invalid statistics for pair {key}"
                raise ValueError(msg)
            samples[key] = sample
    return samples


def merge(rows: list[dict], expected_ids: set[str]) -> dict:
    """Merge per-pair statistics; reject missing, duplicate or incompatible measurements."""
    if not rows or not expected_ids:
        msg = "need measurement chunks and a nonempty dataset"
        raise ValueError(msg)
    first = rows[0]
    if not first.get("source_id"):
        msg = "chunks require an immutable source identifier (eval_flow.py --source-id)"
        raise ValueError(msg)
    if first.get("task") != "flow" or first.get("dataset") != "kitti2015-train":
        msg = "chunks must use the KITTI 2015 training flow protocol"
        raise ValueError(msg)
    samples = _collect_samples(rows)
    if set(samples) != expected_ids:
        msg = (
            f"dataset coverage differs: {len(expected_ids - samples.keys())} missing, "
            f"{len(samples.keys() - expected_ids)} unexpected"
        )
        raise ValueError(msg)
    values = list(samples.values())
    metrics = {
        "epe": round(statistics.mean(s["epe"] for s in values), 4),
        "fl_all": round(100 * sum(s["outliers"] for s in values) / sum(s["valid"] for s in values), 3),
    }
    out = {k: v for k, v in first.items() if k not in {"pair_stats", "command", "date"}}
    out.update(metrics)
    out.update(
        n=len(values),
        metrics=metrics,
        ms_per_img=round(statistics.median(s["ms"] for s in values), 2),
        peak_alloc_mb=max(r["peak_alloc_mb"] for r in rows),
        peak_reserved_mb=max(r["peak_reserved_mb"] for r in rows),
        date=max(r["date"] for r in rows),
        chunks=[{"date": r["date"], "command": r["command"], "n": r["n"]} for r in rows],
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("chunks", nargs="+", type=Path, help="JSONL files produced by eval_flow.py")
    ap.add_argument("--root", type=Path, default=DATA / "stereo/KITTI/training")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    ids = {p.name[:6] for p in (args.root / "flow_occ").glob("*_10.png")}
    if len(ids) != KITTI_TRAIN_PAIRS:
        ap.error(f"expected 200 KITTI flow pairs, found {len(ids)}")
    rows = [
        json.loads(line)
        for path in args.chunks
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    row = merge(rows, ids)
    with args.out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
