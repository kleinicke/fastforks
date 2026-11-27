"""`tools/mlx_core_ab.py`'s statistics and flags, on made-up rows (no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import mlx_core_ab as ab


def _arm(ms: float, *, transient: float = 10.0, resident: float = 100.0, reps: list[list[float]] | None = None) -> dict:
    out: dict[str, Any] = {"net_ms": ms, "net_peak_mb": resident + transient, "resident_mb": resident, "net_calls": 1}
    out["net_transient_mb"] = transient
    if reps is not None:
        out["net_ms_reps"] = reps
    return out


def _row(hand: dict, graph: dict, **kw: Any) -> dict:
    return {"ref": "r", "family": "f", "mode": "portable", "dtype": "float16", "hand": hand, "graph": graph} | kw


def test_ci_separates_and_overlaps() -> None:
    slow = [[1.10, 1.11, 1.09, 1.10, 1.12], [1.10, 1.09, 1.11, 1.10, 1.10]]
    fast = [[1.00, 1.01, 0.99, 1.00, 1.00], [1.00, 1.01, 1.00, 0.99, 1.00]]
    lo, hi = ab._ci(slow, fast)
    assert 1.05 < lo <= 1.10 <= hi < 1.15
    assert ab._ci(slow, fast) == [lo, hi]  # seeded
    noisy = [[1.0, 1.3, 0.9, 1.2, 1.0], [0.9, 1.1, 1.4, 1.0, 1.0]]
    lo, hi = ab._ci(noisy, fast)
    assert lo < ab.SLOWER < hi
    assert ab._ci([], fast) == []


def test_slower_needs_the_interval() -> None:
    noisy = _row(_arm(1.05), _arm(1.0), ci={"net_ms": [0.97, 1.12]})
    assert "slower" not in ab._flags(noisy)
    clear = _row(_arm(1.05), _arm(1.0), ci={"net_ms": [1.03, 1.07]})
    assert "slower" in ab._flags(clear)
    old = _row(_arm(1.05), _arm(1.0))  # no interval: the point ratio
    assert "slower" in ab._flags(old)


def test_heavier_on_the_transient_peak() -> None:
    # more resident (weights) but the same transient: not heavier
    assert "heavier" not in ab._flags(_row(_arm(1, transient=10, resident=300), _arm(1, transient=10)))
    assert "heavier" in ab._flags(_row(_arm(1, transient=50), _arm(1, transient=10)))
    # above the ratio but within the absolute floor
    assert "heavier" not in ab._flags(_row(_arm(1, transient=3), _arm(1, transient=1)))
    # rows measured before the field: net peak less resident
    h = {"net_ms": 1, "net_peak_mb": 160, "resident_mb": 100}
    g = {"net_ms": 1, "net_peak_mb": 120, "resident_mb": 100}
    assert ab._transient(h) == 60
    assert "heavier" in ab._flags(_row(h, g))


def test_shipped_slower() -> None:
    row = _row(_arm(1.0), _arm(1.0), shipped=_arm(1.3), ci={"net_ms": [0.98, 1.01], "shipped": [1.2, 1.4]})
    assert ab._flags(row) == ["shipped-slower"]
    row["ci"]["shipped"] = [0.99, 1.04]
    assert ab._flags(row) == []


def test_pool() -> None:
    a = {"net_ms_runs": [1.0, 1.2], "ms_runs": [5.0, 5.0], "net_peak_mb": 10, "net_transient_mb": 2, "load_s": 1.0}
    b = {"net_ms_runs": [1.1, 1.3], "ms_runs": [6.0, 6.0], "net_peak_mb": 12, "net_transient_mb": 1, "load_s": 3.0}
    for r in (a, b):
        r |= {"net_ms": 0, "first_s": 1, "net_first_s": 1, "net_copy_ms": 0, "resident_mb": 5, "peak_footprint_mb": 9}
    p = ab._pool([a, b | {"quality": {"mean": 1}}, {"error": "boom"}])
    assert p["net_ms"] == 1.15
    assert p["net_ms_reps"] == [[1.0, 1.2], [1.1, 1.3]]
    assert p["ms"] == 5.5
    assert (p["net_peak_mb"], p["net_transient_mb"], p["load_s"]) == (12, 2, 2.0)
    assert p["quality"] == {"mean": 1}
    assert p["repeats"] == 2
    assert p["failed_repeats"] == ["boom"]
    assert "net_ms_runs" not in p
    assert ab._pool([{"error": "x"}]) == {"error": "x"}


def test_loops() -> None:
    assert ab._loops(50.0) == 1
    assert ab._loops(5.0) == 4
    assert ab._loops(0.01) == ab.LOOP_MAX


def test_md_sections() -> None:
    f16 = _row(_arm(1.0, resident=90), _arm(1.0, resident=50), ci={"net_ms": [0.9, 1.0]})
    f32 = _row(_arm(1.0, resident=100), _arm(1.0, resident=100)) | {"dtype": "float32"}
    text = ab.md([f16, f32], {"mlx": {}, "mps": {}})
    assert "## Resident: float16 against float32" in text
    assert "| r | 90 / 100 | 0.90 |" in text
    assert "0 rows flagged" in text
    assert "0.90-1.00" in text
