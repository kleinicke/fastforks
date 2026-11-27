"""`tools/mlx_gate.py`'s parsing and pass / fail rules, on made-up output (no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import mlx_gate as gate

COLLECTED = """tests/test_golden.py::test_cpu_portable[crestereo-eth3d@1]
tests/test_golden.py::test_gpu[crestereo-eth3d@1-portable]
tests/test_golden.py::test_mlx[crestereo-eth3d@1]
tests/test_golden.py::test_mlx[crestereo-eth3d-x@1]
tests/test_golden.py::test_mlx[rife-4.25@2]

12 tests collected in 0.5s
"""

JUNIT = """<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">
<testcase classname="tests.test_golden" name="test_mlx[crestereo-eth3d@1]" time="1.0"/>
<testcase classname="tests.test_golden" name="test_gpu[crestereo-eth3d@1-portable]" time="1.0">
<skipped message="no CUDA"/></testcase>
<testcase classname="tests.test_golden" name="test_mlx[rife-4.25@2]" time="1.0"><failure message="x"/></testcase>
</testsuite></testsuites>"""


def test_lock_prefix() -> None:
    assert gate.lock_prefix(None, "light") == []
    assert gate.lock_prefix("/x/gpu.sh", "light") == ["/x/gpu.sh"]
    assert gate.lock_prefix("/x/gpu2.sh", "timed") == ["/x/gpu2.sh", "--timed"]


def test_golden_ids_and_results() -> None:
    ids = gate.golden_ids(COLLECTED, ["crestereo-eth3d", "rife-4.25", "none"], ["test_mlx", "test_gpu"])
    assert ids["crestereo-eth3d"] == [
        "tests/test_golden.py::test_gpu[crestereo-eth3d@1-portable]",
        "tests/test_golden.py::test_mlx[crestereo-eth3d@1]",
    ]
    assert ids["rife-4.25"] == ["tests/test_golden.py::test_mlx[rife-4.25@2]"]
    res = gate.junit_results(JUNIT)
    assert res["tests/test_golden.py::test_mlx[crestereo-eth3d@1]"] == "passed"
    assert res["tests/test_golden.py::test_gpu[crestereo-eth3d@1-portable]"] == "skipped"
    rows = {r.ref: r for r in gate.golden_checks(ids, res)}
    assert rows["crestereo-eth3d"].ok
    assert rows["crestereo-eth3d"].detail == "1 passed, 1 skipped"
    assert not rows["rife-4.25"].ok
    assert not rows["none"].ok


def _row(ref: str, dt: str, hand_ms: float, *, transient: float = 10.0, **kw: Any) -> dict:
    def arm(ms: float, tr: float) -> dict:
        return {"net_ms": ms, "net_peak_mb": 100 + tr, "resident_mb": 100.0, "net_transient_mb": tr, "net_calls": 1}

    return (
        {"ref": ref, "family": "f", "mode": "portable", "dtype": dt, "hand": arm(hand_ms, transient)}
        | {"graph": arm(10.0, 10.0)}
        | kw
    )


def test_ab_checks() -> None:
    rows = [
        _row("fast", "float16", 8.0),
        _row("slow", "float16", 12.0),
        _row("heavy", "float16", 10.0, transient=200.0),
        {"ref": "broken", "dtype": "float16", "hand": {"error": "boom"}, "graph": {}},
    ]
    got = {r.ref: r for r in gate.ab_checks(rows, ["fast", "slow", "heavy", "broken", "missing"], ["float16"])}
    assert got["fast"].ok
    assert got["fast"].detail == "hand/graph 0.80"
    assert not got["slow"].ok
    assert "slower" in got["slow"].detail
    assert not got["heavy"].ok
    assert "heavier" in got["heavy"].detail
    assert not got["broken"].ok
    assert got["broken"].detail == "boom"
    assert not got["missing"].ok
    text = gate.table(list(got.values()))
    assert "PASS" in text
    assert "FAIL" in text
