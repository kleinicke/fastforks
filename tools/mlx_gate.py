"""The pre-commit gate for an MLX port: goldens, the as-shipped A/B and the generated-file checks, as one table.

    python tools/mlx_gate.py REF [REF ...] [--lock /Volumes/Daten/ff-mlx/gpu2.sh] [--skip ab] [--dtypes float16]

For the refs a change touched, runs (each step a subprocess, in this order):

- `goldens`: the golden tests of each ref (`tests/test_golden.py::test_mlx[REF@N]`, `--golden-tests` for others), pass
  when none fails (a skip, such as a missing golden, is listed but not a failure);
- `ab`: `tools/mlx_core_ab.py run --only REF ...` into a scratch file (all three arms: hand, graph, shipped), pass when
  the row has no flag (`slower`, `heavier`, `worse`, `shipped-slower`) or only a `heavier` that is an accepted trade;
- `gen_auto_mode`, `gen_docs`: `--check` (the generated files up to date).

Prints one pass / fail row per (step, ref, dtype) and exits 1 when any fails. The A/B rows go to `--ab-out` (default: a
temp file), never into `docs/plans/mlx-core-ab.jsonl`: copy them there yourself once they are the ones to keep.

`--lock`: the GPU lock prefix for the two GPU steps. With `gpu2.sh` the goldens run under `--light` and the A/B under
`--timed`; any other script (`gpu.sh`) wraps both as it is. Run the gate itself outside the lock.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
STEPS = ("goldens", "ab", "gen_auto_mode", "gen_docs")


@dataclass
class Check:
    """One gate row."""

    step: str
    ref: str
    ok: bool
    detail: str = ""
    dtype: str = ""


def lock_prefix(lock: str | None, kind: str) -> list[str]:
    """The command prefix for a GPU step (`kind`: `light` for goldens, `timed` for the A/B)."""
    if not lock:
        return []
    return [lock, f"--{kind}"] if Path(lock).name == "gpu2.sh" else [lock]


def golden_ids(collected: str, refs: list[str], tests: list[str]) -> dict[str, list[str]]:
    """Per ref, the node ids in `pytest --collect-only -q` output of the tests `tests` parametrized by `REF@N`."""
    out: dict[str, list[str]] = {r: [] for r in refs}
    pat = re.compile(r"::(\w+)\[(.+?)@\d+[\]-]")
    for line in collected.splitlines():
        m = pat.search(line.strip())
        if m and m.group(1) in tests and m.group(2) in out:
            out[m.group(2)].append(line.strip())
    return out


def junit_results(xml: str) -> dict[str, str]:
    """Node id (`file::name`) -> `passed` / `failed` / `skipped` from a pytest junit XML report."""
    out = {}
    for case in ET.fromstring(xml).iter("testcase"):  # noqa: S314 - pytest's own report
        cls = case.get("classname", "").replace(".", "/") + ".py"
        name = f"{cls}::{case.get('name')}"
        kids = {c.tag for c in case}
        out[name] = "failed" if kids & {"failure", "error"} else "skipped" if "skipped" in kids else "passed"
    return out


def golden_checks(ids: dict[str, list[str]], results: dict[str, str]) -> list[Check]:
    """One row per ref from the junit `results` of its golden node ids."""
    rows = []
    for ref, nodes in ids.items():
        if not nodes:
            rows.append(Check("goldens", ref, ok=False, detail="no golden test collected"))
            continue
        got = [results.get(n, "missing") for n in nodes]
        bad = [n.split("::")[-1] for n, s in zip(nodes, got, strict=True) if s in ("failed", "missing")]
        skipped = got.count("skipped")
        detail = f"{got.count('passed')} passed" + (f", {skipped} skipped" if skipped else "")
        rows.append(Check("goldens", ref, ok=not bad, detail=detail + (f"; failed: {', '.join(bad)}" if bad else "")))
    return rows


def ab_checks(rows: list[dict[str, Any]], refs: list[str], dtypes: list[str]) -> list[Check]:
    """One row per (ref, dtype) from the A/B rows: pass without a flag, or with only an accepted `heavier`."""
    import mlx_core_ab as ab  # noqa: PLC0415 - tools/

    peaks = ab.e2e_peaks()
    out = []
    for ref in refs:
        for dt in dtypes:
            got = [r for r in rows if r.get("ref") == ref and r.get("dtype") == dt]
            if not got:
                out.append(Check("ab", ref, ok=False, detail="no row measured", dtype=dt))
                continue
            r = got[-1]
            if "net_ms" not in r.get("hand", {}) or "net_ms" not in r.get("graph", {}):
                why = r.get("hand", {}).get("error") or r.get("graph", {}).get("error") or "an arm failed"
                out.append(Check("ab", ref, ok=False, detail=str(why)[:80], dtype=dt))
                continue
            flags = ab._flags(r)  # noqa: SLF001 - the A/B's own rules
            acc = "heavier" in flags and ab.accepted(r, peaks)
            left = [f for f in flags if not (f == "heavier" and acc)]
            h, g = r["hand"]["net_ms"], r["graph"]["net_ms"]
            ship = r.get("shipped", {}).get("net_ms")
            detail = f"hand/graph {h / g:.2f}" + (f", shipped/graph {ship / g:.2f}" if ship else "")
            detail += f"; flags: {', '.join(flags)}" + (" (heavier accepted)" if acc else "") if flags else ""
            out.append(Check("ab", ref, ok=not left, detail=detail, dtype=dt))
    return out


def table(rows: list[Check]) -> str:
    """The gate rows as a fixed-width table."""
    w = max([len(r.ref) for r in rows] + [3])
    lines = [f"{'step':<14} {'ref':<{w}} {'dtype':<8} {'':<4}  detail"]
    lines += [f"{r.step:<14} {r.ref:<{w}} {r.dtype:<8} {'PASS' if r.ok else 'FAIL':<4}  {r.detail}" for r in rows]
    return "\n".join(lines)


def _run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:  # noqa: ANN401
    print("$ " + " ".join(cmd), file=sys.stderr, flush=True)
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False, **kw)


def _goldens(a: argparse.Namespace, tmp: Path) -> list[Check]:
    col = _run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/test_golden.py"])
    ids = golden_ids(col.stdout, a.refs, a.golden_tests)
    nodes = [n for v in ids.values() for n in v]
    results: dict[str, str] = {}
    if nodes:
        xml = tmp / "goldens.xml"
        env = os.environ | {"FF_TEST_DEFAULT_CACHES": os.environ.get("FF_TEST_DEFAULT_CACHES", "1")}
        cmd = [*lock_prefix(a.lock, "light"), sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
        p = _run([*cmd, f"--junitxml={xml}", *nodes], env=env)
        if xml.exists():
            results = junit_results(xml.read_text())
        else:
            print(p.stdout[-2000:] + p.stderr[-2000:], file=sys.stderr)
    return golden_checks(ids, results)


def _ab(a: argparse.Namespace, tmp: Path) -> list[Check]:
    out = Path(a.ab_out) if a.ab_out else tmp / "ab.jsonl"
    cmd = [*lock_prefix(a.lock, "timed"), sys.executable, str(TOOLS / "mlx_core_ab.py"), "run", "--only", *a.refs]
    cmd += ["--dtypes", *a.dtypes, "--modes", *a.modes, "--out", str(out), "--tmp", str(tmp), "--force"]
    p = _run(cmd)
    if p.returncode:
        print(p.stdout[-2000:] + p.stderr[-2000:], file=sys.stderr)
    rows = [json.loads(s) for s in out.read_text().splitlines() if s.strip()] if out.exists() else []
    return ab_checks(rows, a.refs, a.dtypes)


def _gen(step: str) -> Check:
    p = _run([sys.executable, str(TOOLS / f"{step}.py"), "--check"])
    tail = (p.stdout + p.stderr).strip().splitlines()
    return Check(
        step, "-", ok=p.returncode == 0, detail="up to date" if not p.returncode else " ".join(tail[-2:])[:120]
    )


def main(argv: list[str] | None = None) -> int:
    """Command line (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("refs", nargs="+")
    ap.add_argument("--lock", default=None, help="GPU lock script for the GPU steps (gpu.sh / gpu2.sh)")
    ap.add_argument("--skip", nargs="*", default=[], choices=STEPS)
    ap.add_argument("--dtypes", nargs="*", default=["float16", "float32"])
    ap.add_argument("--modes", nargs="*", default=["portable"])
    ap.add_argument("--golden-tests", nargs="*", default=["test_mlx"], help="tests/test_golden.py tests to run")
    ap.add_argument("--ab-out", default=None, help="keep the A/B rows here (default: a temp file)")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(TOOLS))
    rows: list[Check] = []
    with tempfile.TemporaryDirectory(prefix="mlx-gate-") as d:
        tmp = Path(d)
        if "goldens" not in a.skip:
            rows += _goldens(a, tmp)
        if "ab" not in a.skip:
            rows += _ab(a, tmp)
    rows += [_gen(s) for s in ("gen_auto_mode", "gen_docs") if s not in a.skip]
    print(table(rows))
    bad = sum(not r.ok for r in rows)
    print(f"gate: {'PASS' if not bad else f'FAIL ({bad} of {len(rows)})'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
