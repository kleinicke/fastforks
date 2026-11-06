"""Per-merge GPU check of the selected families: one model load per (ref, mode) checks its golden output and its speed.

    python tools/merge_check.py estimate SELECTION.json [--family F ...]          # no GPU: jobs, estimated time
    python tools/merge_check.py run SELECTION.json --out ROWS.jsonl [--family F ...] [--prefix CMD] [--force]
    python tools/merge_check.py table SELECTION.json ROWS.jsonl [--family F ...]  # pass / fail per family (markdown)

SELECTION.json is `tools/select_tests.py --json`. `run` starts one worker per (ref, mode) of the selected families' refs
that the speed floor (docs/data/speed.jsonl) has, each under `--prefix` (e.g. a GPU lock with a timeout). The worker is
tools/speed_check.py's own with `golden_step` after its `speed_step` in `MODEL_STEPS`: speed_check times the floor's
inputs with its protocol (so first-call and load times stay comparable with the floor), then, on the same loaded model,
the golden input runs as tests/test_golden.py::test_gpu does (`calls_of`, `gpu_parity`: tolerances, backend, no kernels
fallback) and a row with `input = "golden"` is appended. A ref whose golden needs load options (R-CNN `score_threshold`,
MoGe `normals`) loads a second model for the golden (`golden_reload`). Resumable: jobs whose rows are complete for the
current code (speed_check `code_hash`) are skipped. The speed verdict is speed_check's `check` (slower than the floor by
more than max(5 %, the row's noise band), or peak memory above it by more than 5 % and 8 MB). Exit status of `run` /
`table`: 1 if a family fails.

speed_check.py comes from tools/ or `FASTFORKS_SPEED_CHECK`; its ROOT and FLOOR are set to this checkout.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import io
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = "golden"  # the `input` of a golden row
OVERHEAD_S = 5.0  # per job: interpreter, torch + CUDA init, golden input (hitnet demo: 4-5 s)
PROFILE_S = 2.0  # profiler start-up on the profiled input


def speed_check() -> ModuleType:
    """tools/speed_check.py (or `FASTFORKS_SPEED_CHECK`), pointed at this checkout."""
    if "speed_check" in sys.modules:
        return sys.modules["speed_check"]
    path = Path(os.environ.get("FASTFORKS_SPEED_CHECK") or ROOT / "tools" / "speed_check.py")
    spec = importlib.util.spec_from_file_location("speed_check", path)
    if spec is None or spec.loader is None:
        msg = f"no speed_check at {path}"
        raise ImportError(msg)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["speed_check"] = mod
    spec.loader.exec_module(mod)
    mod.ROOT, mod.FLOOR, mod.ALLOW = ROOT, ROOT / "docs/data/speed.jsonl", ROOT / "docs/data/speed_allowlist.json"
    return mod


def jobs(sel: dict[str, Any], families: list[str], modes: list[str]) -> list[dict[str, Any]]:
    """speed_check jobs (ref, task, mode, floor inputs) of the selected families, plus `family`.

    Refs without floor rows get golden-only jobs (no inputs).
    """
    sc = speed_check()
    out = []
    for fam, v in sorted(sel["families"].items()):
        if families and fam not in families:
            continue
        refs = sorted({r.split("@")[0] for r in v["refs"]})
        got = sc.jobs(refs, modes)
        missing = sorted(set(refs) - {j["ref"] for j in got})
        got += [{"ref": r, "task": v["task"], "mode": m, "inputs": []} for r in missing for m in modes]
        out += [j | {"family": fam} for j in got]
    return out


def estimate_s(job: dict[str, Any], *, profile: bool = False) -> float:
    """Seconds of one job from its floor rows: load, first call and the protocol's calls per input."""
    sc = speed_check()
    floor = sc._floor_index()  # noqa: SLF001 - speed_check's floor index
    calls = 1 + sc.WARMUP + sc.ROUNDS * sc.RUNS + sc.WARMUP + sc.RUNS  # first, warm-up, timed, core_ms
    t = OVERHEAD_S
    for i, inp in enumerate(job["inputs"]):
        f = floor.get((job["ref"], job["mode"], inp["label"]), {})
        t += f.get("load_s", 0.0) if i == 0 else 0.0
        t += f.get("first_call_s", 0.0) + calls * f.get("ms", 0.0) / 1e3
        if profile and i == 0:
            t += PROFILE_S + 3 * f.get("ms", 0.0) / 1e3
    return t


# Worker side (GPU).
def _test_golden() -> ModuleType:
    for d in ("tools", "tests"):  # pytest `pythonpath`: tests/ first
        if str(ROOT / d) not in sys.path:
            sys.path.insert(0, str(ROOT / d))
    return importlib.import_module("test_golden")


def golden_step(model: Any, spec: dict[str, Any], common: dict[str, Any], calls: list[Any], out: Path) -> None:  # noqa: ANN401, ARG001
    """speed_check MODEL_STEPS entry: the golden check of tests/test_golden.py::test_gpu on the loaded model."""
    import gc  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import pytest  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from fastforks._core import kernels  # noqa: PLC0415

    tg = _test_golden()
    ref = model.variant.ref
    row: dict[str, Any] = {**common, "input": GOLDEN, "golden_ref": ref}
    t0 = time.perf_counter()
    m = model
    try:
        want = tg.golden(ref)
        opts = tg.options(ref, want)
        if opts:
            m = tg.API[tg.TASK[ref]].load(ref, mode=spec["mode"], device="cuda", options=opts)
            row["golden_reload"] = True
        kernels.reset_fallbacks()
        got, backend = tg.calls_of(m, ref, want, built=tg.gpu_built(ref, spec["mode"]))
        err = tg.gpu_parity(ref, spec["mode"], want, got, backend)
        row |= {"golden": "pass", "golden_mean": round(float(err.mean()), 5)}
        row["golden_p99"] = round(float(np.percentile(err, 99)), 5)
    except pytest.skip.Exception as e:
        row |= {"golden": "skip", "golden_msg": str(e)[:300]}
    except (Exception, pytest.fail.Exception) as e:  # noqa: BLE001 - recorded as a failed golden; speed still runs
        row |= {"golden": "fail", "error": f"golden: {type(e).__name__}: {e}"[:600]}
    row["golden_s"] = round(time.perf_counter() - t0, 2)
    speed_check()._append(out, row)  # noqa: SLF001 - speed_check's row writer
    del m
    gc.collect()
    torch.cuda.empty_cache()


def worker(spec: dict[str, Any]) -> int:
    from fastforks._core import registry  # noqa: PLC0415

    os.environ.setdefault(registry.NONCOMMERCIAL_ENV, "1")  # as tests/conftest.py: goldens may need such weights
    sc = speed_check()
    sc.MODEL_STEPS.append(golden_step)
    return sc.worker(spec)


# Orchestration (no torch).
def _current(sc: ModuleType, r: dict[str, Any]) -> bool:
    return r.get("code_hash", "")[:16] == sc.files_hash(r.get("files", []))


def _done(sc: ModuleType, rows: list[dict[str, Any]], job: dict[str, Any]) -> bool:
    golden = [r for r in rows if r.get("input") == GOLDEN and (r["ref"], r["mode"]) == (job["ref"], job["mode"])]
    ok = bool(golden) and golden[-1].get("golden") in {"pass", "skip"} and _current(sc, golden[-1])
    return ok and sc._done(rows, "lib", job)  # noqa: SLF001 - speed_check's resume rule


def run(a: argparse.Namespace) -> int:
    sc = speed_check()
    sel = json.loads(a.selection.read_text())
    rows = sc.read_rows(a.out)
    commit = sc._commit()  # noqa: SLF001
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")]))}
    for job in jobs(sel, a.family, a.modes):
        tag = f"{job['family']} {job['ref']} {job['mode']}"
        if not a.force and _done(sc, rows, job):
            print(f"== {tag}: done on this code", flush=True)
            continue
        spec = job | {"side": "lib", "out": str(a.out), "commit": commit, "forks": "", "fork_key": ""}
        spec["no_profile"] = not a.profile
        print(f"== {tag} {time.strftime('%T')} (estimate {estimate_s(job, profile=a.profile):.0f} s)", flush=True)
        before = len(rows)
        cmd = [*shlex.split(a.prefix), sys.executable, str(Path(__file__).resolve()), "worker", json.dumps(spec)]
        rc = subprocess.run(cmd, env=env, check=False).returncode
        rows = sc.read_rows(a.out)
        if rc != 0 and len(rows) == before:
            err = {k: job[k] for k in ("ref", "task", "mode")} | {"side": "lib", "input": "-", "commit": commit}
            sc._append(a.out, err | {"error": f"worker rc={rc}", "date": time.strftime("%F")})  # noqa: SLF001
            rows = sc.read_rows(a.out)
    return table(sel, a.out, a.family, a.modes)


def _regressions(sc: ModuleType, path: Path) -> dict[tuple[str, str, str], str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sc.check(path)
    out = {}
    for line in buf.getvalue().splitlines():
        if line.startswith("REGRESSION "):
            _, ref, mode, inp, msg = line.split(" ", 4)
            out[ref, mode, inp] = msg
    return out


def _job_line(
    fam: str,
    j: dict[str, Any],
    latest: dict[tuple[str, str, str], dict[str, Any]],
    reg: dict[Any, str],
    n: dict[str, int],
) -> str:
    """One markdown row of a job; counts its golden and input outcomes into `n`."""
    floor = speed_check()._floor_index()  # noqa: SLF001
    g = latest.get((j["ref"], j["mode"], GOLDEN), {})
    if latest.get((j["ref"], j["mode"], "-"), {}).get("_i", -1) > g.get("_i", -1):  # load / worker error is the last
        n["error"] += 1
        g = {"golden": "missing", "error": latest[j["ref"], j["mode"], "-"]["error"]}
    gold = g.get("golden", "missing")
    n[gold] += 1
    cells = []
    for inp in j["inputs"]:
        key = (j["ref"], j["mode"], inp["label"])
        r, f = latest.get(key, {}), floor[key]
        if "ms" not in r:
            n["untimed"] += 1
            cells.append(f"{inp['label']} {r.get('error', 'not run')[:60]}")
            continue
        n["timed"] += 1
        n["slow"] += key in reg
        cells.append(
            f"{inp['label']} {r['ms']:.2f} / {f['ms']:.2f} ms ({r['ms'] / f['ms'] - 1:+.1%}), "
            f"{r['peak_alloc_mb']}/{r['peak_reserved_mb']} MB" + (" **slow**" if key in reg else "")
        )
    if gold == "pass":
        gold = f"pass ({g['golden_mean']:.4f} / {g['golden_p99']:.4f})"
    elif "error" in g or "golden_msg" in g:
        gold += f": {g.get('error', g.get('golden_msg', ''))[:80]}"
    first = latest.get((j["ref"], j["mode"], j["inputs"][0]["label"]), {}) if j["inputs"] else {}
    return f"| {fam} | {j['ref']} | {j['mode']} | {gold} | {first.get('load_s', '-')} | {'; '.join(cells)} |"


def table(sel: dict[str, Any], path: Path, families: list[str], modes: list[str]) -> int:
    """Markdown: one line per family (verdict), then one per job (golden error, ms vs floor, peak MB)."""
    sc = speed_check()
    reg = _regressions(sc, path)
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for i, r in enumerate(sc.read_rows(path)):
        if r.get("side") == "lib":
            latest[r["ref"], r["mode"], r.get("input", "-")] = r | {"_i": i}
    by_family: dict[str, list[dict[str, Any]]] = {}
    for j in jobs(sel, families, modes):
        by_family.setdefault(j["family"], []).append(j)
    fam_lines, job_lines, bad = [], [], 0
    for fam, js in by_family.items():
        n = dict.fromkeys(("pass", "fail", "skip", "missing", "slow", "error", "timed", "untimed"), 0)
        job_lines += [_job_line(fam, j, latest, reg, n) for j in js]
        ok = n["fail"] == n["missing"] == n["slow"] == n["error"] == n["untimed"] == 0
        bad += not ok
        fam_lines.append(
            f"| {fam} | {len(js)} | {n['pass']} / {n['fail']} / {n['skip']} / {n['missing']} | "
            f"{n['timed']} / {n['slow']} / {n['untimed']} | {'PASS' if ok else 'FAIL'} |"
        )
    print("| family | jobs | golden pass / fail / skip / missing | inputs timed / slow / not timed | verdict |")
    print("|---|---|---|---|---|")
    print("\n".join(fam_lines))
    print("\n| family | ref | mode | golden (mean / p99 error) | load s | input: ms / floor ms, peak MB alloc/res |")
    print("|---|---|---|---|---|---|")
    print("\n".join(job_lines))
    for key, msg in sorted(reg.items()):
        print(f"\nREGRESSION {' '.join(key)}: {msg}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("estimate", "run", "table"):
        p = sub.add_parser(name)
        p.add_argument("selection", type=Path)
        if name == "table":
            p.add_argument("out", type=Path)
        p.add_argument("--family", nargs="*", default=[])
        p.add_argument("--modes", nargs="*", default=["portable", "kernels"])
        p.add_argument("--profile", action="store_true", help="also speed_check's profile of one input")
    p = sub.choices["run"]
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prefix", default="", help="command prefix of each worker, e.g. a GPU lock with a timeout")
    p.add_argument("--force", action="store_true", help="run again even if the code did not change")
    sub.add_parser("worker").add_argument("spec")
    a = ap.parse_args()
    if a.cmd == "worker":
        return worker(json.loads(a.spec))
    sel = json.loads(a.selection.read_text())
    if a.cmd == "run":
        return run(a)
    if a.cmd == "table":
        return table(sel, a.out, a.family, a.modes)
    js = jobs(sel, a.family, a.modes)
    per: dict[str, float] = {}
    for j in js:
        per[j["family"]] = per.get(j["family"], 0.0) + estimate_s(j, profile=a.profile)
    for fam, t in per.items():
        print(f"{fam}: {sum(j['family'] == fam for j in js)} jobs, ~{t / 60:.1f} min")
    print(f"total: {len(js)} jobs, ~{sum(per.values()) / 60:.0f} min GPU (one lock per job)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
