"""Per-merge GPU check of the selected families: one model load per (ref, mode) checks its golden output and its speed.

    python tools/merge_check.py estimate SELECTION.json [--family F ...]          # no GPU: jobs, estimated time
    python tools/merge_check.py run SELECTION.json --out ROWS.jsonl [--family F ...] [--prefix CMD] [--force]
    python tools/merge_check.py table SELECTION.json ROWS.jsonl [--family F ...]  # pass / fail per family (markdown)

Add `--golden-only` to the three commands for a correctness check of the exact selected revisions. Passing
checks are reused across edits limited to unrelated model families.
For correctness-only runs, `--batch-size 4` shares interpreter and CUDA startup across up to four jobs.
If a group fails, its unfinished jobs retry in fresh processes.
Add `--no-core` to `estimate` and `run` when the full-call timing is enough for the speed floor.

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
import hashlib
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
GENERATED_DOCS = ("docs/data", "docs/models", "docs/benchmarks.md", "docs/overview.md")  # legacy receipt scope
GOLDEN_DOCS = ("docs", "README.md", "CONTRIBUTING.md")
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


def jobs(
    sel: dict[str, Any], families: list[str], modes: list[str], *, golden_only: bool = False
) -> list[dict[str, Any]]:
    """speed_check jobs (ref, task, mode, floor inputs) of the selected families, plus `family`.

    Refs without floor rows get golden-only jobs (no inputs). A golden-only selection uses the exact revisions
    from the selector, including when multiple revisions share a benchmark row.
    """
    if golden_only:
        return [
            {"ref": ref, "task": v["task"], "mode": mode, "inputs": [], "family": fam}
            for fam, v in sorted(sel["families"].items())
            if not families or fam in families
            for ref in sorted(set(v["refs"]))
            for mode in modes
        ]
    timed = speed_check().jobs(modes=modes)
    timed_refs = {j["ref"] for j in timed}
    out = []
    for fam, v in sorted(sel["families"].items()):
        if families and fam not in families:
            continue
        refs = {ref: ref.partition("@")[0] for ref in sorted(set(v["refs"]))}
        names = {ref: model_id if model_id in timed_refs else ref for ref, model_id in refs.items()}
        selected = set(names.values())
        got = [j for j in timed if j["ref"] in selected]
        measured = {(j["ref"], j["mode"]) for j in got}
        got += [
            {"ref": ref, "task": v["task"], "mode": mode, "inputs": []}
            for ref, name in names.items()
            for mode in modes
            if (name, mode) not in measured
        ]
        out += [j | {"family": fam} for j in got]
    return out


def estimate_s(
    job: dict[str, Any],
    floor: dict[tuple[str, str, str], dict[str, Any]],
    *,
    profile: bool = False,
    no_core: bool = False,
) -> float:
    """Seconds of one job from its floor rows: load, first call and the protocol's calls per input."""
    if not job["inputs"]:
        return OVERHEAD_S
    sc = speed_check()
    calls = sc.WARMUP + sc.ROUNDS * sc.RUNS
    if not no_core:
        calls += 1 + sc.WARMUP + sc.RUNS  # capture once, then warm up and time the runner
    t = OVERHEAD_S
    for i, inp in enumerate(job["inputs"]):
        f = floor.get((job["ref"], job["mode"], inp["label"]), {})
        t += f.get("load_s", 0.0) if i == 0 else 0.0
        seconds = f.get("ms", 0.0) / 1e3
        t += f.get("first_call_s", seconds) + calls * seconds
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
    if "golden_source" in spec:
        row["golden_source"] = spec["golden_source"]
    t0 = time.perf_counter()
    m = model
    try:
        want = tg.golden(ref)
        opts = tg.options(ref, want)
        if opts:
            m = tg.API[tg.TASK[ref]].load(
                ref, mode=spec["mode"], device=spec.get("device", "cuda"), dtype=spec.get("dtype"), **(opts or {})
            )
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def worker(spec: dict[str, Any]) -> int:
    from fastforks._core import registry  # noqa: PLC0415

    os.environ.setdefault(registry.NONCOMMERCIAL_ENV, "1")  # as tests/conftest.py: goldens may need such weights
    sc = speed_check()
    if golden_step not in sc.MODEL_STEPS:
        sc.MODEL_STEPS.append(golden_step)
    return sc.worker(spec)


def worker_batch(specs: list[dict[str, Any]]) -> int:
    """Run correctness checks in one process, stopping before a CUDA error can affect later jobs."""
    import gc  # noqa: PLC0415

    import torch  # noqa: PLC0415

    for i, spec in enumerate(specs):
        path = Path(spec["out"])
        offset = path.stat().st_size if path.exists() else 0
        if status := worker(spec | {"warm_worker": i > 0}):
            return status
        with path.open("rb") as stream:
            stream.seek(offset)
            rows = [json.loads(line) for line in stream if line.strip()]
        if not rows or rows[-1].get("golden") not in {"pass", "skip"}:
            return 1
        gc.collect()
        torch.cuda.empty_cache()
    return 0


# Orchestration (no torch).
def _current(sc: ModuleType, r: dict[str, Any]) -> bool:
    return r.get("code_hash", "")[:16] == sc.files_hash(r.get("files", []))


def _golden_sources(commit: str) -> tuple[str, ...]:
    """Fingerprint the tested tree and accept the previous format for existing results."""
    git = ["git", "-C", str(ROOT)]
    paths = [".", *(f":(exclude){path}" for path in GENERATED_DOCS)]
    index = subprocess.check_output([*git, "ls-files", "--stage", "-z", "--", *paths])
    changes = subprocess.check_output([*git, "diff", "--binary", "HEAD", "--", *paths])
    tree = hashlib.sha256(index + b"\0" + changes)
    untracked = subprocess.check_output(
        [*git, "ls-files", "--others", "--exclude-standard", "-z", "--", "src", "tests", "tools"]
    )
    if untracked:
        tree.update(b"\0untracked\0" + untracked)
        for name in filter(None, untracked.split(b"\0")):
            path = ROOT / os.fsdecode(name)
            if path.is_symlink():
                tree.update(b"link" + hashlib.sha256(os.fsencode(path.readlink())).digest())
            else:
                with path.open("rb") as stream:
                    tree.update(b"file" + hashlib.file_digest(stream, "sha256").digest())
        return ("tree:" + tree.hexdigest(),)
    full_diff = subprocess.check_output([*git, "diff", "--binary", "HEAD"])
    return (
        "tree:" + tree.hexdigest(),
        f"{commit}:{hashlib.sha256(full_diff).hexdigest()}",
    )


def _model_directory(name: str) -> str:
    match name.split("/", 5):
        case ["src", "fastforks", task, "models", family, _]:
            return f"src/fastforks/{task}/models/{family}"
        case _:
            return ""


def _golden_file_digests(
    roots: set[str],
) -> tuple[bytes, dict[str, bytes], dict[str, dict[str, object]]]:
    """Fingerprint common source, each model directory and the separate fixture records."""
    from select_tests import (  # noqa: PLC0415 - only for golden-only orchestration
        DEPENDENCIES,
        configuration_without_local_version,
        split_golden_manifest,
    )

    git = ["git", "-C", str(ROOT), "ls-files", "-z"]
    paths = [".", *(f":(exclude){path}" for path in GOLDEN_DOCS)]
    tracked = subprocess.check_output([*git, "--cached", "--", *paths])
    untracked = subprocess.check_output([*git, "--others", "--exclude-standard", "--", "src", "tests", "tools"])
    common = hashlib.sha256(b"goldens-family-v2\0")
    per_directory = {directory: hashlib.sha256() for directory in roots}
    fixture_records: dict[str, dict[str, object]] = {}
    for name in sorted(set(filter(None, tracked.split(b"\0") + untracked.split(b"\0")))):
        relative = os.fsdecode(name)
        path = ROOT / relative
        directory = _model_directory(relative)
        scope = per_directory.get(directory, common)
        manifest = directory in roots and relative == f"{directory}/golden.toml"
        scope.update(name + b"\0")
        if path.is_symlink():
            scope.update(b"link\0" + os.fsencode(path.readlink()) + b"\0")
        if path.is_file():
            with path.open("rb") as stream:
                if relative in DEPENDENCIES or manifest:
                    raw = stream.read().decode("utf-8", "surrogateescape")
                    if manifest:
                        sections = split_golden_manifest(raw)
                        if sections is not None:
                            settings, fixture_records[directory] = sections
                            keys = sorted(fixture_records[directory])
                            scope.update(
                                b"golden-settings\0" + hashlib.sha256(repr((settings, keys)).encode()).digest()
                            )
                            continue
                    else:
                        config = configuration_without_local_version(relative, raw)
                        if config is not None:
                            scope.update(b"configuration\0" + hashlib.sha256(repr(config).encode()).digest())
                            continue
                    stream.seek(0)
                scope.update(b"file\0" + hashlib.file_digest(stream, "sha256").digest())
        else:
            scope.update(b"missing\0")

    return common.digest(), {directory: scope.digest() for directory, scope in per_directory.items()}, fixture_records


def _golden_sources_by_ref(js: list[dict[str, Any]], commit: str) -> dict[str, tuple[str, ...]]:
    """Keep passing results reusable across unrelated model and fixture edits."""
    from select_tests import Graph, families, family_closures  # noqa: PLC0415 - only for golden-only orchestration

    family_info = families(ROOT)
    roots = {f["dir"] for f in family_info.values()}
    reachable = family_closures(Graph(ROOT), family_info)
    common, per_directory, fixture_records = _golden_file_digests(roots)
    legacy = _golden_sources(commit)
    families_digests = {}
    for family in {j["family"] for j in js} & reachable.keys():
        dependencies = {_model_directory(path) for path in reachable[family]} & roots
        dependencies.add(family_info[family]["dir"])
        digest = hashlib.sha256(common)
        for directory in sorted(dependencies):
            digest.update(os.fsencode(directory) + b"\0" + per_directory[directory])
        families_digests[family] = digest

    sources: dict[str, tuple[str, ...]] = {}
    for job in js:
        family, ref = job["family"], job["ref"]
        if family not in families_digests:
            sources[ref] = legacy
            continue
        digest = families_digests[family].copy()
        records = fixture_records.get(family_info[family]["dir"], {})
        if ref in records:
            digest.update(b"fixture\0" + hashlib.sha256(repr(records[ref]).encode()).digest())
        sources[ref] = ("family:" + digest.hexdigest(), *legacy)
    return sources


def _done(
    sc: ModuleType,
    rows: list[dict[str, Any]],
    job: dict[str, Any],
    *,
    golden_sources: tuple[str, ...] = (),
    no_core: bool = False,
) -> bool:
    last = next(
        (
            r
            for r in reversed(rows)
            if r.get("side") == "lib"
            and r.get("input") in {GOLDEN, "-"}
            and (r["ref"], r["mode"]) == (job["ref"], job["mode"])
        ),
        {},
    )
    if golden_sources:
        return last.get("golden") == "pass" and last.get("golden_source") in golden_sources and _current(sc, last)
    ok = last.get("golden") in {"pass", "skip"} and _current(sc, last)
    timing_job = job | {"no_core": no_core}
    return ok and (not job["inputs"] or sc._done(rows, "lib", timing_job))  # noqa: SLF001 - speed_check's resume rule


def _launch(specs: tuple[dict[str, Any], ...], prefix: list[str], env: dict[str, str]) -> int:
    payload = specs[0] if len(specs) == 1 else specs
    cmd = [*prefix, sys.executable, str(Path(__file__).resolve()), "worker", json.dumps(payload)]
    return subprocess.run(cmd, env=env, check=False).returncode


def _isolated_worker(sc: ModuleType, spec: dict[str, Any], prefix: list[str], env: dict[str, str]) -> None:
    if rc := _launch((spec,), prefix, env):
        err = {k: spec[k] for k in ("ref", "task", "mode", "side", "commit")} | {"input": "-"}
        sc._append(Path(spec["out"]), err | {"error": f"worker rc={rc}", "date": time.strftime("%F")})  # noqa: SLF001


def run(a: argparse.Namespace) -> int:
    sc = speed_check()
    sel = json.loads(a.selection.read_text(encoding="utf-8"))
    rows = sc.read_rows(a.out)
    commit = sc._commit()  # noqa: SLF001
    checks = jobs(sel, a.family, a.modes, golden_only=a.golden_only)
    floor = sc._floor_index() if not a.golden_only else {}  # noqa: SLF001
    sources = _golden_sources_by_ref(checks, commit) if a.golden_only else {}
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")]))}
    prefix = shlex.split(a.prefix)
    pending = []
    for job in checks:
        golden_sources = sources.get(job["ref"], ())
        tag = f"{job['family']} {job['ref']} {job['mode']}"
        if not a.force and _done(sc, rows, job, golden_sources=golden_sources, no_core=a.no_core):
            print(f"== {tag}: done on this code", flush=True)
            continue
        spec = job | {"side": "lib", "out": str(a.out), "commit": commit, "forks": "", "fork_key": ""}
        if golden_sources:
            spec["golden_source"] = golden_sources[0]
        spec["no_profile"] = not a.profile
        spec["no_core"] = a.no_core
        pending.append(spec)
    for start in range(0, len(pending), a.batch_size):
        group = tuple(pending[start : start + a.batch_size])
        for spec in group:
            duration = estimate_s(spec, floor, profile=a.profile, no_core=a.no_core)
            tag = f"{spec['family']} {spec['ref']} {spec['mode']}"
            print(f"== {tag} {time.strftime('%T')} (estimate {duration:.0f} s)", flush=True)
        if len(group) == 1:
            _isolated_worker(sc, group[0], prefix, env)
            continue
        offset = len(sc.read_rows(a.out))
        if _launch(group, prefix, env):
            completed = {
                (r["ref"], r["mode"])
                for r in sc.read_rows(a.out)[offset:]
                if r.get("input") == GOLDEN and r.get("golden") in {"pass", "skip"}
            }
            for spec in group:
                if (spec["ref"], spec["mode"]) not in completed:
                    print(f"== {spec['ref']} {spec['mode']}: retrying in a fresh process", flush=True)
                    _isolated_worker(sc, spec, prefix, env)
    return table(sel, a.out, a.family, a.modes, golden_only=a.golden_only)


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
    j: dict[str, Any],
    floor: dict[tuple[str, str, str], dict[str, Any]],
    latest: dict[tuple[str, str, str], dict[str, Any]],
    reg: dict[Any, str],
    n: dict[str, int],
) -> str:
    """One markdown row of a job; counts its golden and input outcomes into `n`."""
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
    return f"| {j['family']} | {j['ref']} | {j['mode']} | {gold} | {first.get('load_s', '-')} | {'; '.join(cells)} |"


def table(sel: dict[str, Any], path: Path, families: list[str], modes: list[str], *, golden_only: bool = False) -> int:
    """Markdown: one line per family (verdict), then one per job (golden error, ms vs floor, peak MB)."""
    sc = speed_check()
    reg = {} if golden_only else _regressions(sc, path)
    checks = jobs(sel, families, modes, golden_only=golden_only)
    floor = sc._floor_index() if not golden_only else {}  # noqa: SLF001
    sources = _golden_sources_by_ref(checks, sc._commit()) if golden_only else {}  # noqa: SLF001
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for i, r in enumerate(sc.read_rows(path)):
        if r.get("side") != "lib":
            continue
        if (
            golden_only
            and r.get("input") == GOLDEN
            and (r.get("golden_source") not in sources.get(r["ref"], ()) or not _current(sc, r))
        ):
            continue
        latest[r["ref"], r["mode"], r.get("input", "-")] = r | {"_i": i}
    by_family: dict[str, list[dict[str, Any]]] = {}
    for j in checks:
        by_family.setdefault(j["family"], []).append(j)
    fam_lines, job_lines, bad = [], [], 0
    for fam, js in by_family.items():
        n = dict.fromkeys(("pass", "fail", "skip", "missing", "slow", "error", "timed", "untimed"), 0)
        job_lines += [_job_line(j, floor, latest, reg, n) for j in js]
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
        kind = p.add_mutually_exclusive_group()
        kind.add_argument("--profile", action="store_true", help="also speed_check's profile of one input")
        kind.add_argument("--golden-only", action="store_true", help="only check goldens of the selected revisions")
        if name != "table":
            p.add_argument("--no-core", action="store_true", help="skip the optional runner-only timing")
    p = sub.choices["run"]
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prefix", default="", help="command prefix of each worker, e.g. a GPU lock with a timeout")
    p.add_argument("--force", action="store_true", help="run again even if the code did not change")
    p.add_argument("--batch-size", type=int, default=1, help="correctness-only jobs per worker process")
    sub.add_parser("worker").add_argument("spec")
    a = ap.parse_args()
    if a.cmd == "worker":
        spec = json.loads(a.spec)
        return worker_batch(spec) if isinstance(spec, list) else worker(spec)
    if a.cmd == "run" and (a.batch_size < 1 or (a.batch_size > 1 and not a.golden_only)):
        ap.error("--batch-size must be positive and requires --golden-only for groups larger than one")
    sel = json.loads(a.selection.read_text(encoding="utf-8"))
    if a.cmd == "run":
        return run(a)
    if a.cmd == "table":
        return table(sel, a.out, a.family, a.modes, golden_only=a.golden_only)
    js = jobs(sel, a.family, a.modes, golden_only=a.golden_only)
    floor = speed_check()._floor_index() if not a.golden_only else {}  # noqa: SLF001
    per: dict[str, float] = {}
    for j in js:
        per[j["family"]] = per.get(j["family"], 0.0) + estimate_s(j, floor, profile=a.profile, no_core=a.no_core)
    for fam, t in per.items():
        print(f"{fam}: {sum(j['family'] == fam for j in js)} jobs, ~{t / 60:.1f} min")
    print(f"total: {len(js)} jobs, ~{sum(per.values()) / 60:.0f} min GPU (one lock per job)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
