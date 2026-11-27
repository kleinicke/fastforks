"""`tools/gpu_lock.py` with dummy jobs in a private lock directory (no GPU)."""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
LOCK = TOOLS / "gpu_lock.py"

# a dummy job: appends "<t> start|end <name>" around a short sleep
JOB = """
import sys, time
f, n, d = sys.argv[1], sys.argv[2], float(sys.argv[3])
open(f, "a").write(f"{time.time()} start {n}\\n")
time.sleep(d)
open(f, "a").write(f"{time.time()} end {n}\\n")
"""


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    return {**os.environ, "FASTFORKS_GPU_LOCK_DIR": str(tmp_path / "lock"), "PYTHONPATH": str(TOOLS)}


def _job(env: dict[str, str], log: Path, kind: str, name: str, secs: float) -> subprocess.Popen:
    cmd = [sys.executable, str(LOCK), *([kind] if kind else []), sys.executable, "-c", JOB, str(log), name, str(secs)]
    return subprocess.Popen(cmd, env=env, stderr=subprocess.DEVNULL)


def _events(log: Path) -> list[tuple[float, str, str]]:
    rows = [line.split() for line in log.read_text().splitlines()]
    return sorted((float(t), what, n) for t, what, n in rows)


def _overlap(ev: list[tuple[float, str, str]], a: str, b: str) -> bool:
    span = {n: [t for t, _, m in ev if m == n] for n in (a, b)}
    return span[a][0] < span[b][1] and span[b][0] < span[a][1]


def test_exclusive_light_and_short(env: dict[str, str], tmp_path: Path) -> None:
    log = tmp_path / "ev.log"
    ps = [_job(env, log, "", "X1", 1.5)]
    time.sleep(0.4)
    ps.append(_job(env, log, "", "X2", 0.5))  # queued first, but yields to the short and light jobs after it
    time.sleep(0.3)
    ps += [_job(env, log, "--light", "L1", 1.0), _job(env, log, "--light", "L2", 1.0)]
    ps += [_job(env, log, "--light", "L3", 0.3), _job(env, log, "--short", "S1", 0.3)]
    for p in ps:
        assert p.wait(30) == 0
    ev = _events(log)
    order = [n for _, what, n in ev if what == "start"]
    assert order[0] == "X1"
    assert order[-1] == "X2"
    assert _overlap(ev, "L1", "L2")  # two light slots
    for x in ("X1", "S1", "X2"):
        assert not any(_overlap(ev, x, o) for o in ("L1", "L2", "L3") if o != x)
    lights = [n for n in order if n.startswith("L")]
    assert not (_overlap(ev, "L3", "L1") and _overlap(ev, "L3", "L2") and _overlap(ev, "L1", "L2"))  # never three
    assert len(lights) == 3


def test_excludes_old_gpu_sh(env: dict[str, str], tmp_path: Path) -> None:
    """A holder of `gpu.lock` as the old `gpu.sh` takes it (exclusive flock) keeps light jobs out."""
    d = Path(env["FASTFORKS_GPU_LOCK_DIR"])
    d.mkdir()
    log = tmp_path / "ev.log"
    with (d / "gpu.lock").open("w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        p = _job(env, log, "--light", "L", 0.1)
        time.sleep(1.0)
        assert not log.exists()
        fcntl.flock(f, fcntl.LOCK_UN)
    assert p.wait(30) == 0
    assert log.exists()


TOOL = """
import sys, time
import gpu_lock
f, n = sys.argv[1], sys.argv[2]
def span(what):
    open(f, "a").write(f"{time.time()} start {n}-{what}\\n")
    time.sleep(0.6)
    open(f, "a").write(f"{time.time()} end {n}-{what}\\n")
with gpu_lock.session(n):
    span("light")
    with gpu_lock.timed():
        span("timed")
"""


def test_timed_tools(env: dict[str, str], tmp_path: Path) -> None:
    """Under `--timed` a tool's worker shares the GPU outside its timed section only."""
    tool = tmp_path / "mlx_core_ab.py"
    tool.write_text(TOOL)
    log = tmp_path / "ev.log"
    ps = [
        subprocess.Popen([sys.executable, str(LOCK), "--timed", sys.executable, str(tool), str(log), n], env=env)
        for n in ("A", "B")
    ]
    for p in ps:
        assert p.wait(30) == 0
    ev = _events(log)
    assert _overlap(ev, "A-light", "B-light")
    assert not _overlap(ev, "A-timed", "B-timed")
    assert not _overlap(ev, "A-timed", "B-light")
    assert not _overlap(ev, "B-timed", "A-light")
    # outside --timed the helpers do nothing (the tools under the old gpu.sh)
    out = subprocess.run([sys.executable, str(tool), str(tmp_path / "plain.log"), "C"], env=env, check=False)
    assert out.returncode == 0
    # --timed refuses other commands
    other = subprocess.run([sys.executable, str(LOCK), "--timed", "true"], env=env, check=False, capture_output=True)
    assert other.returncode == 2
