"""The shared Mac's GPU lock: exclusive for timing, two shared "light" slots for correctness jobs.

    python tools/gpu_lock.py [--light | --short | --timed] CMD ARGS...   # run CMD under the lock
    python tools/gpu_lock.py status                                      # who holds / waits
    python tools/gpu_lock.py install DIR                                 # copy this file to DIR/gpu_lock.py

One lock file, `gpu.lock` in `FASTFORKS_GPU_LOCK_DIR` (default `/Volumes/Daten/ff-mlx`), the one the old `gpu.sh`
takes exclusively for a whole command; so jobs under either script exclude each other the same way.

- default: the exclusive lock for the whole command, as `gpu.sh` (timing, anything unknown).
- `--short`: the same, but queued ahead of the exclusive waiters that are not short (a check of a few minutes should
  not wait behind a survey).
- `--light`: a correctness-only job (goldens, match tests): the lock shared, at most `SLOTS` light jobs at once. Light
  jobs run beside each other, never beside an exclusive holder; they are queued ahead of non-short exclusive waiters.
- `--timed`: for `tools/speed_survey.py` and `tools/mlx_core_ab.py` only. The command runs without a lock; each of its
  worker processes holds a light slot while it loads, converts and checks goldens, and the exclusive lock only around
  its timed calls (`session()` / `timed()` below). Between two timed sections the lock is free, so short and light
  jobs get in between the refs of a long survey.

Priority: a waiter for the exclusive lock that is not `--short` first waits (up to `YIELD_S`) while a short or light
waiter is queued; a light waiter in turn holds back (up to `YIELD_S`) while an exclusive waiter has waited longer than
`STARVE_S`, so neither side starves. Waits over 5 s print `[gpu.sh] waited Ns ...` on stderr, as `gpu.sh` does.

Tools use it as:

    with gpu_lock.session():        # light while the process uses the GPU outside the timed calls
        ...load, first call...
        with gpu_lock.timed():      # exclusive around the timed calls
            ...warm-up and timed calls...

Both are no-ops unless the process runs under `--timed` (`FASTFORKS_GPU_LOCK=timed`), so the tools behave as before
under the old `gpu.sh` (whose exclusive lock their workers must not ask for again: that would deadlock).
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

MODE_ENV = "FASTFORKS_GPU_LOCK"  # "timed": the tools lock their timed sections themselves
DIR_ENV = "FASTFORKS_GPU_LOCK_DIR"
DEFAULT_DIR = "/Volumes/Daten/ff-mlx"
SLOTS = 2  # light jobs at once
YIELD_S = 600.0  # longest an exclusive waiter steps back for queued short / light jobs (and light for a starved one)
STARVE_S = 300.0  # an exclusive waiter queued this long holds new light jobs back
POLL_S = 0.25
TIMED_TOOLS = ("speed_survey.py", "mlx_core_ab.py")


def lock_dir() -> Path:
    return Path(os.environ.get(DIR_ENV) or DEFAULT_DIR)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class _Queue:
    """Waiter markers `<dir>/gpu-queue/<kind>-<pid>-<n>.json` (kinds: `prio`, `excl`), dead pids ignored."""

    def __init__(self, kind: str, what: str) -> None:
        self.path = lock_dir() / "gpu-queue" / f"{kind}-{os.getpid()}-{time.monotonic_ns()}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"pid": os.getpid(), "since": time.time(), "what": what}))

    def done(self) -> None:
        self.path.unlink(missing_ok=True)

    @staticmethod
    def waiting(kind: str, *, older_than: float = 0.0) -> list[dict[str, Any]]:
        out = []
        for p in (lock_dir() / "gpu-queue").glob(f"{kind}-*.json"):
            with contextlib.suppress(OSError, json.JSONDecodeError, KeyError):
                w = json.loads(p.read_text())
                if not _alive(int(w["pid"])):
                    p.unlink(missing_ok=True)
                    continue
                if time.time() - w["since"] >= older_than:
                    out.append(w | {"file": p.name})
        return out


def _flock(f: Any, op: int) -> None:  # noqa: ANN401 - a file object
    fcntl.flock(f, op)


def _try(f: Any, op: int) -> bool:  # noqa: ANN401
    try:
        fcntl.flock(f, op | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _report(t0: float, what: str) -> None:
    waited = time.monotonic() - t0
    if waited > 5:  # noqa: PLR2004 - as gpu.sh
        print(f"[gpu.sh] waited {waited:.0f}s for the GPU lock ({what})", file=sys.stderr, flush=True)


class Lock:
    """One process's hold on the GPU lock: nothing, light (shared + a slot) or exclusive."""

    def __init__(self, what: str = "") -> None:
        """Holding nothing yet; `what` names the job in the queue markers."""
        self.what = what or " ".join(sys.argv[:3])
        self.main: Any = None
        self.slot: Any = None
        self.state = "none"

    def _open(self) -> Any:  # noqa: ANN401
        if self.main is None:
            d = lock_dir()
            d.mkdir(parents=True, exist_ok=True)
            self.main = (d / "gpu.lock").open("a")
        return self.main

    def exclusive(self, *, short: bool = False) -> None:
        """Take the lock exclusively (from none or light: a light hold is given up first)."""
        self.release()
        f = self._open()
        t0 = time.monotonic()
        q = _Queue("prio" if short else "excl", self.what)
        try:
            while True:
                # step back for queued short / light jobs (not forever), then try
                yielding = not short and time.monotonic() - t0 < YIELD_S and _Queue.waiting("prio")
                if not yielding and _try(f, fcntl.LOCK_EX):
                    break
                time.sleep(POLL_S)
        finally:
            q.done()
        _report(t0, "exclusive")
        self.state = "exclusive"

    def light(self) -> None:
        """Take a light slot and the lock shared (from none or exclusive)."""
        self.release()
        f = self._open()
        t0 = time.monotonic()
        q = _Queue("prio", self.what)
        try:
            d = lock_dir()
            while self.slot is None:
                for i in range(SLOTS):
                    s = (d / f"gpu-light-{i}.lock").open("a")
                    if _try(s, fcntl.LOCK_EX):
                        self.slot = s
                        break
                    s.close()
                else:
                    time.sleep(POLL_S)
            while _Queue.waiting("excl", older_than=STARVE_S) and time.monotonic() - t0 < YIELD_S:
                time.sleep(POLL_S)  # an exclusive waiter is starving: let it in first
            while not _try(f, fcntl.LOCK_SH):
                time.sleep(POLL_S)
        finally:
            q.done()
        _report(t0, "light")
        self.state = "light"

    def release(self) -> None:
        """Hold nothing."""
        if self.main is not None and self.state != "none":
            _flock(self.main, fcntl.LOCK_UN)
        if self.slot is not None:
            _flock(self.slot, fcntl.LOCK_UN)
            self.slot.close()
            self.slot = None
        self.state = "none"


_HELD: Lock | None = None


def _timed_mode() -> bool:
    return os.environ.get(MODE_ENV) == "timed"


@contextlib.contextmanager
def session(what: str = "") -> Iterator[None]:
    """A light hold for the rest of the block (a no-op outside `--timed`)."""
    global _HELD  # noqa: PLW0603 - one hold per process
    if not _timed_mode() or _HELD is not None:
        yield
        return
    _HELD = Lock(what)
    _HELD.light()
    try:
        yield
    finally:
        _HELD.release()
        _HELD = None


@contextlib.contextmanager
def timed() -> Iterator[None]:
    """The exclusive lock for the block, then back to light (a no-op outside `--timed` or outside a `session`)."""
    if _HELD is None:
        yield
        return
    _HELD.exclusive()
    try:
        yield
    finally:
        _HELD.light()


def status() -> dict[str, Any]:
    """Who waits (the queue markers) and whether the lock is free right now (probed without waiting)."""
    d = lock_dir()
    free = False
    with contextlib.suppress(OSError), (d / "gpu.lock").open("a") as f:
        free = _try(f, fcntl.LOCK_EX)
        if free:
            _flock(f, fcntl.LOCK_UN)
    light = 0
    for i in range(SLOTS):
        with contextlib.suppress(OSError), (d / f"gpu-light-{i}.lock").open("a") as s:
            if _try(s, fcntl.LOCK_EX):
                _flock(s, fcntl.LOCK_UN)
            else:
                light += 1
    return {"free": free, "light_slots_taken": light, "prio": _Queue.waiting("prio"), "excl": _Queue.waiting("excl")}


def main(argv: list[str]) -> int:
    """Command line (see the module docstring)."""
    if argv[:1] == ["status"]:
        print(json.dumps(status(), indent=1))
        return 0
    if argv[:1] == ["install"] and len(argv) == 2:  # noqa: PLR2004
        dest = Path(argv[1]) / "gpu_lock.py"
        shutil.copyfile(__file__, dest)
        print(f"installed {dest}")
        return 0
    kind = "exclusive"
    if argv[:1] in (["--light"], ["--short"], ["--timed"], ["--exclusive"]):
        kind, argv = argv[0][2:], argv[1:]
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    if kind == "timed":
        if not any(Path(a).name in TIMED_TOOLS for a in argv[:4]):
            tools = " / ".join(TIMED_TOOLS)
            print(f"[gpu.sh] --timed is for {tools} only (they lock their timed calls)", file=sys.stderr)
            return 2
        return subprocess.call(argv, env={**os.environ, MODE_ENV: "timed", DIR_ENV: str(lock_dir())})
    held = Lock(" ".join(argv[:3]))
    if kind == "light":
        held.light()
    else:
        held.exclusive(short=kind == "short")
    env = {k: v for k, v in os.environ.items() if k != MODE_ENV}  # a nested tool must not lock again
    try:
        return subprocess.call(argv, env=env)
    finally:
        held.release()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
