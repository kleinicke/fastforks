"""Read and write the per-family `golden.toml` files (`src/fastforks/<task>/models/<family>/golden.toml`).

Each holds `[tolerance]` (parity bounds), `[refs."<id>@<rev>"]` (SHA-256 + provenance of each golden output) and
`[validated."<id>@<rev>"]` (kernel builds a GPU run checked on Linux; a Windows run's record is its
`[validated."<id>@<rev>".windows]` sub-table). A reference's optional `file` names its published
file when multiple versions of a fixture are available. The files stay out of the wheels.

A golden whose reference run itself differs by OS (PyTorch's CPU ops built with MSVC vs GCC) has a per-OS golden: a
`[refs."<id>@<rev>".windows]` sub-table with its own `sha256` and provenance, file `<id>@<rev>.windows.npz`, written
by the family's `tools/golden/<family>.py` run on that OS (`FASTFORKS_GOLDEN_OS=windows`, see `_common.write`).
The tests on that OS compare with it (`platform_golden`); everything else, tolerances included, stays the entry's.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src" / "fastforks"
SECTIONS = ("tolerance", "refs", "validated")
HEADER = """\
# Golden outputs of this family (tests/synth.py pair, files in $FASTFORKS_GOLDEN_PATH), written by
# tools/golden/<family>.py from `upstream` (+ `patch` from tools/golden/patches/). [tolerance]: max |diff| on CPU
# float32 (`cpu_atol`), and on GPU the main output's mean |diff| and 99th percentile (disparity: px; divided by the
# reference's median |value| if `relative = true`). [validated]: kernel builds (`lib@build:target`) a GPU golden run
# passed with (`pytest -m gpu --record`). Not shipped in wheels.
"""
DEFAULT_TOLERANCE = {"cpu_atol": 2e-3, "gpu_mean": 0.02, "gpu_p99": 0.25}
# Per-OS records inside a `[validated."<ref>"]` entry, as `[validated."<ref>".<os>]` sub-tables: a Windows GPU run
# validates the Windows DLL builds of the `shared` kernel libraries (manifest.windows.toml) next to the Linux record,
# which stays the entry's own keys. The same sub-tables in `[refs."<ref>"]`: a per-OS golden file.
OS_SLOTS = ("windows",)


def os_slot() -> str | None:
    """The OS slot of this platform (`"windows"` on win32), or None (Linux, macOS: the entries' own keys)."""
    return "windows" if sys.platform == "win32" else None


def golden_files(ref: str, entry: dict) -> list[tuple[str | None, str, str, str]]:
    """`(slot, local file, published file, sha256)` of every golden file of a `[refs]` entry: its own, then per OS."""
    own = f"{ref}.npz"
    out: list[tuple[str | None, str, str, str]] = [(None, own, entry.get("file", own), entry["sha256"])]
    for slot in OS_SLOTS:
        if slot in entry:
            name = f"{ref}.{slot}.npz"
            out.append((slot, name, entry[slot].get("file", name), entry[slot]["sha256"]))
    return out


def platform_golden(ref: str, entry: dict) -> tuple[str, str, str]:
    """`(local file, published file, sha256)` of the golden the tests on this OS compare with."""
    files = {slot: rest for slot, *rest in golden_files(ref, entry)}
    local, published, sha = files.get(os_slot(), files[None])
    return local, published, sha


def text_codes(texts: list[str]) -> np.ndarray:
    """OCR goldens store texts as `[N, L]` float32 Unicode code points, zero-padded (exact up to 2^24)."""
    out = np.zeros((len(texts), max(map(len, texts), default=0)), np.float32)
    for i, t in enumerate(texts):
        out[i, : len(t)] = [ord(c) for c in t]
    return out


def path(family: str, task: str = "stereo") -> Path:
    """The `golden.toml` of `family`."""
    return SRC / task / "models" / family / "golden.toml"


def load(family: str, task: str = "stereo") -> dict:
    """Contents of a family's file (empty sections if it does not exist yet)."""
    p = path(family, task)
    data = tomllib.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    unknown = set(data) - set(SECTIONS)
    if unknown:
        msg = f"{p}: unknown section(s) {sorted(unknown)}"
        raise ValueError(msg)
    return {"tolerance": dict(DEFAULT_TOLERANCE), "refs": {}, "validated": {}} | data


def load_all(tasks: tuple[str, ...]) -> dict[str, dict]:
    """Every family's file by family name, plus its task under `"task"` (family names are unique across tasks)."""
    out: dict[str, dict] = {}
    for t in tasks:
        for p in sorted(SRC.glob(f"{t}/models/*/golden.toml")):
            if p.parent.name in out:
                msg = f"family {p.parent.name!r} exists in two tasks"
                raise ValueError(msg)
            out[p.parent.name] = load(p.parent.name, t) | {"task": t}
    return out


def _value(v: object) -> str:
    """A TOML value: JSON for scalars and lists, sorted inline tables for dicts (`{"lib" = {"fn" = "hash"}}`)."""
    if isinstance(v, dict):
        return "{" + ", ".join(f"{json.dumps(k)} = {_value(v[k])}" for k in sorted(v)) + "}"
    return json.dumps(v)


def _kept(p: Path) -> dict[tuple[str, str], tuple[list[str], str]]:
    """Lines of an existing file by (table header, key): the comment lines above the key and the key's own line."""
    out: dict[tuple[str, str], tuple[list[str], str]] = {}
    if not p.is_file():
        return out
    table, pending = "", []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            pending.append(line)
        elif line.startswith("["):
            table, pending = line, []
        elif "=" in line:
            out[table, line.split("=", 1)[0].strip()] = (pending, line)
            pending = []
        else:
            pending = []
    return out


def dump(family: str, data: dict, task: str = "stereo") -> None:
    """Write a family's file (sorted, stable formatting; comments and unchanged lines are kept)."""
    p = path(family, task)
    kept = _kept(p)

    def line(header: str, key: str, value: object) -> list[str]:
        above, old = kept.get((header, key), ([], ""))
        same = old and tomllib.loads(old).get(key) == value  # unchanged: keep its formatting and trailing comment
        return [*above, old if same else f"{key} = {_value(value)}"]

    def table(header: str, values: dict) -> list[str]:
        return [header, *(x for k, v in values.items() for x in line(header, k, v))]

    lines = [HEADER, *table("[tolerance]", data["tolerance"])]
    for section in ("refs", "validated"):
        for ref in sorted(data[section]):
            entry = data[section][ref]
            slots = {k: entry[k] for k in OS_SLOTS if k in entry}
            lines += ["", *table(f'[{section}."{ref}"]', {k: v for k, v in entry.items() if k not in slots})]
            for k, v in slots.items():
                lines += ["", *table(f'[{section}."{ref}".{k}]', v)]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")  # LF on Windows too


def validated_records(entry: dict) -> list[dict]:
    """The records of a `[validated."<ref>"]` entry: the Linux one (the entry's own keys, if any), then per-OS ones."""
    own = {k: v for k, v in entry.items() if k not in OS_SLOTS}
    return ([own] if own else []) + [entry[k] for k in OS_SLOTS if k in entry]
