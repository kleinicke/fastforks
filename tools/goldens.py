"""Read and write the per-family `golden.toml` files (`src/fastforks/<task>/models/<family>/golden.toml`).

Each holds `[tolerance]` (parity bounds), `[refs."<id>@<rev>"]` (SHA-256 + provenance of each golden output) and
`[validated."<id>@<rev>"]` (kernel builds a GPU run checked). The files stay out of the wheels.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

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


def path(family: str, task: str = "stereo") -> Path:
    """The `golden.toml` of `family`."""
    return SRC / task / "models" / family / "golden.toml"


def load(family: str, task: str = "stereo") -> dict:
    """Contents of a family's file (empty sections if it does not exist yet)."""
    p = path(family, task)
    data = tomllib.loads(p.read_text()) if p.is_file() else {}
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


def dump(family: str, data: dict, task: str = "stereo") -> None:
    """Write a family's file (sorted, stable formatting)."""
    lines = [HEADER, "[tolerance]", *(f"{k} = {_value(v)}" for k, v in data["tolerance"].items())]
    for section in ("refs", "validated"):
        for ref in sorted(data[section]):
            lines += ["", f'[{section}."{ref}"]', *(f"{k} = {_value(v)}" for k, v in data[section][ref].items())]
    path(family, task).write_text("\n".join(lines) + "\n")
