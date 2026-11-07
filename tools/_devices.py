"""The measured devices (docs/data/devices.toml) and their speed rows: one reference device, any number of surveys.

The reference device's rows are docs/data/speed.jsonl (the speed floor, read as recorded); every other device's rows
are docs/data/speed/<slug>.jsonl in the survey schema of docs/data/README.md ("Devices"), checked by `survey_rows`.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"
REGISTRY = DATA / "devices.toml"
FLOOR = DATA / "speed.jsonl"
BACKENDS = ("cuda", "rocm", "mps")
MODES = ("upstream-default", "upstream", "portable", "kernels")  # the cell order of every table
STATUSES = ("ok", "unsupported", "failed", "oom")
GOLDENS = ("pass", "fail", "skip")  # optional `golden`: the golden check of the same loaded model
# memory_kind -> (row fields, column title). Kinds are never compared with each other.
MEMORY_KINDS = {
    "torch-peak": (("peak_alloc_mb", "peak_reserved_mb"), "peak torch memory (MiB, allocated / reserved)"),
    "process-footprint": (
        ("peak_footprint_mb",),
        "peak process footprint (MiB, model + activations in unified memory)",
    ),
}
# protocol_id -> what it means; a cross-device cell needs the same id on both devices.
PROTOCOLS = {
    "warm-median": "median of warm, synchronised calls after warm-up (at least 5; 2 for calls over 2 s on surveys)",
}
_SLUG = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_DTYPES = ("float32", "bfloat16", "float16")
_WARM_RUNS = 5  # fewest timed calls of a reference row counted as `warm-median`


@dataclass(frozen=True)
class Device:
    """One entry of docs/data/devices.toml."""

    slug: str
    name: str
    backend: str
    match: str
    memory: str
    details: str
    software: str
    reference: bool

    @property
    def data(self) -> Path:
        """The device's rows."""
        return FLOOR if self.reference else DATA / "speed" / f"{self.slug}.jsonl"

    @property
    def page(self) -> str:
        """The device's docs page, relative to docs/."""
        return "benchmarks.md" if self.reference else f"devices/{self.slug}.md"


def devices() -> list[Device]:
    """The registry in its order; exactly one reference device."""
    out = []
    for slug, d in tomllib.loads(REGISTRY.read_text()).items():
        extra = set(d) - {"name", "backend", "match", "memory", "details", "software", "reference"}
        if not _SLUG.match(slug) or extra or d.get("backend") not in BACKENDS:
            msg = f"devices.toml [{slug}]: bad slug, unknown keys {sorted(extra)} or backend not one of {BACKENDS}"
            raise ValueError(msg)
        out.append(
            Device(
                slug,
                d["name"],
                d["backend"],
                d["match"],
                d.get("memory", ""),
                d.get("details", ""),
                d.get("software", ""),
                bool(d.get("reference", False)),
            )
        )
    if sum(d.reference for d in out) != 1:
        msg = "devices.toml: exactly one device needs `reference = true`"
        raise ValueError(msg)
    return out


def reference() -> Device:
    return next(d for d in devices() if d.reference)


def dtype_of(row: dict[str, Any]) -> str:
    """A row's compute dtype: `dtype`, else the recorded backend's, else `float32` for a plain original run."""
    if row.get("dtype"):
        return str(row["dtype"]).removeprefix("torch.")
    backend = row.get("backend")
    if isinstance(backend, dict) and backend.get("dtype"):
        return str(backend["dtype"]).removeprefix("torch.")
    return "float32" if row["mode"] == "upstream-default" else "default"


def protocol_id(row: dict[str, Any]) -> str:
    """`protocol_id` of a survey row; reference rows: `warm-median` for the median-of-warm-runs protocols."""
    if row.get("protocol_id"):
        return str(row["protocol_id"])
    text = row.get("protocol", "")
    m = re.search(r"median of (\d+)", text)
    return "warm-median" if m and int(m[1]) >= _WARM_RUNS and "earlier harness" not in text else "other"


def status_of(row: dict[str, Any]) -> str:
    """`status` of a survey row; a reference row is `ok` when it has a time, else `failed`."""
    return str(row.get("status") or ("ok" if isinstance(row.get("ms"), int | float) else "failed"))


def _check(path: Path, n: int, row: dict[str, Any], dev: Device) -> list[str]:
    where = f"{path.relative_to(ROOT)}:{n}"
    errs = [f"{where}: missing {k}" for k in ("ref", "task", "mode", "input", "date", "protocol") if not row.get(k)]
    status = row.get("status", "")
    if row.get("device") != dev.slug:
        errs.append(f"{where}: device must be {dev.slug!r}")
    allowed = {"mode": MODES, "status": STATUSES, "protocol_id": tuple(PROTOCOLS), "dtype": _DTYPES}
    errs += [f"{where}: {k} not one of {v}" for k, v in allowed.items() if row.get(k) not in v]
    if row.get("golden", "pass") not in GOLDENS:
        errs.append(f"{where}: golden not one of {GOLDENS}")
    if status == "ok":
        if not isinstance(row.get("ms"), int | float):
            errs.append(f"{where}: an ok row needs ms")
        kind = row.get("memory_kind")
        if kind is not None and (kind not in MEMORY_KINDS or any(k not in row for k in MEMORY_KINDS[kind][0])):
            errs.append(f"{where}: memory_kind not one of {sorted(MEMORY_KINDS)} or its fields missing")
    elif "ms" in row or not (row.get("note") or row.get("error")):
        errs.append(f"{where}: a {status} row has no ms and needs a note or error")
    return errs


def survey_rows(dev: Device) -> list[dict[str, Any]]:
    """The rows of a survey device, checked; raises ValueError naming every bad row."""
    path = dev.data
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
    errs = [e for n, r in enumerate(rows, 1) for e in _check(path, n, r, dev)]
    seen: dict[tuple[str, str, str, str], int] = {}
    for n, r in enumerate(rows, 1):
        key = (r.get("ref", ""), r.get("mode", ""), r.get("dtype", ""), r.get("input", ""))
        if key in seen:
            errs.append(f"{path.relative_to(ROOT)}:{n}: same ref / mode / dtype / input as line {seen[key]}")
        seen[key] = n
    inputs: dict[str, set[str]] = {}
    for r in rows:
        inputs.setdefault(r.get("ref", ""), set()).add(r.get("input", ""))
    errs += [  # one input per model, its standard one (speed_survey.plan): the device's tables compare like with like
        f"{path.relative_to(ROOT)}: {ref} has rows at {sorted(s)}; keep only its standard input"
        for ref, s in sorted(inputs.items())
        if len(s) > 1
    ]
    if errs:
        raise ValueError("\n".join(errs))
    return rows


def rows(dev: Device) -> list[dict[str, Any]]:
    """The device's rows: the floor as recorded for the reference, else the checked survey rows."""
    if not dev.reference:
        return survey_rows(dev)
    floor = [json.loads(line) for line in FLOOR.read_text().splitlines() if line.strip()]
    return [r for r in floor if not r.get("cold") or "ms" in r]  # cold-start rows without timings: startup only
