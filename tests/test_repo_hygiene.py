"""Nothing machine-specific ships in the repository: no maintainer home paths, GPU-queue wrappers or run scripts."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"/home/florian|/mnt/hdd|gpulock|cpuslot|\bexp(?:eriment)? 0\d\d\b|\bresults/20\d\d-")
SKIP = {"uv.lock", "tests/test_repo_hygiene.py"}
SKIP_SUFFIXES = (
    ".cu",
    ".jsonl",
)  # .cu comments are hashed into the kernel manifests; rows are cleaned by the release script


def _tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False)  # noqa: S607
    return out.stdout.split() if out.returncode == 0 else []


def test_no_machine_specific_strings():
    hits = []
    for name in _tracked():
        path = ROOT / name
        if name in SKIP or name.endswith(SKIP_SUFFIXES) or not path.is_file():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        hits += [f"{name}: {m.group(0)}" for m in PATTERN.finditer(text)][:1]
    assert not hits, hits


def test_no_run_scripts_tracked():
    assert not [n for n in _tracked() if n.startswith("runs/")]
