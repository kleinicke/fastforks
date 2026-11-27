"""`tools/merge_rows.py`: per-agent fragments of the shared row files, and their merge (temp files only)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import merge_rows as mr


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    speed = tmp_path / "docs/data/speed"
    speed.mkdir(parents=True)
    monkeypatch.setattr(mr, "ROOT", tmp_path)
    monkeypatch.setattr(mr, "SPEED", speed)
    monkeypatch.setattr(mr, "PENDING", speed / "_pending")
    monkeypatch.setattr(mr, "AB", tmp_path / "docs/plans/mlx-core-ab.jsonl")
    monkeypatch.delenv(mr.AGENT_ENV, raising=False)
    return tmp_path


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _speed(ref: str, ms: float, mode: str = "portable") -> dict:
    return {"task": "depth", "ref": ref, "mode": mode, "dtype": "float16", "input": "518x518", "ms": ms}


def test_fragment_redirects_only_shared_files(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dev = mr.SPEED / "apple-m1-pro-mlx-16gb.jsonl"
    assert mr.fragment(dev) == dev
    monkeypatch.setenv(mr.AGENT_ENV, "opus-port-seg")
    assert mr.fragment(dev) == mr.PENDING / "opus-port-seg" / dev.name
    assert mr.fragment(mr.AB) == mr.PENDING / "opus-port-seg" / mr.AB.name
    scratch = tree / "scratch/ab.jsonl"
    assert mr.fragment(scratch) == scratch
    with pytest.raises(ValueError, match="plain name"):
        mr.fragment(dev, agent="../x")
    assert mr.target_of(mr.fragment(dev)) == dev
    assert mr.target_of(mr.fragment(mr.AB)) == mr.AB


def test_merge_last_writer_wins_and_removes_fragments(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dev = mr.SPEED / "apple-m1-pro-mlx-16gb.jsonl"
    _write(dev, [_speed("b", 5.0), _speed("a", 1.0)])
    old, new = mr.PENDING / "one" / dev.name, mr.PENDING / "two" / dev.name
    _write(old, [_speed("a", 2.0), _speed("c", 9.0, mode="kernels")])
    _write(new, [_speed("a", 3.0)])
    os.utime(old, (1, 1))  # the older fragment
    assert mr.main(["--dry-run"]) == 0
    assert old.exists()
    assert new.exists()
    monkeypatch.setattr(mr.subprocess, "run", lambda *_a, **_k: pytest.fail("no A/B rows: no md"))
    assert mr.main([]) == 0
    rows = mr.read(dev)
    assert [(r["ref"], r["mode"], r["ms"]) for r in rows] == [
        ("a", "portable", 3.0),
        ("b", "portable", 5.0),
        ("c", "kernels", 9.0),
    ]
    assert not mr.PENDING.exists() or not any(mr.PENDING.iterdir())


def test_ab_fragment_regenerates_md(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    row = {"ref": "r", "mode": "portable", "dtype": "float16", "hand": {}, "graph": {}}
    _write(mr.PENDING / "one" / mr.AB.name, [row])
    calls = []
    monkeypatch.setattr(mr.subprocess, "run", lambda cmd, **_k: calls.append(cmd) or type("P", (), {"returncode": 0})())
    assert mr.main(["--keep"]) == 0
    assert mr.read(mr.AB) == [row]
    assert calls
    assert calls[0][-1] == "md"
    assert (mr.PENDING / "one" / mr.AB.name).exists()
