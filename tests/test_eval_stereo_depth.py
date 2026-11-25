"""Stereo and depth evaluation resume from the recorded dataset keys."""

from __future__ import annotations

import argparse
import json
import time
from threading import current_thread, main_thread
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

import eval_stereo_depth as evaluator

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_depth_read_ahead_resumes_in_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ref = "depth-model"
    target = tmp_path / "depth" / f"{ref}__portable.jsonl"
    target.parent.mkdir()
    target.write_text('{"ds": "nyu", "key": "done"}\n', encoding="utf-8")
    decoded = []

    def read(key: str) -> tuple[str, str]:
        assert current_thread() is not main_thread()
        decoded.append(key)
        return key, key

    names = ("done", "first", "second")
    monkeypatch.setattr(evaluator, "depth_items", lambda _dataset: [(key, lambda key=key: read(key)) for key in names])
    monkeypatch.setattr(evaluator, "_commit", lambda: "test")
    monkeypatch.setattr(evaluator, "_item", lambda pred, *_: {"prediction": pred})
    predictor = SimpleNamespace(
        mode="portable",
        dtype="float16",
        how=None,
        note="",
        backend="test",
        kind="relative",
        prep=tuple,
        call=lambda images, _ds: (images[0], "depth"),
    )
    monkeypatch.setattr(evaluator, "Predictor", lambda *_args, **_kwargs: predictor)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test")
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    args = argparse.Namespace(
        task="depth", out=str(tmp_path), datasets="nyu", tiers="portable", limit=0, max_seconds=60, build_reserve=0
    )
    assert evaluator.run_one(args, ref, time.perf_counter()) == 0
    assert decoded == ["first", "second"]
    records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert [row["key"] for row in records if "key" in row] == list(names)
    assert [row["prediction"] for row in records if "prediction" in row] == decoded
