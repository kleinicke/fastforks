"""TAP-Vid prefetch keeps the scoring order and the warm-up sequence intact."""

from __future__ import annotations

import argparse
import json
from threading import current_thread, main_thread
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

import eval_tracking as ev

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("budget", [30.0, -1.0])
def test_tracking_prefetch_resumes_and_reuses_warmup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: float
) -> None:
    ref, tier = "track-on2", "portable"
    path = ev._f(tmp_path, ref, tier)
    path.write_text('{"seq": "finished"}\n')
    decoded: list[str] = []
    infer: list[np.ndarray] = []

    def load(name: str, *, native: bool) -> ev.SequenceData:
        assert current_thread() is not main_thread()
        assert not native
        decoded.append(name)
        frame_id = {"first": 1, "second": 2}[name]
        frames = np.full((2, 4, 6, 3), frame_id, dtype=np.uint8)
        return frames, np.zeros((1, 3)), np.zeros((1, 2, 2)), np.zeros((1, 2), dtype=bool)

    class Tracker:
        backend = SimpleNamespace(mode="portable", dtype="float16")

        def __call__(self, frames: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            assert queries.shape == (1, 3)
            infer.append(frames)
            return np.zeros((1, 2, 2)), np.zeros((1, 2), dtype=bool)

    cuda = SimpleNamespace(
        empty_cache=lambda: None,
        reset_peak_memory_stats=lambda: None,
        synchronize=lambda: None,
        get_device_name=lambda: "test",
        max_memory_allocated=lambda: 0,
        max_memory_reserved=lambda: 0,
    )
    monkeypatch.setattr(ev, "torch", SimpleNamespace(cuda=cuda, __version__="test"))
    monkeypatch.setattr(ev, "Ours", lambda _ref, _tier: Tracker())
    monkeypatch.setattr(ev, "sequences", lambda _step: ["finished", "first", "second"])
    monkeypatch.setattr(ev, "load_sequence", load)

    args = argparse.Namespace(out=str(tmp_path), every=3, budget=budget)
    complete = ev._run(args, ref, tier)
    assert decoded == ["first", "second"]
    records = [json.loads(line)["seq"] for line in path.read_text().splitlines()]
    if budget < 0:
        assert not complete
        assert len(infer) == 1
        assert records == ["finished"]
    else:
        assert complete
        assert [int(frames.flat[0]) for frames in infer] == [1, 1, 2]
        assert infer[0] is infer[1]
        assert records == ["finished", "first", "second"]
