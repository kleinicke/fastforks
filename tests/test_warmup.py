"""The CUDA warm-up thread: runs once per device, calls wait for it, and failures stay silent."""

from __future__ import annotations

import contextlib
import threading
import time

import pytest
import torch

from fastforks._core import warmup


def test_wait_without_a_warm_up_returns() -> None:
    warmup.wait(torch.device("cuda", 97))


def test_a_failing_warm_up_is_silent(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def boom(device: torch.device, stop: object) -> None:
        msg = f"no {device}"
        raise RuntimeError(msg)

    monkeypatch.setattr(warmup, "_ops", boom)
    monkeypatch.setattr(warmup, "_threads", {})
    with caplog.at_level("DEBUG", logger="fastforks"):
        warmup.start(torch.device("cuda", 98))
        warmup.wait(torch.device("cuda", 98))
    assert "no cuda:98" in caplog.text
    assert not warmup._threads[98][0].is_alive()


def test_runs_once_per_device(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(warmup, "_ops", lambda device, _stop: calls.append(device))
    monkeypatch.setattr(warmup, "_threads", {})
    for _ in range(3):
        warmup.start(torch.device("cuda", 99))
        warmup.wait(torch.device("cuda", 99))
    assert calls == [torch.device("cuda", 99)]


def test_wait_skips_the_remaining_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    started, done = threading.Event(), []

    def steps(device: torch.device) -> list:
        def first() -> None:
            started.set()
            time.sleep(0.2)
            done.append(1)

        return [first, lambda: done.append(2)]

    monkeypatch.setattr(warmup, "_steps", steps)
    monkeypatch.setattr(warmup.torch.cuda, "device", lambda _d: contextlib.nullcontext())
    monkeypatch.setattr(warmup.torch.cuda, "synchronize", lambda _d: None)
    monkeypatch.setattr(warmup, "_threads", {})
    warmup.start(torch.device("cuda", 96))
    started.wait()
    warmup.wait(torch.device("cuda", 96))
    assert done == [1]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_the_ops_run_on_the_gpu() -> None:
    warmup._ops(torch.device("cuda", 0), threading.Event())
    a = torch.ones(8, 8, device="cuda", dtype=torch.float16)
    assert (a @ a).eq(8).all()
