"""The shared SDPA backend policy (`fastforks.shared.sdpa`): parsing, the flash rule, the CUDA-only context."""

import contextlib
import logging

import pytest
import torch
from torch.nn.attention import SDPBackend

from fastforks.shared import sdpa


@pytest.fixture(autouse=True)
def _fresh() -> None:
    sdpa._resolve.cache_clear()  # the per-value cache


@pytest.mark.parametrize("flash", [True, False])
def test_auto_keeps_torch_order_with_flash(monkeypatch: pytest.MonkeyPatch, flash: bool) -> None:  # noqa: FBT001
    monkeypatch.delenv(sdpa.ENV, raising=False)
    monkeypatch.setattr(torch.backends.cuda, "is_flash_attention_available", lambda: flash)
    want = None if flash else (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION)
    assert sdpa.backends() == want


def test_no_flash_choice_is_logged(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv(sdpa.ENV, "auto")
    monkeypatch.setattr(torch.backends.cuda, "is_flash_attention_available", lambda: False)
    with caplog.at_level(logging.INFO, logger="fastforks"):
        sdpa.backends()
    assert "no flash attention" in caplog.text


@pytest.mark.parametrize(
    ("value", "want"),
    [
        ("torch", None),
        ("efficient,math", (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH)),
        (" CUDNN ", (SDPBackend.CUDNN_ATTENTION,)),
        ("flash,flash,math", (SDPBackend.FLASH_ATTENTION, SDPBackend.MATH)),
    ],
)
def test_override(monkeypatch: pytest.MonkeyPatch, value: str, want: tuple[SDPBackend, ...] | None) -> None:
    monkeypatch.setenv(sdpa.ENV, value)
    assert sdpa.backends() == want


@pytest.mark.parametrize("value", ["fast", "cudnn,", "efficient;math"])
def test_unknown_backend_raises(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(sdpa.ENV, value)
    with pytest.raises(ValueError, match="unknown SDPA backend"):
        sdpa.backends()


def test_policy_only_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(sdpa.ENV, "math")
    assert isinstance(sdpa.policy(torch.device("cpu")), contextlib.nullcontext)
    monkeypatch.setenv(sdpa.ENV, "torch")
    assert isinstance(sdpa.policy(torch.device("cuda")), contextlib.nullcontext)
