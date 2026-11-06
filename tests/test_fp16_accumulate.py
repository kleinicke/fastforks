"""The fp16-accumulate flag is refcounted across threads (`_core.half.fp16_accumulate`, shared by the kernels ViT and
the generic patch() path)."""

from __future__ import annotations

import threading

import pytest
import torch

from fastforks import _optimize
from fastforks._core.half import fp16_accumulate
from fastforks.shared.depth_anything_v2 import fast_vit


@pytest.mark.parametrize("ctx", [fp16_accumulate, fast_vit._fp16_accumulate, _optimize.fp16_accumulate])
def test_fp16_accumulate_threads(ctx):
    assert ctx is fp16_accumulate  # one counter for every user
    mm = torch.backends.cuda.matmul
    old = mm.allow_fp16_accumulation
    entered, a_out = threading.Barrier(2), threading.Event()
    seen: dict[str, bool] = {}

    def a() -> None:
        with ctx():
            entered.wait()
        a_out.set()

    def b() -> None:
        with ctx():
            entered.wait()
            a_out.wait()
            seen["b"] = mm.allow_fp16_accumulation  # A left first: B must still accumulate in fp16

    try:
        mm.allow_fp16_accumulation = False
        threads = [threading.Thread(target=f) for f in (a, b)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert seen == {"b": True}
        assert mm.allow_fp16_accumulation is False  # the last one out restores the caller's value
        with ctx(), ctx():
            assert mm.allow_fp16_accumulation is True
        assert mm.allow_fp16_accumulation is False
    finally:
        mm.allow_fp16_accumulation = old
