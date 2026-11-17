"""Compiled TAPNext++ must accept numpy frames and retain its GPU golden accuracy."""

import pytest
import torch

from fastforks import tracking
from test_golden import _check, forward, golden, options, outputs


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.model
def test_compiled_tapnext_pp_numpy_video_matches_golden() -> None:
    ref = "tapnext-pp@1"
    want = golden(ref)
    model = tracking.load(ref, mode="portable", device="cuda", **(options(ref, want) or {}))
    with torch.inference_mode():
        model.compile()
        actual, _ = forward(model, ref, want)
    _check(ref, "portable", outputs(ref, want), actual, "compiled: ")
