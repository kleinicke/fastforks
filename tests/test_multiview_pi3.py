"""Pi3 decoder block streaming."""

from types import SimpleNamespace

import pytest
import torch

from fastforks._core.errors import OptionsError
from fastforks.multiview.models.pi3.model import Params, _net


@pytest.mark.parametrize(("device", "dtype"), [("cpu", "float16"), ("cuda", "float32")])
def test_stream_requires_cuda_fp16_before_build(device: str, dtype: str) -> None:
    ctx = SimpleNamespace(device=torch.device(device), dtype=dtype, params=Params(255000, stream=True))
    with pytest.raises(OptionsError, match="requires CUDA and float16"):
        _net(ctx)  # ty: ignore[invalid-argument-type] - no checkpoint access should occur
