"""`cnn_kernels.nchw`: a channels-last map as NCHW with the same values (identity GEMM on CUDA fp16 / bf16)."""

from __future__ import annotations

import pytest
import torch

from fastforks.shared import cnn_kernels


def test_nchw_cpu_is_contiguous_copy() -> None:
    x = torch.randn(2, 16, 5, 7).contiguous(memory_format=torch.channels_last)
    y = cnn_kernels.nchw(x)
    assert y.is_contiguous()
    assert torch.equal(y, x)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("channels", [8, 160, 320, 3])
def test_nchw_cuda_exact(dtype: torch.dtype, channels: int) -> None:
    g = torch.Generator(device="cuda").manual_seed(0)
    x = (torch.randn(2, channels, 33, 47, device="cuda", generator=g) * 1e4).to(dtype)
    x = x.contiguous(memory_format=torch.channels_last)
    y = cnn_kernels.nchw(x)
    assert y.is_contiguous()
    assert torch.equal(y, x)
