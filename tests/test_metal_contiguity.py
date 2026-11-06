"""Metal kernel calls reject non-contiguous tensors (CPU test with a fake shader library; no Mac available)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fastforks._core.metal import MetalLibrary
from fastforks.kernels.vggt_metal import qk_norm_rope


def _lib(calls: list) -> MetalLibrary:
    lib = MetalLibrary.__new__(MetalLibrary)
    lib._lib = SimpleNamespace(
        k_float=lambda *a, **kw: calls.append((a, kw)), qk_norm_rope_float=lambda *a, **kw: calls.append((a, kw))
    )
    return lib


def test_kernel_call_rejects_strided_tensor_args():
    calls: list = []
    k = _lib(calls).kernel("k", torch.float32)
    x = torch.zeros(4, 8)
    k(x, 3, 1.5, threads=4)  # contiguous tensors and scalars pass through unchanged
    assert calls == [((x, 3, 1.5), {"threads": 4})]
    with pytest.raises(ValueError, match=r"argument 1.*not contiguous"):
        k(x, x.t(), threads=4)
    with pytest.raises(ValueError, match=r"not contiguous"):
        k(x[:, ::2], threads=4)
    k(torch.zeros(1, 5).expand(1, 5), threads=1)  # size-1 dims never count as strided


def test_vggt_wrapper_passes_contiguous_tables_and_affine():
    """Slices / transposes of the RoPE tables and the LayerNorm affine are made contiguous before the kernel."""
    calls: list = []
    qk = torch.randn(1, 3, 2, 2, 64)
    cos, sin = (torch.randn(64, 3).t() for _ in range(2))  # [3, 64] but strided
    affine = torch.randn(64, 2).t(), torch.randn(2, 128)[:, ::2]
    qk_norm_rope(_lib(calls), qk, cos, sin, affine, 1e-6)
    ((args, _),) = calls
    assert all(a.is_contiguous() for a in args if isinstance(a, torch.Tensor))
