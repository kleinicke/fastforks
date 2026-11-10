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


def _any_lib(calls: list) -> MetalLibrary:
    """A fake library whose every kernel records its arguments."""
    lib = MetalLibrary.__new__(MetalLibrary)

    class _Funcs:
        def __getattr__(self, _name: str):
            return lambda *a, **kw: calls.append((a, kw))

    lib._lib = _Funcs()
    return lib


def test_pose_and_vit_wrappers_check_shapes_and_pass_contiguous_args():
    """The pose_metal / vit_metal wrappers make strided operands contiguous and reject shapes the kernels misread."""
    from fastforks.kernels import pose_metal, vit_metal  # noqa: PLC0415

    calls: list = []
    lib = _any_lib(calls)
    cols = torch.randn(2, 16 * 64, 3)[..., :2].transpose(1, 2)  # [2, 2, 1024]: strided, 2 x 1 input, 64 channels
    assert pose_metal.deconv4s2_relu(lib, cols, torch.zeros(64), 2, 1).shape == (2, 8, 64)
    with pytest.raises(ValueError, match="deconv4s2_relu"):
        pose_metal.deconv4s2_relu(lib, cols, torch.zeros(64), 1, 1)
    hm = torch.randn(3, 17, 8, 6)
    assert pose_metal.dark_decode(lib, hm, 17, channels_last=False, sigma=0.8, radius=5, clip=(1e-3, 50)).shape == (
        3,
        17,
        3,
    )
    with pytest.raises(ValueError, match="dark_decode"):
        pose_metal.dark_decode(lib, hm, 17, channels_last=True, sigma=0.8, radius=5, clip=(1e-3, 50))
    with pytest.raises(ValueError, match="dark_decode"):
        pose_metal.dark_decode(lib, hm.half(), 17, channels_last=False, sigma=0.8, radius=5, clip=(1e-3, 50))
    assert pose_metal.warp_crop(lib, torch.rand(1, 3, 9, 7), torch.zeros(2, 4), torch.zeros(2, 5)).shape == (2, 3, 5, 4)
    with pytest.raises(ValueError, match="warp_crop"):
        pose_metal.warp_crop(lib, torch.rand(1, 3, 9, 7), torch.zeros(2, 4), torch.zeros(3, 5))
    qkv = torch.randn(2, 5, 3, 4, 8)
    q, k = vit_metal.pe_rope(lib, qkv, torch.randn(16, 4).t(), 1)  # [4, 16] strided table
    assert q.shape == k.shape == (2, 4, 5, 8)
    with pytest.raises(ValueError, match="pe_rope"):
        vit_metal.pe_rope(lib, qkv, torch.randn(5, 16), 1)  # the prefix token has no row
    assert all(a.is_contiguous() for args, _ in calls for a in args if isinstance(a, torch.Tensor))
