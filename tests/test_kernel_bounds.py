"""Bounds and error paths of the kernels (review 2026-10-31): conv_fused destinations, per-shape CUTLASS
fallback, fs_cutlass return codes, and the attn_d384 fp16 QK range."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from fastforks._core import kernels
from fastforks.kernels import fs_cutlass
from fastforks.kernels import s2m2 as s2m2_kernels
from fastforks.shared import cnn_kernels
from fastforks.stereo.models.s2m2.kernels import inference as s2m2

CL = torch.channels_last


def _conv(k=128, c=16):
    m = nn.Conv2d(c, k, 3, padding=1, bias=False).half().to(memory_format=CL)
    m._fuse = True
    return m


def _x(h=8, w=8, c=16):
    return torch.zeros((1, c, h, w), dtype=torch.half).contiguous(memory_format=CL)


@pytest.fixture
def fake_cutlass(monkeypatch):
    """cnn_kernels.cutlass replaced by a stub that rejects the sizes in `reject` and records the accepted calls."""
    state = {"reject": set(), "ran": []}

    def run(cfg, x, w, hw, e, padding=None, stride=1):  # noqa: PLR0913, PLR0917
        if tuple(hw) in state["reject"]:
            return False
        state["ran"].append(tuple(hw))
        return True

    monkeypatch.setattr(cnn_kernels, "cutlass", run)
    monkeypatch.setattr(cnn_kernels, "fused_cfg", lambda p: 0)  # noqa: ARG005
    monkeypatch.setattr(cnn_kernels, "narrow_cfg", lambda p, k, dev: 0)  # noqa: ARG005
    return state


@pytest.mark.parametrize(
    ("c", "c0", "match"),
    [
        (256, 4, "multiple of 8"),
        (256, 136, r"136 \+ 128 <= 256"),
        (256, -8, "multiple of 8"),
        (120, 0, r"0 \+ 128 <= 120"),
    ],
)
def test_conv_fused_rejects_bad_channel_offset(fake_cutlass, c, c0, match):
    buf = torch.zeros((1, c, 8, 8), dtype=torch.half).contiguous(memory_format=CL)
    with pytest.raises(RuntimeError, match=match):
        cnn_kernels.conv_fused(_conv(), _x(), None, None, 0, out=(buf, c0))
    assert fake_cutlass["ran"] == []


def test_conv_fused_rejects_misfit_buffers(fake_cutlass):
    m, x = _conv(), _x()
    nchw = torch.zeros((1, 256, 8, 8), dtype=torch.half)
    with pytest.raises(RuntimeError, match="channels-last fp16"):
        cnn_kernels.conv_fused(m, x, None, None, 0, out=(nchw, 0))
    small = torch.zeros((1, 256, 4, 8), dtype=torch.half).contiguous(memory_format=CL)
    with pytest.raises(RuntimeError, match="channels-last fp16"):
        cnn_kernels.conv_fused(m, x, None, None, 0, out=(small, 0))
    buf = torch.zeros((1, 256, 8, 8), dtype=torch.half).contiguous(memory_format=CL)
    other = torch.zeros((1, 384, 8, 8), dtype=torch.half).contiguous(memory_format=CL)
    with pytest.raises(RuntimeError, match="out2"):
        cnn_kernels.conv_fused(m, x, None, None, 0, out=(buf, 0), out2=(other, 0))
    res = torch.zeros((1, 128, 8, 8), dtype=torch.half).contiguous(memory_format=CL)
    with pytest.raises(RuntimeError, match="`res` with `out`"):
        cnn_kernels.conv_fused(m, x, None, None, 3, out=(buf, 0), res=res)
    with pytest.raises(RuntimeError, match="res must be"):
        cnn_kernels.conv_fused(m, x, None, None, 3, res=res[:, :64])
    assert fake_cutlass["ran"] == []
    assert cnn_kernels.conv_fused(m, x, None, None, 0, out=(buf, 128)) is buf  # a valid offset runs


def test_conv_fused_fallback_is_per_shape(fake_cutlass):
    m = _conv()
    fake_cutlass["reject"].add((8, 8))
    assert cnn_kernels.conv_fused(m, _x(8, 8), None, None, 0) is None
    assert m._fuse  # the module stays marked for the kernel
    assert not cnn_kernels.fuse_ok(m, 8, 8)
    assert cnn_kernels.fuse_ok(m, 16, 8)
    assert cnn_kernels.conv_fused(m, _x(16, 8), None, None, 0) is not None  # another size keeps the kernel
    assert cnn_kernels.conv_fused(m, _x(8, 8), None, None, 0) is None  # the rejected size is not retried
    assert fake_cutlass["ran"] == [(16, 8)]


def test_fs_cutlass_return_codes():
    kernels.reset_fallbacks()
    assert fs_cutlass._failed(1, "conv3d x", unsupported_ok=True) is False
    assert kernels.fallbacks() == {("fs_cutlass", "conv3d x"): 1}
    cases = [(1, "geometry not supported"), (-1, "unknown tile config"), (3, "CUDA launch error"), (7, "status 7")]
    for rc, match in cases:
        with pytest.raises(RuntimeError, match=f"fs_cutlass: conv x: {match}"):
            fs_cutlass._failed(rc, "conv x")
    with pytest.raises(RuntimeError, match="shared memory"):
        fs_cutlass._failed(2, "conv3d x", unsupported_ok=True)  # only "unsupported" may fall back
    kernels.reset_fallbacks()


@pytest.mark.gpu
@pytest.mark.parametrize("mag", [1.0, 40.0])
def test_attn_d384_large_qk_matches_fp32(mag):
    """q k^T rows above the fp16 range (|q k| up to ~1e5 at mag 40) stay finite and match an fp32 reference."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    dev = torch.device("cuda")
    s2m2._K[0] = s2m2_kernels.load(dev)
    g = torch.Generator(device=dev).manual_seed(0)
    n, ln, d = 2, 200, 384
    qkv = torch.randn((n, ln, 3 * d), device=dev, generator=g) * mag
    qkv[:, :, :d] = qkv[:, :, d : 2 * d] + 0.1 * qkv[:, :, :d]  # q ~ k: large diagonal scores
    qkv = qkv.half()
    q, k, v = (qkv[:, None, :, i * d : (i + 1) * d] for i in range(3))
    k_sc = torch.einsum("nhld,nhmd->nhlm", q.float(), k.float())
    assert (k_sc.abs().max() > 65504) == (mag > 1)  # mag 40: the overflow regime of an unscaled fp16 sum
    o = s2m2._attn384(q, k, v)
    assert o is not False
    ref = torch.softmax(k_sc * d**-0.5, -1) @ v.float()
    assert torch.isfinite(o).all()
    err = (o.float() - ref).abs().max().item() / ref.abs().max().item()
    assert err < 2e-2, err


@pytest.mark.parametrize("dev", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_instance_norm_normalises_each_sample(dev):
    """A batch of 2 (RT-MonSter++ stems: left and right) gets per-sample statistics (it used sample 0's for all)."""
    dt = torch.float32
    if dev == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        cnn_kernels.load(torch.device(dev))
        dt = torch.half
    g = torch.Generator().manual_seed(0)
    x = (
        torch.randn((2, 16, 12, 10), generator=g) * torch.tensor([1.0, 3.0]).view(2, 1, 1, 1)
        + torch.tensor([0, 2.0])[:, None, None, None]
    )
    ref = torch.nn.functional.instance_norm(x.to(dt).float(), eps=1e-5).relu()
    y = cnn_kernels.instance_norm(x.to(dev, dt).contiguous(memory_format=CL), 1e-5, cnn_kernels.ACT["relu"])
    assert (y.float().cpu() - ref).abs().max().item() < 1e-2
