"""`seg_conv` (the mmseg kernels' CUTLASS conv): `act(conv(x) * s + t + r)` against the fp32 PyTorch conv."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fastforks._core import kernels

CL = torch.channels_last


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("c", "k", "ksize", "stride", "dil", "hw"),
    [
        (64, 64, 3, 1, 1, (37, 45)),  # narrow 3x3
        (128, 256, 1, 1, 1, (33, 64)),  # 1x1 expansion
        (256, 512, 3, 2, 1, (31, 40)),  # strided
        (256, 256, 3, 1, 2, (29, 50)),  # D8 dilated
        (512, 512, 3, 1, 4, (24, 33)),
        (48, 96, 3, 2, 1, (64, 96)),  # HRNet widths
        (2048, 512, 3, 1, 12, (16, 24)),  # ASPP rate
    ],
)
@pytest.mark.parametrize("mode", ["affine-relu", "residual", "bias"])
def test_seg_conv(c: int, k: int, ksize: int, stride: int, dil: int, hw: tuple[int, int], mode: str) -> None:  # noqa: PLR0913, PLR0917 - parametrized case
    from fastforks.kernels import seg_conv  # noqa: PLC0415 - needs the GPU library

    dev = torch.device("cuda")
    kernels.reset_fallbacks()
    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(1, c, *hw, device=dev, generator=g).half().contiguous(memory_format=CL)
    w = (torch.randn(k, c, ksize, ksize, device=dev, generator=g) / (c * ksize * ksize) ** 0.5).half()
    w = w.contiguous(memory_format=CL)
    pad = dil * (ksize // 2)
    ref = F.conv2d(x.float(), w.float(), None, stride, pad, dil)
    s = t = r = None
    act = 1
    if mode != "bias":
        s = torch.rand(k, device=dev, generator=g) + 0.5
        ref = ref * s.view(1, -1, 1, 1)
    t = torch.randn(k, device=dev, generator=g)
    ref = ref + t.view(1, -1, 1, 1)
    if mode == "residual":
        r = torch.randn(ref.shape, device=dev, generator=g).half().contiguous(memory_format=CL)
        ref = ref + r.float()
    if mode == "bias":
        act = 0
    else:
        ref = ref.relu()
    y = torch.empty(ref.shape, device=dev, dtype=torch.float16, memory_format=CL)
    lib = seg_conv.load(dev)
    assert lib.conv(x, w, y, s, t, r, padding=(pad, pad), stride=(stride, stride), dilation=(dil, dil), act=act)
    assert not kernels.fallbacks()
    err = (y.float() - ref).abs()
    # fp16 accumulation: |error| grows with the magnitude and as a random walk over the c * ksize² products
    tol = 2e-2 * (1 + ref.abs()) + 5e-4 * (c * ksize * ksize) ** 0.5
    assert bool((err <= tol).all()), (err.max().item(), err.mean().item())
    assert err.mean().item() < 2e-3, err.mean().item()
    assert err.mean().item() < 4e-3


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("c", "zhw", "scale", "yhw"),
    [(48, (16, 32), 2, (32, 64)), (96, (13, 17), 2, (25, 33)), (48, (7, 9), 8, (50, 70)), (18, (13, 17), 2, (25, 33))],
)
@pytest.mark.parametrize("relu", [False, True])
def test_hrnet_up_add(c: int, zhw: tuple[int, int], scale: int, yhw: tuple[int, int], *, relu: bool) -> None:
    """`seg_up.up_add` (8 or 2 channels per thread) = fp16 PyTorch `relu?(y + resize(resize(z, z * scale), y size))`."""
    from fastforks.segmentation.models.mmseg import model as mmm  # noqa: PLC0415 - needs the GPU library
    from fastforks.segmentation.models.mmseg import net  # noqa: PLC0415

    dev = torch.device("cuda")
    kernels.reset_fallbacks()
    g = torch.Generator(device=dev).manual_seed(0)
    y = torch.randn(1, c, *yhw, device=dev, generator=g).half().contiguous(memory_format=CL)
    z = torch.randn(1, c, *zhw, device=dev, generator=g).half().contiguous(memory_format=CL)
    mid = [t * scale for t in zhw]
    ref = y + net.resize(net.resize(z, mid), yhw)
    ref = ref.relu() if relu else ref
    out = mmm._up_add(y, z, mid, relu=relu, inplace=False)
    assert not kernels.fallbacks()
    err = (out.float() - ref.float()).abs()
    assert bool((err <= 2e-3 * (1 + ref.float().abs())).all()), err.max().item()


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("c", "ksize", "stride", "dil", "hw"),
    [(2048, 3, 1, 12, (33, 65)), (304, 3, 1, 1, (64, 96)), (64, 5, 2, 2, (31, 40))],
)
def test_seg_dwconv(c: int, ksize: int, stride: int, dil: int, hw: tuple[int, int]) -> None:
    """`seg_conv.dwconv` (depthwise, BN affine + ReLU epilogue) against the fp32 PyTorch conv."""
    from fastforks.kernels import seg_conv  # noqa: PLC0415 - needs the GPU library

    dev = torch.device("cuda")
    kernels.reset_fallbacks()
    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(1, c, *hw, device=dev, generator=g).half().contiguous(memory_format=CL)
    w = (torch.randn(c, 1, ksize, ksize, device=dev, generator=g) / ksize).half()
    s = torch.rand(c, device=dev, generator=g) + 0.5
    t = torch.randn(c, device=dev, generator=g)
    pad = dil * (ksize // 2)
    ref = F.conv2d(x.float(), w.float(), None, stride, pad, dil, c) * s.view(1, -1, 1, 1) + t.view(1, -1, 1, 1)
    ref = ref.relu()
    y = torch.empty(ref.shape, device=dev, dtype=torch.float16, memory_format=CL)
    w_rsc = w[:, 0].permute(1, 2, 0).contiguous()
    geo = {"padding": (pad, pad), "stride": (stride, stride), "dilation": (dil, dil), "act": 1}
    assert seg_conv.load(dev).dwconv(x, w_rsc, y, s, t, **geo)
    assert not kernels.fallbacks()
    err = (y.float() - ref).abs()
    assert bool((err <= 2e-3 * (1 + ref.abs())).all()), err.max().item()
