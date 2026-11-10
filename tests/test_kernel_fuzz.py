"""Randomized kernel ops against plain PyTorch (GPU): seeded random shapes (odd sizes, sizes off the 8 / 16 / 32 tiles,
the smallest valid size, batch 1 and 2), layouts (channel slices, strided views) and value ranges (unit, near-zero,
large, negative offsets).

`FASTFORKS_FUZZ_SEEDS=n` runs n seeds per op (default 3: one per value range). The tail errors of every case (max and
p99.9 of |got - ref| relative to max |ref|) are recorded as test properties (`--junitxml`) and, with
`FASTFORKS_FUZZ_LOG=<file>`, appended there as JSON lines; the half-accumulate paths (CUTLASS fp16-accumulate convs,
cuDNN fp16-accumulate convs, flash64 `hh`) report them per magnitude.
"""

from __future__ import annotations

import json
import math
import os
import random

import pytest
import torch
import torch.nn.functional as F

from fastforks._core.half import cublas_gemms
from fastforks._optimize import conv_ops
from fastforks.kernels import cudnn_hacc, fastbb, seg_up
from fastforks.shared import cnn_kernels
from fastforks.stereo.models.las2 import kernels as las2_kernels

pytestmark = pytest.mark.gpu
CL = torch.channels_last
SEEDS = range(int(os.environ.get("FASTFORKS_FUZZ_SEEDS", "3")))
MAGS = (1.0, 1e-3, 1e3)  # value range per seed (seed % 3): unit, near-zero, large
FVM_ACT = {  # fvm act codes (`fastforks._optimize.ops`)
    0: lambda v: v,
    1: F.relu,
    2: F.relu6,
    3: F.silu,
    4: F.hardswish,
    5: F.gelu,
    6: lambda v: F.gelu(v, approximate="tanh"),
}
CNN_ACT = {"none": lambda v: v, "relu": F.relu, "relu6": F.relu6, "silu": F.silu, "gelu": F.gelu}


@pytest.fixture
def dev():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    return torch.device("cuda", torch.cuda.current_device())


def _rng(seed: int, name: str) -> random.Random:
    return random.Random(f"{name}:{seed}")  # noqa: S311 - test shapes


def _randn(g: torch.Generator, *shape: int, mag: float = 1.0, offset: float = 0.0) -> torch.Tensor:
    return torch.randn(shape, generator=g) * mag + offset * mag


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _cmp(  # noqa: PLR0913 - the pair, its tolerance and scale, and what to record
    record, name: str, got: torch.Tensor, ref: torch.Tensor, rel: float, *, pre=None, **info: object
) -> None:
    """|got - ref| <= rel * max |pre| + one fp16 subnormal step; records the tail errors.

    `pre`: the reference before a clamping activation (relu6 of 1e3-sized sums), else `ref` sets the scale.
    """
    got, ref = got.float().cpu(), ref.float().cpu()
    assert got.shape == ref.shape, (got.shape, ref.shape)
    assert torch.isfinite(got).all() == torch.isfinite(ref).all(), "non-finite values differ"
    scale = max((ref if pre is None else pre).abs().max().item(), 1e-30)
    diff = (got - ref).abs().flatten()
    err = diff / scale
    q = err.numel()
    p999 = err.sort().values[min(q - 1, math.ceil(0.999 * q) - 1)].item() if q else 0.0
    row = {"op": name, "max_rel": err.max().item() if q else 0.0, "p999_rel": p999, "scale": scale, **info}
    record(name, json.dumps(row))
    if log := os.environ.get("FASTFORKS_FUZZ_LOG"):
        with open(log, "a", encoding="utf-8") as f:  # noqa: PTH123
            f.write(json.dumps(row) + "\n")
    assert q == 0 or ((diff - 2**-24).clamp(min=0) / scale).max().item() <= rel, row


def _half_cl(t: torch.Tensor, dev: torch.device) -> torch.Tensor:
    return t.to(dev).half().contiguous(memory_format=CL)


# --- fvm_conv (generic graphs: arbitrary shapes from user models) -----------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_fvm_conv_fused(dev, seed, record_property):
    """act(conv + bias (+ residual)) on CUTLASS (fp16 accumulate), incl. the 8-channel stem and a channel slice."""
    rng, g, mag = _rng(seed, "conv_fused"), _gen(seed), MAGS[seed % 3]
    for case in range(6):
        n, c, k = rng.choice((1, 2)), 8 * rng.randint(1, 12), 8 * rng.randint(1, 10)
        if case == 0:
            c, k = 8, rng.choice((8, 16))  # stem_c8
        r = rng.choice((1, 3, 5, 7))
        st, pad = rng.choice((1, 2)), rng.choice((0, r // 2))
        h, w = rng.randint(max(1, r - 2 * pad), 37), rng.randint(max(1, r - 2 * pad), 41)
        act = rng.randrange(7)
        sliced = case == 1 and c > 8
        xb = _half_cl(_randn(g, n, c + (8 if sliced else 0), h, w, mag=mag, offset=rng.choice((0.0, -0.5))), dev)
        x = xb[:, 8:] if sliced else xb  # a channel slice: read in place
        wt = _half_cl(_randn(g, k, c, r, r) / math.sqrt(c * r * r), dev)
        b = _randn(g, k, mag=mag).to(dev).half() if rng.random() < 0.7 else None
        p, q = (h + 2 * pad - r) // st + 1, (w + 2 * pad - r) // st + 1
        res = _half_cl(_randn(g, n, k, p, q, mag=mag), dev) if c > 8 and rng.random() < 0.5 else None
        got = conv_ops.conv_fused(x, wt, b, res, [pad, pad], [st, st], act)
        ref = F.conv2d(x.float(), wt.float(), None if b is None else b.float(), st, pad)
        pre = ref + (0 if res is None else res.float())
        _cmp(
            record_property,
            "fvm conv_fused",
            got,
            FVM_ACT[act](pre),
            2e-2,
            pre=pre,
            mag=mag,
            shape=[n, c, h, w, k, r, st, pad, act],
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_fvm_conv_fused_into_slice(dev, seed, record_property):
    """`conv_fused_into` writes a channel slice of a wider buffer and leaves the other channels alone."""
    rng, g, mag = _rng(seed, "conv_into"), _gen(seed), MAGS[seed % 3]
    n, c, k, h, w = (
        rng.choice((1, 2)),
        8 * rng.randint(2, 6),
        8 * rng.randint(1, 6),
        rng.randint(1, 23),
        rng.randint(1, 29),
    )
    x = _half_cl(_randn(g, n, c, h, w, mag=mag), dev)
    wt = _half_cl(_randn(g, k, c, 3, 3) / math.sqrt(9 * c), dev)
    buf = _half_cl(_randn(g, n, k + 16, h, w), dev)
    keep = buf.clone()
    conv_ops.conv_fused_into(x, wt, None, None, [1, 1], [1, 1], 1, buf[:, 8 : 8 + k])
    ref = F.relu(F.conv2d(x.float(), wt.float(), None, 1, 1))
    _cmp(record_property, "fvm conv_fused_into", buf[:, 8 : 8 + k], ref, 2e-2, mag=mag, shape=[n, c, h, w, k])
    torch.testing.assert_close(buf[:, :8], keep[:, :8], rtol=0, atol=0)
    torch.testing.assert_close(buf[:, 8 + k :], keep[:, 8 + k :], rtol=0, atol=0)


def test_fvm_conv_fused_into_stride_contract(dev):
    """1x1 output slices (NCHW-like strides on size-1 dims) are accepted; a batch-strided out view is rejected."""
    g = _gen(0)
    x = _half_cl(_randn(g, 2, 16, 1, 1), dev)
    wt = _half_cl(_randn(g, 24, 16, 3, 3) / 12, dev)
    buf = _half_cl(_randn(g, 2, 40, 1, 1), dev)
    conv_ops.conv_fused_into(x, wt, None, None, [1, 1], [1, 1], 1, buf[:, 8:32])
    ref = F.relu(F.conv2d(x.float(), wt.float(), None, 1, 1))
    torch.testing.assert_close(buf[:, 8:32].float(), ref, rtol=2e-2, atol=2e-2)
    x = _half_cl(_randn(g, 2, 16, 5, 7), dev)
    wide = _half_cl(_randn(g, 4, 24, 5, 7), dev)
    with pytest.raises(ValueError, match="channel slice"):
        conv_ops.conv_fused_into(x, wt, None, None, [1, 1], [1, 1], 1, wide[::2])


@pytest.mark.parametrize("seed", SEEDS)
def test_fvm_pointwise_ops(dev, seed, record_property):
    """dwconv, max / avg pool (ceil mode, count_include_pad), affine_act on a slice, input_nhwc8."""
    rng, g, mag = _rng(seed, "fvm_pw"), _gen(seed), MAGS[seed % 3]
    for _ in range(3):
        n, c, h, w = rng.choice((1, 2)), 8 * rng.randint(1, 9), rng.randint(1, 33), rng.randint(1, 35)
        x = _half_cl(_randn(g, n, c, h, w, mag=mag, offset=rng.choice((0.0, -1.0))), dev)
        r = rng.choice((3, 5, 7)) if min(h, w) >= 3 else 1
        st, pad = rng.choice((1, 2)), r // 2
        wr = _randn(g, r, r, c) / r
        b = _randn(g, c, mag=mag)
        act = rng.randrange(7)
        got = conv_ops.dwconv(x, wr.to(dev).half().contiguous(), b.to(dev).half(), [st, st], [pad, pad], act)
        wq, bq = wr.half().float().permute(2, 0, 1)[:, None].to(dev), b.half().float().to(dev)
        pre = F.conv2d(x.float(), wq, bq, st, pad, groups=c)
        _cmp(
            record_property,
            "fvm dwconv",
            got,
            FVM_ACT[act](pre),
            5e-3,
            pre=pre,
            mag=mag,
            shape=[n, c, h, w, r, st, act],
        )

        kp = rng.choice((2, 3)) if min(h, w) >= 3 else 1
        sp, pp, ceil = rng.choice((1, 2)), rng.randint(0, kp // 2), rng.random() < 0.5
        got = conv_ops.maxpool(x, [kp, kp], [sp, sp], [pp, pp], ceil)
        _cmp(record_property, "fvm maxpool", got, F.max_pool2d(x.float(), kp, sp, pp, 1, ceil), 0.0, mag=mag)
        cip = rng.random() < 0.5
        got = conv_ops.avgpool(x, [kp, kp], [sp, sp], [pp, pp], cip)
        ref = F.avg_pool2d(x.float(), kp, sp, pp, False, cip)  # noqa: FBT003
        _cmp(record_property, "fvm avgpool", got, ref, 2e-3, mag=mag, shape=[n, c, h, w, kp, sp, pp, cip])

        xb = _half_cl(_randn(g, n, c + 8, h, w, mag=mag), dev)
        s, t = _randn(g, c).to(dev), _randn(g, c, mag=mag).to(dev)
        got = conv_ops.affine_act(xb[:, 8:], s, t, act)
        pre = xb[:, 8:].float() * s.view(1, -1, 1, 1) + t.view(1, -1, 1, 1)
        _cmp(record_property, "fvm affine_act", got, FVM_ACT[act](pre), 2e-3, pre=pre, mag=mag)

        ci = rng.randint(1, 8)
        xi = _randn(g, n, ci, h, w, mag=mag).to(dev)
        got = conv_ops.input_nhwc8(xi)
        ref = F.pad(xi.half().float(), (0, 0, 0, 0, 0, 8 - ci))
        _cmp(record_property, "fvm input_nhwc8", got, ref, 0.0, mag=mag)


@pytest.mark.parametrize("seed", SEEDS)
def test_fvm_lin(dev, seed, record_property):
    """Token GEMM `act(x @ w^T + b (+ res))` on the fused 1x1 conv (fp16 accumulate)."""
    rng, g, mag = _rng(seed, "lin"), _gen(seed), MAGS[seed % 3]
    for _ in range(3):
        b_, t, k, nout = rng.choice((1, 2)), rng.randint(1, 300), 8 * rng.randint(1, 64), 8 * rng.randint(1, 48)
        x = _randn(g, b_, t, k, mag=mag).to(dev).half()
        w4 = _half_cl(_randn(g, nout, k, 1, 1) / math.sqrt(k), dev)
        bias = _randn(g, nout, mag=mag).to(dev).half()
        res = _randn(g, b_, t, nout, mag=mag).to(dev).half() if rng.random() < 0.5 else None
        act = rng.choice((0, 1, 5))
        got = conv_ops.lin(x, w4, bias, res, act)
        ref = F.linear(x.float(), w4.float().view(nout, k), bias.float()) + (0 if res is None else res.float())
        _cmp(record_property, "fvm lin", got, FVM_ACT[act](ref), 2e-2, pre=ref, mag=mag, shape=[b_, t, k, nout, act])
        with cublas_gemms():  # every GEMM through cuBLAS (fp32 accumulation): only the fp16 output rounding
            got = conv_ops.lin(x, w4, bias, res, act)
        _cmp(record_property, "lin cublas", got, FVM_ACT[act](ref), 2e-3, pre=ref, mag=mag, shape=[b_, t, k, nout, act])


# --- fastbb ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_fastbb_ln_dwconv(dev, seed, record_property):
    """ln_lastdim (all four vector widths, bias-free), dwconv3_nhwc, dwgate (all three vector widths)."""
    rng, g, mag = _rng(seed, "fastbb_ln"), _gen(seed), MAGS[seed % 3]
    for i in range(4):
        rows, c = rng.randint(1, 700), 8 * rng.choice((rng.randint(1, 4), rng.randint(5, 32), rng.randint(33, 256)))
        x = _randn(g, rows, c, mag=mag, offset=rng.choice((0.0, 3.0))).to(dev).half()
        w, b = _randn(g, c).to(dev), _randn(g, c).to(dev)
        bias_free = i % 2 == 1
        got = fastbb.ln_lastdim(x, w, None if bias_free else b, 1e-5, bias_free=bias_free)
        xf = x.float()
        if bias_free:  # Restormer BiasFree: x * rstd * w (variance around the mean, x not centred)
            ref = xf * torch.rsqrt(xf.var(-1, unbiased=False, keepdim=True) + 1e-5) * w
        else:
            ref = F.layer_norm(xf, (c,), w, b, 1e-5)
        _cmp(record_property, "fastbb ln_lastdim", got, ref, 5e-3, mag=mag, shape=[rows, c, bias_free])

    n, h, w_ = rng.choice((1, 2)), rng.randint(1, 19), rng.randint(1, 23)
    c = 8 * rng.randint(1, 12)
    x = _randn(g, n, h, w_, c, mag=mag).to(dev).half()
    wt, b = (_randn(g, 3, 3, c) / 3).to(dev).half(), _randn(g, c, mag=mag).to(dev).half()
    got = fastbb.dwconv3(x, wt, b)
    ref = F.conv2d(x.float().permute(0, 3, 1, 2), wt.float().permute(2, 0, 1)[:, None], b.float(), 1, 1, groups=c)
    _cmp(record_property, "fastbb dwconv3", got, ref.permute(0, 2, 3, 1), 5e-3, mag=mag, shape=[n, h, w_, c])

    gm = min(mag, 30.0)  # the gate multiplies two convs: 1e3 inputs overflow fp16 (as in PyTorch)
    for hh in (rng.randint(1, 20) * 4, rng.randint(1, 20) * 4 + 2, rng.randint(0, 20) * 2 + 1):  # v4, v2, v1
        x = _randn(g, n, h, w_, 2 * hh, mag=gm).to(dev).half()
        wt, b = (_randn(g, 9, 2 * hh) / 3).to(dev), _randn(g, 2 * hh, mag=gm).to(dev)
        got = fastbb.dwgate(x, wt, b)
        d = F.conv2d(x.float().permute(0, 3, 1, 2), wt.t().reshape(2 * hh, 1, 3, 3), b, 1, 1, groups=2 * hh)
        ref = (F.gelu(d[:, :hh]) * d[:, hh:]).permute(0, 2, 3, 1)
        _cmp(record_property, "fastbb dwgate", got, ref, 5e-3, mag=gm, shape=[n, h, w_, hh])


def _flash(k, qkv: torch.Tensor, rm: int, variant: str) -> torch.Tensor:
    q, kk, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # strided views (token stride 3 * heads * 64)
    b, n, h, d = q.shape
    o = torch.empty(b, n, h, d, device=q.device, dtype=q.dtype)
    fastbb.check_flash64(q, kk, v, o)
    args: list = [q, kk, v, o, n, h]
    for t in (q, kk, v, o):
        args += [t.stride(0), t.stride(1), t.stride(2)]
    log2e = 1.4426950408889634
    k.launch(
        f"flash64_r{rm}_{variant}", ((n + 64 * rm - 1) // (64 * rm), b * h, 1), (128, 1, 1), *args, d**-0.5 * log2e
    )
    return o


@pytest.mark.parametrize("variant", ["hh", "fh", "ff"])
@pytest.mark.parametrize("seed", SEEDS)
def test_fastbb_flash64(dev, seed, variant, record_property):
    """flash64 attention on strided `[B, N, heads, 64]` views of a packed qkv, token counts off the 64 / 128 tiles."""
    rng, g = _rng(seed, f"flash_{variant}"), _gen(seed)
    mag = (1.0, 0.05, 4.0)[seed % 3]  # q k^T logits ~ mag^2 * 8: up to ~130 before the scale (softmax saturates)
    k = fastbb.load(dev)
    for rm in (1, 2):
        b, n, h = rng.choice((1, 2)), rng.choice((1, rng.randint(2, 63), rng.randint(65, 400))), rng.randint(1, 6)
        qkv = _randn(g, b, n, 3, h, 64, mag=mag).to(dev).half()
        got = _flash(k, qkv, rm, variant)
        ref = F.scaled_dot_product_attention(*qkv.float().permute(2, 0, 3, 1, 4)).transpose(1, 2)
        _cmp(record_property, f"fastbb flash64_{variant}", got, ref, 2e-2, mag=mag, shape=[b, n, h, rm])


# --- stereo_ops, seg_up, cnn_kernels (igev) -------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_stereo_ops(dev, seed, record_property):
    """corr_volume, soft_argmax (top-k), dw_line with random sizes."""
    from fastforks.stereo.models.las2.net.submodule import build_correlation_volume  # noqa: PLC0415

    rng, g, mag = _rng(seed, "stereo_ops"), _gen(seed), MAGS[seed % 3]
    so = las2_kernels.stereo_ops
    k = so.load(dev)
    n, c, h, w = rng.choice((1, 2)), rng.randint(1, 80), rng.randint(1, 9), rng.randint(1, 90)
    d = rng.randint(1, 70)
    fl, fr = (_half_cl(_randn(g, n, c, h, w, mag=math.sqrt(mag)), dev) for _ in range(2))
    got = las2_kernels.corr_volume(k, fl, fr, d)
    _cmp(record_property, "stereo_ops corr_volume", got, build_correlation_volume(fl.float(), fr.float(), d), 2e-3)

    topk = rng.choice((0, 0, 2, 3))
    cost = _randn(g, n, 2 * d, h, w, mag=min(mag, 8.0)).to(dev).half()[:, ::2]  # a strided view
    got = so.soft_argmax(k, cost, topk)
    if topk:  # ties: the kernel takes the lower d first, torch.topk any; checked for range only
        assert ((got >= 0) & (got <= d - 1)).all()
    else:
        disp = torch.arange(d, device=dev, dtype=torch.float32).view(1, d, 1, 1)
        ref = (F.softmax(cost.float(), 1) * disp).sum(1, keepdim=True)
        _cmp(record_property, "stereo_ops soft_argmax", got, ref, 1e-4, mag=mag, shape=[n, d, h, w])

    kk, axis = 2 * rng.randint(0, 10) + 1, rng.randint(0, 1)
    x = _half_cl(_randn(g, n, c, h, w, mag=mag), dev)
    res = _half_cl(_randn(g, n, c, h, w, mag=mag), dev)
    wt, b = _randn(g, c, kk).to(dev) / math.sqrt(kk), _randn(g, c, mag=mag).to(dev)
    shape = (c, 1, 1, kk) if axis == 0 else (c, 1, kk, 1)
    pad = (0, kk // 2) if axis == 0 else (kk // 2, 0)
    ref = F.conv2d(x.float(), wt.view(shape), b, padding=pad, groups=c) + res.float()
    got = so.dw_line(k, x, wt.t().contiguous(), b, axis, res=res)
    _cmp(record_property, "stereo_ops dw_line", got, ref, 5e-3, mag=mag, shape=[n, c, h, w, kk, axis])


@pytest.mark.parametrize("seed", SEEDS)
def test_seg_up_add(dev, seed, record_property):
    """`up_add`: base + bilinear resize of z (one or two steps), 2 or 8 channels per thread, relu."""
    rng, g, mag = _rng(seed, "seg_up"), _gen(seed), MAGS[seed % 3]
    for _ in range(4):
        b, c = rng.choice((1, 2)), 2 * rng.randint(1, 40)
        h, w = rng.randint(1, 17), rng.randint(1, 19)
        mid = (h * rng.randint(1, 3), w * rng.randint(1, 3))
        hh, ww = mid if rng.random() < 0.4 else (mid[0] * 2 + rng.randint(0, 3), mid[1] * 2 + rng.randint(0, 3))
        z = _randn(g, b, h, w, c, mag=mag).to(dev).half()
        base = _randn(g, b, hh, ww, c, mag=mag).to(dev).half()
        out = torch.empty_like(base)
        relu = rng.random() < 0.5
        seg_up.up_add(z, base, out, mid, relu=relu)
        zc = z.float().permute(0, 3, 1, 2)
        up = F.interpolate(zc, size=mid, mode="bilinear", align_corners=False)
        if tuple(mid) != (hh, ww):
            up = F.interpolate(up, size=(hh, ww), mode="bilinear", align_corners=False)  # composed in fp32
        ref = base.float() + up.permute(0, 2, 3, 1)
        _cmp(
            record_property,
            "seg_up up_add",
            out,
            F.relu(ref) if relu else ref,
            5e-3,
            mag=mag,
            shape=[b, c, h, w, *mid, hh, ww],
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_cnn_epilogue_instance_norm(dev, seed, record_property):
    """`epilogue` (8-channel and scalar kernels, residual, extended activations) and `instance_norm` (batch 1 and 2)."""
    rng, g, mag = _rng(seed, "cnn_epi"), _gen(seed), MAGS[seed % 3]
    cnn_kernels.load(dev)
    for _ in range(3):
        n, c, h, w = (
            rng.choice((1, 2)),
            rng.choice((rng.randint(1, 7), 8 * rng.randint(1, 16))),
            rng.randint(1, 31),
            rng.randint(1, 33),
        )
        act = rng.choice(list(CNN_ACT))
        y = _half_cl(_randn(g, n, c, h, w, mag=mag), dev)
        s, t = _randn(g, c).to(dev), _randn(g, c, mag=mag).to(dev)
        r = _half_cl(_randn(g, n, c, h, w, mag=mag), dev) if rng.random() < 0.5 else None
        ref = y.float() * s.view(1, -1, 1, 1) + t.view(1, -1, 1, 1) + (0 if r is None else r.float())
        got = cnn_kernels.epilogue(y, s, t, cnn_kernels.ACT[act], r)
        _cmp(record_property, "igev epilogue", got, CNN_ACT[act](ref), 2e-3, pre=ref, mag=mag, shape=[n, c, h, w, act])

        if h * w > 1:
            x = _half_cl(_randn(g, n, c, h, w, mag=mag, offset=rng.choice((0.0, 2.0))), dev)
            ref = F.instance_norm(x.float(), eps=1e-5).relu()
            got = cnn_kernels.instance_norm(x, 1e-5, cnn_kernels.ACT["relu"])
            _cmp(record_property, "igev instance_norm", got, ref, 5e-3, mag=mag, shape=[n, c, h, w])


# --- cuDNN fp16-accumulate convs (cudnn_hacc) ------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_cudnn_hacc_conv(dev, seed, record_property):
    """Random 2-D / 3-D conv geometries on the fp16-accumulate cuDNN plans against an fp32 conv."""
    rng, g, mag = _rng(seed, "hacc"), _gen(seed), MAGS[seed % 3]
    cudnn_hacc.load(dev)
    conv = cudnn_hacc.CudnnHalfConv()
    for i in range(4):
        three_d = i == 3
        n, c, k = rng.choice((1, 2)), 8 * rng.randint(1, 12), 8 * rng.randint(1, 12)
        r = rng.choice((1, 3)) if three_d else rng.choice((1, 3, 5))
        st, pad = rng.choice((1, 2)), r // 2
        sp = [rng.randint(r, 13) for _ in range(3 if three_d else 2)]
        mf = torch.channels_last_3d if three_d else CL
        x = _randn(g, n, c, *sp, mag=mag).to(dev).half().contiguous(memory_format=mf)
        wt = (_randn(g, k, c, *[r] * len(sp)) / math.sqrt(c * r ** len(sp))).to(dev).half().contiguous(memory_format=mf)
        b = _randn(g, k, mag=mag).to(dev).half()
        got = conv.conv(x, wt, b, [pad] * len(sp), [st] * len(sp))
        fn = F.conv3d if three_d else F.conv2d
        ref = fn(x.float(), wt.float(), b.float(), st, pad)
        _cmp(record_property, "cudnn_hacc conv", got, ref, 2e-2, mag=mag, shape=[n, c, *sp, k, r, st])


# --- RAFT (raft_flow): the lookup without a correlation volume, the coordinate grid, both upsamplings ----------------
# The other raft_flow kernels (InstanceNorm, conv epilogues, the GRU gates, the flow convs, SEA-RAFT's depthwise conv)
# take the engines' packed weights; tests/test_flow_raft.py and tests/test_flow_sea_raft.py check them through the
# engines against the portable fp32 runners.


@pytest.mark.parametrize("large", [True, False])
@pytest.mark.parametrize("seed", SEEDS)
def test_raft_flow(dev, seed, large, record_property):
    """avgpool2 + corr_lookup against the all-pairs pyramid of `raft/net.py`, init_coords, convex and bilinear 8x."""
    from fastforks._core.kernels import linear  # noqa: PLC0415
    from fastforks.flow.models.raft import net  # noqa: PLC0415
    from fastforks.kernels import raft_flow  # noqa: PLC0415

    rng, g, mag = _rng(seed, f"raft_flow{large}"), _gen(seed), MAGS[seed % 3]
    k = raft_flow.load(dev)
    c, radius = (256, 4) if large else (128, 3)
    h, w = rng.randint(16, 40), rng.randint(16, 40)  # 16: the engines' smallest grid (2 x 2 cells at level 3)
    p = h * w
    f1, f2 = (_randn(g, h, w, c, mag=math.sqrt(mag)).to(dev).half() for _ in range(2))
    pyr = [f2]
    for _ in range(3):
        src = pyr[-1]
        hd, wd = src.shape[0] // 2, src.shape[1] // 2
        dst = torch.empty(hd, wd, c, device=dev, dtype=torch.half)
        k.launch("avgpool2", *linear(hd * wd * c // 8), src, src.shape[0], src.shape[1], dst, hd, wd, c)
        pyr.append(dst)

    coords = torch.full((p, 2), float("nan"), device=dev)
    k.launch("init_coords", *linear(p), coords, w, p)
    grid = net._coords(1, h, w, dev)

    def nchw(t: torch.Tensor) -> torch.Tensor:  # the kernels' [h * w, C] -> [1, C, h, w]
        return t.view(h, w, -1).permute(2, 0, 1)[None]

    assert torch.equal(nchw(coords), grid)
    move = rng.choice((0.7, 5.0, 40.0))  # sub-cell, inside the window of the fine levels, over the border
    coords += _randn(g, p, 2, mag=move).to(dev)

    n = 4 * (2 * radius + 1) ** 2
    stride = (n + 7) // 8 * 8 + 8 * rng.randint(0, 1)
    look = torch.full((h, w, stride), float("nan"), device=dev, dtype=torch.half)
    name = "corr_lookup_c256_r4" if large else "corr_lookup_c128_r3"
    k.launch(name, ((p + 7) // 8, 1, 1), (256, 1, 1), f1, *pyr, h, w, coords, look, stride, 1.0 / math.sqrt(c))
    f1n, f2n = (t.permute(2, 0, 1)[None].float() for t in (f1, f2))
    ref = net._lookup(net._pyramid(f1n, f2n, 4), nchw(coords), radius)
    info = {"mag": mag, "shape": [h, w, c], "move": move}
    _cmp(record_property, f"raft_flow {name}", look[..., :n].permute(2, 0, 1)[None], ref, 6.4e-4, **info)
    assert not look[..., n:].any()  # the channels up to the stride are zero padding

    flow, hh, ww = nchw(coords) - grid, 8 * h, 8 * w
    out = torch.full((2, hh, ww), float("nan"), device=dev)
    k.launch("upsample_bilinear", *linear(hh * ww), coords, h, w, out)
    _cmp(record_property, "raft_flow upsample_bilinear", out[None], net._upsample(flow, None), 1e-6, **info)
    mask = _randn(g, p, 576, mag=min(mag, 30.0)).to(dev).half()  # the raw conv output, before the 0.25
    out = torch.full((2, hh, ww), float("nan"), device=dev)
    k.launch("upsample_convex", *linear(hh * ww), coords, mask, h, w, out)
    _cmp(
        record_property,
        "raft_flow upsample_convex",
        out[None],
        net._upsample(flow, 0.25 * nchw(mask).float()),
        4.7e-7,
        **info,
    )
