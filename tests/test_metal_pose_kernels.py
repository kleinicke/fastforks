"""The real Metal kernels of `pose_metal` and `vit_metal` on Apple MPS against their PyTorch forms (no weights)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fastforks._core.errors import KernelUnavailableError
from fastforks._optimize import transformer
from fastforks.kernels import pose_metal, vit_metal
from fastforks.pose.models.rtmpose import model as rtmpose
from fastforks.pose.models.vitpose import model as vitpose

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS"),
]

MPS = torch.device("mps")
SEEDS = range(4)  # fixed: the seeds the bounds below were measured on (Apple M1 Pro, macOS 27, torch 2.14)
# Bounds, fixed: 1.5 x the worst of `SEEDS` measured on that machine, rounded up to 2 digits.
# deconv4s2_relu against float64 conv_transpose2d + ReLU of the same (dtype-rounded) operands, max error / max value
DECONV_REL = {torch.float32: 1.8e-7, torch.float16: 6.4e-4}  # fixed: measured 1.2e-7, 4.3e-4 (fp16: the fp16 GEMM)
# dark_decode against the PyTorch decode on the CPU, heatmap pixels
DARK_PX_PEAKS = 5.8e-6  # fixed: measured 3.9e-6 (Gaussian peaks + noise, as a network's heatmaps)
DARK_PX_NOISE = 7.0e-4  # fixed: measured 4.7e-4 (white noise: a badly conditioned Newton step)
# pe_rope against `_pe_rope` on the CPU, max error / max value
ROPE_REL = {torch.float32: 9.8e-8, torch.float16: 2.0e-4}  # fixed: measured 6.5e-8, 1.3e-4 (fp16: one ulp)
DARK = {"sigma": vitpose.SIGMA, "radius": vitpose.RADIUS, "clip": vitpose.CLIP}
NET = (256, 192)  # fixed: the pose networks' input (height, width)


@pytest.fixture(scope="module")
def lib():
    return pose_metal.load(MPS)


def _same(got: torch.Tensor, want: torch.Tensor) -> bool:
    """Equal values, NaN at the same places."""
    return (
        got.shape == want.shape
        and torch.equal(got.isnan(), want.isnan())
        and torch.equal(got.nan_to_num(nan=0.0), want.nan_to_num(nan=0.0))
    )


# ---------------------------------------------------------------------------------------------------- warp_crop


def _coords(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(centre, size, x, y) of `boxes` as `rtmpose.MetalKernels.__call__` computes them."""
    h, w = NET
    centre, size = rtmpose.crop_boxes(boxes.float(), h, w)
    s = size[:, 0] / w
    return (
        centre,
        size,
        rtmpose.fixed(centre[:, 0] - s * (w / 2), s, w),
        rtmpose.fixed(centre[:, 1] - s * (h / 2), s, h),
    )


def _crop_pair(lib, image: torch.Tensor, boxes: list[list[float]]) -> tuple[torch.Tensor, torch.Tensor]:
    """(Metal crop, PyTorch crop on the CPU) of a CPU `image` `[1, 3, H, W]` and `x0, y0, x1, y1` boxes."""
    centre, size, x, y = _coords(torch.tensor(boxes))
    want = rtmpose.crop(image, centre, size, *NET)
    got = pose_metal.warp_crop(lib, image.to(MPS), x.to(MPS), y.to(MPS))
    return got.cpu(), want


BOXES = {
    "inside": [[20.0, 30.0, 90.0, 170.0], [5.5, 7.25, 60.75, 99.5]],
    "partly outside": [[-40.0, -25.0, 60.0, 80.0], [70.0, 120.0, 400.0, 500.0], [-30.0, 50.0, 200.0, 90.0]],
    "fully outside": [[-500.0, -500.0, -300.0, -200.0], [900.0, 900.0, 1000.0, 1100.0]],
    "zero size": [[40.0, 50.0, 40.0, 50.0], [0.0, 0.0, 0.0, 0.0]],
    "sub-pixel": [[40.2, 50.7, 40.9, 51.3]],
    "whole image and more": [[0.0, 0.0, 128.0, 200.0], [-1000.0, -1000.0, 2000.0, 2000.0]],
    "far but finite": [[0.0, 0.0, 1e30, 1e30], [-1e30, -1e30, 0.0, 0.0], [3e9, 3e9, 3.1e9, 3.1e9]],
}


@pytest.mark.parametrize("name", sorted(BOXES))
def test_warp_crop_is_the_pytorch_crop(lib, name):
    """Bit-exact with `rtmpose.crop` for boxes inside, across and outside the image."""
    torch.manual_seed(0)
    image = torch.rand(1, 3, 200, 128)
    image[0, :, :40] = torch.randint(0, 256, (3, 40, 128)) / 255  # uint8 images: exact levels
    got, want = _crop_pair(lib, image, BOXES[name])
    assert torch.equal(got, want)
    if name == "inside":
        assert got.std() > 10  # a real crop, not zeros


def test_warp_crop_overflowing_box_gives_nan_without_reading(lib):
    """A box whose size overflows float32 gives NaN coordinates: NaN crops as PyTorch, no image access (a NaN
    passed both range checks and indexed the image before)."""
    image = torch.rand(1, 3, 40, 50)
    boxes = [
        [0.0, 0.0, 1e37, 1e37],
        [float("inf"), 0.0, 1.0, 1.0],
        [0.0, float("nan"), 9.0, 9.0],
        [5.0, 5.0, 30.0, 30.0],
    ]
    _, _, x, y = _coords(torch.tensor(boxes))
    assert not x[:3].isfinite().any()
    got, want = _crop_pair(lib, image, boxes)
    assert got[:3].isnan().all()
    assert got[3].isfinite().all()
    assert _same(got, want)
    # NaN in single coordinates: only their rows / columns
    x, y = x[3:].clone(), y[3:].clone()
    x[0, 7], y[0, 11], x[0, 100] = float("nan"), float("inf"), float("-inf")
    got = pose_metal.warp_crop(lib, image.to(MPS), x.to(MPS), y.to(MPS)).cpu()
    bad = torch.zeros(NET, dtype=torch.bool)
    bad[11], bad[:, 7], bad[:, 100] = True, True, True
    assert torch.equal(got.isnan(), bad.expand(1, 3, -1, -1))


def test_warp_crop_strided_images(lib):
    """Sliced, stepped, flipped and offset images are read as the tensors they are."""
    torch.manual_seed(1)
    big = torch.rand(2, 5, 230, 150)
    boxes = BOXES["inside"] + BOXES["partly outside"]
    centre, size, x, y = _coords(torch.tensor(boxes))
    views = {
        "offset batch": lambda t: t[1:2, :3],
        "cropped": lambda t: t[:1, 1:4, 10:210, 7:135],
        "stepped": lambda t: t[:1, ::2, ::2, 1::2],
        "flipped": lambda t: t[:1, :3].flip(1).flip(-1),
        "permuted": lambda t: t[:1, :3].transpose(2, 3),
        "expanded": lambda t: t[:1, :1].expand(1, 3, -1, -1),
    }
    for name, view in views.items():
        want = rtmpose.crop(view(big).contiguous(), centre, size, *NET)
        got = pose_metal.warp_crop(lib, view(big.to(MPS)), x.to(MPS), y.to(MPS)).cpu()
        assert torch.equal(got, want), name
    # strided coordinates and a float16 image
    x2, y2 = x.to(MPS).repeat_interleave(2, 1)[:, ::2], y.to(MPS).flip(0).flip(0)
    half = big[:1, :3].half()
    got = pose_metal.warp_crop(lib, half.to(MPS), x2, y2).cpu()
    assert torch.equal(got, rtmpose.crop(half.float(), centre, size, *NET))


def test_warp_crop_image_above_2_24_pixels(lib):
    """Pixel offsets above 2^24 (inexact as float32 indices) are read exactly."""
    torch.manual_seed(2)
    side = 4200  # fixed: 4200^2 > 2^24 pixels
    image = torch.rand(1, 3, side, side)
    boxes = [[4000.0, 4000.0, 4190.0, 4199.0], [3000.5, 4100.0, 4300.0, 4250.0], [0.0, 0.0, 4200.0, 4200.0]]
    got, want = _crop_pair(lib, image, boxes)
    assert torch.equal(got, want)
    assert got[0].std() > 10  # a real crop


def test_warp_crop_rejects_bad_shapes_and_no_people(lib):
    image = torch.rand(1, 3, 20, 30, device=MPS)
    out = pose_metal.warp_crop(lib, image, torch.zeros(0, 192, device=MPS), torch.zeros(0, 256, device=MPS))
    assert out.shape == (0, 3, 256, 192)
    with pytest.raises(ValueError, match="warp_crop"):
        pose_metal.warp_crop(
            lib, image.expand(2, -1, -1, -1), torch.zeros(1, 4, device=MPS), torch.zeros(1, 5, device=MPS)
        )
    with pytest.raises(ValueError, match="warp_crop"):
        pose_metal.warp_crop(lib, image, torch.zeros(2, 4, device=MPS), torch.zeros(1, 5, device=MPS))


# ----------------------------------------------------------------------------------------------- deconv4s2_relu


def _deconv_error(
    lib, dtype: torch.dtype, seed: int, shape: tuple[int, int, int, int, int] = (3, 24, 16, 5, 7)
) -> float:
    """max |kernel - float64 conv_transpose2d + ReLU| / max |reference| for `shape` = (batch, in, out, h, w)."""
    torch.manual_seed(seed)
    b, cin, cout, h, w = shape
    x = torch.randn(b, cin, h, w).to(dtype)
    wt = (torch.randn(cin, cout, 4, 4) / (4 * cin**0.5)).to(dtype)
    bias = torch.randn(cout).to(dtype)
    want = F.relu(F.conv_transpose2d(x.double(), wt.double(), bias.double(), stride=2, padding=1))
    tokens = x.flatten(2).transpose(1, 2).to(MPS)  # [B, h * w, in]
    cols = tokens @ wt.permute(0, 2, 3, 1).reshape(cin, -1).to(MPS)  # as `vitpose.MetalKernels`
    got = pose_metal.deconv4s2_relu(lib, cols, bias.to(MPS), h, w)
    assert got.dtype == dtype
    assert got.shape == (b, 4 * h * w, cout)
    got = got.view(b, 2 * h, 2 * w, cout).permute(0, 3, 1, 2).cpu().double()
    assert (want == 0).any()  # the ReLU acts
    assert (want > 0).any()
    return float((got - want).abs().max() / want.abs().max())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=str)
def test_deconv4s2_relu_is_conv_transpose2d(lib, dtype):
    for seed in SEEDS:
        assert _deconv_error(lib, dtype, seed) <= DECONV_REL[dtype]
    for shape in ((1, 8, 4, 1, 1), (2, 8, 4, 1, 6), (1, 8, 4, 6, 1)):  # one row / column: every tap at a border
        assert _deconv_error(lib, dtype, 0, shape) <= DECONV_REL[dtype]


def test_deconv4s2_relu_strided_operands_and_no_people(lib):
    torch.manual_seed(0)
    cols = torch.randn(2, 6, 32, device=MPS)
    bias = torch.randn(2, device=MPS)
    want = pose_metal.deconv4s2_relu(lib, cols, bias, 2, 3)
    strided = pose_metal.deconv4s2_relu(
        lib, cols.transpose(1, 2).contiguous().transpose(1, 2), bias.repeat_interleave(2)[::2], 2, 3
    )
    assert torch.equal(strided, want)
    assert torch.equal(
        pose_metal.deconv4s2_relu(lib, cols.half(), bias, 2, 3),
        pose_metal.deconv4s2_relu(lib, cols.half(), bias.half(), 2, 3),
    )
    assert pose_metal.deconv4s2_relu(lib, cols[:0], bias, 2, 3).shape == (0, 24, 2)
    with pytest.raises(ValueError, match="deconv4s2_relu"):
        pose_metal.deconv4s2_relu(lib, cols, bias, 3, 3)


# -------------------------------------------------------------------------------------------------- dark_decode


def _peaks(seed: int, shape: tuple[int, int, int, int] = (3, 17, 64, 48)) -> torch.Tensor:
    """Heatmaps `[P, K, h, w]` as a pose network's: one Gaussian peak each (some at the border) + noise."""
    g = torch.Generator().manual_seed(seed)
    p, k, h, w = shape
    cx = torch.rand(p, k, 1, 1, generator=g) * (w + 2) - 1.5
    cy = torch.rand(p, k, 1, 1, generator=g) * (h + 2) - 1.5
    amp = torch.rand(p, k, 1, 1, generator=g) * 0.9 + 0.05
    yy, xx = torch.arange(h).view(1, 1, h, 1), torch.arange(w).view(1, 1, 1, w)
    return amp * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 8) + torch.randn(p, k, h, w, generator=g) * 0.01


def _dark(lib, hm: torch.Tensor, *, channels_last: bool) -> torch.Tensor:
    """The kernel's `[P, K, 3]` of CPU heatmaps `[P, K, h, w]`, passed in either layout."""
    x = hm.to(MPS)
    x = x.permute(0, 2, 3, 1).contiguous() if channels_last else x
    return pose_metal.dark_decode(lib, x, hm.shape[1], channels_last=channels_last, **DARK).cpu()


def _dark_error(lib, hm: torch.Tensor) -> float:
    """Largest coordinate difference to the PyTorch decode (pixels); scores and both layouts must agree exactly."""
    coords, score = vitpose.decode(hm)
    got = _dark(lib, hm, channels_last=False)
    assert torch.equal(got, _dark(lib, hm, channels_last=True))
    assert torch.equal(got[..., 2], score)
    return float((got[..., :2] - coords).abs().max())


def test_dark_decode_is_the_pytorch_decode(lib):
    for seed in SEEDS:
        assert _dark_error(lib, _peaks(seed)) <= DARK_PX_PEAKS
        noise = torch.randn(2, 5, 20, 31, generator=torch.Generator().manual_seed(seed))
        assert _dark_error(lib, noise) <= DARK_PX_NOISE
    small = _peaks(0, (2, 3, vitpose.RADIUS, vitpose.RADIUS))  # the smallest map the blur's border allows
    assert _dark_error(lib, small) <= DARK_PX_PEAKS
    ties = torch.zeros(1, 2, 9, 11)  # constant, and two equal maxima: the first one
    ties[0, 1, 3, 4], ties[0, 1, 6, 2] = 1.0, 1.0
    got = _dark(lib, ties, channels_last=False)
    assert _dark_error(lib, ties) <= DARK_PX_PEAKS
    assert got[0, 0].tolist() == [0.0, 0.0, 0.0]
    assert got[0, 1, :2].round().tolist() == [4.0, 3.0]
    wide = _peaks(1, (1, 2, 30, 300))  # more pixels than 256 threads x one stride, a long row
    assert _dark_error(lib, wide) <= DARK_PX_PEAKS


def test_dark_decode_nan_and_infinite_heatmaps(lib):
    """NaN propagates as in PyTorch (x, y and score NaN); all -inf decodes at pixel 0; never a wild index."""
    hm = _peaks(0, (2, 8, 16, 12))
    nan, inf = float("nan"), float("inf")
    hm[0, 0] = nan  # all NaN: no maximum (the kernel's index stayed 0xffffffff)
    hm[0, 1] = -inf  # no finite maximum
    hm[0, 2, 3, 4] = nan  # one NaN far from the peak
    hm[0, 3, 15, 11] = nan  # the last pixel
    hm[0, 4, 5, 5], hm[0, 4, 5, 6] = inf, -inf  # a NaN blur, a finite score
    hm[0, 5] = inf
    hm[0, 6, 7, 7] = inf
    hm[1, 0, 0, 0] = nan
    coords, score = vitpose.decode(hm)
    want = torch.cat([coords, score[..., None]], -1)
    for channels_last in (False, True):
        got = _dark(lib, hm, channels_last=channels_last)
        for p, k in ((0, 0), (0, 2), (0, 3), (1, 0)):
            assert got[p, k].isnan().all()
        assert got[0, 1].tolist() == [0.0, 0.0, -inf]
        assert got[0, 4, :2].isnan().all()
        assert got[0, 4, 2] == inf
        assert torch.equal(got.isnan(), want.isnan())
        assert torch.equal(got[..., 2].nan_to_num(nan=0.0), want[..., 2].nan_to_num(nan=0.0))
        clean = ~want.isnan().any(-1)
        assert (got[clean][:, :2] - want[clean][:, :2]).abs().max() <= DARK_PX_PEAKS
        assert clean.sum() == 11  # the 8 untouched maps, all -inf, all +inf, the +inf peak


def test_dark_decode_rejects_maps_smaller_than_the_blur_radius(lib):
    """The blur's reflected border needs `radius` pixels: an error as PyTorch's index error, not other numbers."""
    r = vitpose.RADIUS
    for shape in ((1, 2, r - 1, 12), (1, 2, 12, r - 1), (1, 2, 1, 1), (1, 2, 0, 8)):
        hm = torch.zeros(shape, device=MPS)
        for channels_last in (False, True):
            x = hm.permute(0, 2, 3, 1).contiguous() if channels_last else hm
            with pytest.raises(ValueError, match="smaller than the blur radius"):
                pose_metal.dark_decode(lib, x, 2, channels_last=channels_last, **DARK)
    with pytest.raises(IndexError):
        vitpose.decode(torch.zeros(1, 2, r - 1, 12))
    assert pose_metal.dark_decode(
        lib, torch.zeros(0, 17, 64, 48, device=MPS), 17, channels_last=False, **DARK
    ).shape == (
        0,
        17,
        3,
    )


# ------------------------------------------------------------------------------------------------------ pe_rope


def _rope_error(dtype: torch.dtype, seed: int, npt: int, shape: tuple[int, int, int, int] = (2, 37, 3, 16)) -> float:
    """max |kernel - `_pe_rope`| / max |reference| over q and k for `shape` = (batch, tokens, heads, head dim)."""
    torch.manual_seed(seed)
    b, n, h, d = shape
    qkv = torch.randn(b, n, 3, h, d).to(dtype)
    angle = torch.rand(n - npt, d) * 6.283
    rope = torch.cat([angle.sin(), angle.cos()], -1)
    want = torch.stack(transformer._pe_rope(qkv, rope, npt))
    q, k = vit_metal.pe_rope(vit_metal.load(MPS), qkv.to(MPS), rope.to(MPS), npt)
    assert q.dtype == dtype
    assert q.is_contiguous()
    assert k.is_contiguous()
    got = torch.stack([q, k]).cpu()
    assert torch.equal(got[:, :, :, :npt], want[:, :, :, :npt])  # prefix tokens: copied
    return float((got.double() - want.double()).abs().max() / want.double().abs().max())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=str)
def test_pe_rope_is_the_portable_rope(dtype):
    for seed in SEEDS:
        for npt in (0, 1, 5):
            assert _rope_error(dtype, seed, npt) <= ROPE_REL[dtype]
    assert _rope_error(dtype, 0, 1, (1, 2, 1, 2)) <= ROPE_REL[dtype]  # one pair, one rotated token


def test_pe_rope_strided_operands_empty_batch_and_bad_shapes():
    lib = vit_metal.load(MPS)
    torch.manual_seed(0)
    qkv, rope = torch.randn(2, 9, 3, 4, 8, device=MPS), torch.randn(8, 16, device=MPS)
    want = vit_metal.pe_rope(lib, qkv, rope, 1)
    got = vit_metal.pe_rope(lib, qkv.transpose(0, 1).contiguous().transpose(0, 1), rope.t().contiguous().t(), 1)
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])
    q, k = vit_metal.pe_rope(lib, qkv[:0], rope, 1)
    assert q.shape == k.shape == (0, 4, 9, 8)
    for bad_qkv, bad_rope in ((qkv, rope[:7]), (qkv, rope.half()), (qkv[..., :7], rope[:, :14]), (qkv[:, :, :2], rope)):
        with pytest.raises(ValueError, match="pe_rope"):
            vit_metal.pe_rope(lib, bad_qkv, bad_rope, 1)


# ------------------------------------------------------------------------------------------------- wrong device


def test_cpu_tensors_and_devices_raise(lib):
    """Nothing runs on the CPU in place of a Metal kernel: a CPU operand raises, a CPU device has no library."""
    with pytest.raises(RuntimeError, match="CPU tensor"):
        pose_metal.warp_crop(lib, torch.rand(1, 3, 9, 7), torch.zeros(2, 4), torch.zeros(2, 5))
    with pytest.raises(RuntimeError, match="CPU tensor"):
        pose_metal.warp_crop(lib, torch.rand(1, 3, 9, 7), torch.zeros(2, 4, device=MPS), torch.zeros(2, 5, device=MPS))
    with pytest.raises(RuntimeError, match="CPU tensor"):
        pose_metal.deconv4s2_relu(lib, torch.zeros(1, 6, 32), torch.zeros(2), 2, 3)
    with pytest.raises(RuntimeError, match="CPU tensor"):
        pose_metal.dark_decode(lib, torch.zeros(1, 2, 8, 8), 2, channels_last=False, **DARK)
    with pytest.raises(RuntimeError, match="CPU tensor"):
        vit_metal.pe_rope(vit_metal.load(MPS), torch.zeros(1, 5, 3, 2, 4), torch.zeros(4, 8), 1)
    for load in (pose_metal.load, vit_metal.load):
        with pytest.raises(KernelUnavailableError, match="Apple MPS only"):
            load(torch.device("cpu"))
