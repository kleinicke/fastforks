"""Depth family helpers (no weights) and per-family behaviour (weights, skipped without them; GPU tests marked)."""

from __future__ import annotations

import os
import types

import numpy as np
import pytest
import torch

from fastforks import BackendUnavailableError, InputError, WeightsError, depth
from fastforks._core import registry
from fastforks.depth.models.depth_anything_v2.model import network_size
from fastforks.depth.models.unidepth_v2.model import RATIO_BOUNDS, _geometry, _prepare, _restore
from fastforks.depth.models.unidepth_v2.net.utils import PinholeCamera
from fastforks.depth.models.unik3d import model as unik3d_model
from fastforks.shared import midas_beit
from fastforks.shared.midas_beit.net.beit import Attention
from synth import image
from test_golden import HEAVY_CPU

ZOO2 = ["metric3d-vit-large", "midas-dpt-beit-large-512", "zoedepth-nk", "marigold-depth-v1-1", "unidepth-v2-vitl14"]
ZOO3 = [
    *("da3mono-large", "da3metric-large", "da3-large-1.1", "da3-giant-1.1", "da3nested-giant-large-1.1"),
    *("moge-2-vits-normal", "moge-2-vitl-normal", "moge-3-vitl", "depth-pro", "unik3d-vitl"),
]


def _load(ref: str, **options: object) -> depth.DepthModel:
    card, variant = registry.resolve("depth", ref)
    try:
        registry.weights(card, variant, allow_download=False)
    except WeightsError as e:
        pytest.skip(f"weights of {ref} not available locally: {e}")
    return depth.load(ref, device="cpu", options=options)


def _map(result: depth.DepthResult) -> torch.Tensor:
    out = result.depth if result.depth is not None else result.inverse_depth
    assert out is not None
    return out


@pytest.mark.parametrize(
    ("h", "w", "short_side", "want"),
    [
        (375, 1242, 518, (518, 1722)),  # KITTI: shorter side to 518, longer side rounded to a multiple of 14
        (994, 1482, 518, (518, 770)),  # Middlebury-H
        (1, 1, 518, (518, 518)),
        (480, 640, 500, (504, 672)),  # short_side not a multiple of 14: rounded up
        (505, 505, 505, (518, 518)),  # rounding down would go below short_side: ceil instead
        (500, 511, 500, (504, 504)),  # 511 / 14 = 36.5 rounds half to even (numpy / upstream cv2 sizes)
        (1, 1, 14, (14, 14)),
    ],
)
def test_dav2_network_size(h, w, short_side, want):
    assert network_size(h, w, short_side) == want


@pytest.mark.parametrize(("h", "w"), [(352, 480), (480, 640), (375, 1242), (994, 1482), (1000, 400), (97, 131)])
def test_unidepth_padding(h, w):
    """Pad to the aspect range only outside it (352 x 480 padded by -1 px before: W * H / W rounded down)."""
    g = _geometry(h, w)
    assert min(g.paddings) >= 0
    assert g.padded == (h + g.paddings[2] + g.paddings[3], w + g.paddings[0] + g.paddings[1])
    lo, hi = RATIO_BOUNDS
    if lo <= w / h <= hi:
        assert g.paddings == (0, 0, 0, 0)


@pytest.mark.parametrize(("h", "w"), [(480, 640), (1000, 400)])
def test_unidepth_given_camera_round_trip(h, w):
    """Given K -> padded, resized network camera -> rays -> points at the input size: projecting the points with K
    gives the pixel centres `i` (OpenCV), through padding and resize: no half-pixel shift."""
    g = _geometry(h, w)
    k = torch.tensor([[[0.8 * w, 0.0, 0.47 * w], [0.0, 0.8 * w, 0.52 * h], [0.0, 0.0, 1.0]]])
    _, _, cam = _prepare(types.SimpleNamespace(io=types.SimpleNamespace(input="0-255")), torch.zeros(1, 3, h, w), k)
    rays = cam.get_rays((1, *g.net))
    pts = _restore(g, rays / rays[:, 2:] * 2.0, None, None, k)["points"][0]  # a plane at z = 2
    u = k[0, 0, 0] * pts[..., 0] / pts[..., 2] + k[0, 0, 2]
    v = k[0, 1, 1] * pts[..., 1] / pts[..., 2] + k[0, 1, 2]
    inner = (slice(2, -2), slice(2, -2))  # bilinear resizing clamps at the border
    torch.testing.assert_close(u[inner], torch.arange(w).float().expand(h, w)[inner], rtol=0, atol=1e-3)
    torch.testing.assert_close(v[inner], torch.arange(h)[:, None].float().expand(h, w)[inner], rtol=0, atol=1e-3)


def _reprojection_error(pts: torch.Tensor, k: torch.Tensor) -> float:
    """Max distance (px) of the points `[H, W, 3]` projected with `k` `[1, 3, 3]` from the pixel centres (OpenCV)."""
    h, w = pts.shape[:2]
    u = k[0, 0, 0] * pts[..., 0] / pts[..., 2] + k[0, 0, 2] - torch.arange(w)
    v = k[0, 1, 1] * pts[..., 1] / pts[..., 2] + k[0, 1, 2] - torch.arange(h)[:, None]
    inner = (slice(2, -2), slice(2, -2))  # bilinear resizing clamps at the border
    return max(u[inner].abs().max().item(), v[inner].abs().max().item())


@pytest.mark.parametrize(("h", "w"), [(480, 640), (1000, 400), (375, 1242)])
def test_unidepth_predicted_intrinsics_round_trip(h, w):
    """Predicted network camera -> returned intrinsics: they project the returned points onto their pixels.

    480 x 640 runs at 490 x 644; the scalar resize factor (upstream) gave 4 / 10 px errors."""
    g = _geometry(h, w)
    k_net = torch.tensor([[[0.8 * g.net[1], 0, 0.5 * g.net[1]], [0, 0.8 * g.net[1], 0.5 * g.net[0]], [0, 0, 1.0]]])
    rays = PinholeCamera(k_net).get_rays((1, *g.net))
    out = _restore(g, rays / rays[:, 2:] * 2.0, None, k_net, None)
    assert _reprojection_error(out["points"][0], out["intrinsics"]) < 1e-2


@pytest.mark.parametrize("level", [0, 9])
@pytest.mark.parametrize(("h", "w"), [(480, 640), (1000, 400), (375, 1242)])
def test_unik3d_given_camera_round_trip(h, w, level):
    """Given K -> rays at the network size -> points at the input size: projecting them with K gives the pixel centres.

    The scalar resize factor (upstream) put the rays up to 17 px off (the sides are rounded to multiples of 14)."""
    ctx = types.SimpleNamespace(
        io=types.SimpleNamespace(input="0-255"), params=types.SimpleNamespace(resolution_level=level)
    )
    k = torch.tensor([[[0.8 * w, 0.0, 0.47 * w], [0.0, 0.8 * w, 0.52 * h], [0.0, 0.0, 1.0]]])
    g, _, rays = unik3d_model._prepare(ctx, torch.zeros(1, 3, h, w), k)
    assert rays is not None
    pts = unik3d_model._restore(g, torch.zeros(1, 1, *g.net), None, rays)["points"][0]
    assert _reprojection_error(pts, k) < 1e-2


@pytest.mark.parametrize("ref", ZOO2 + ZOO3)
def test_kernels_needs_cuda(ref):
    with pytest.raises(BackendUnavailableError, match="needs a CUDA device"):
        depth.load(ref, mode="kernels", device="cpu")


@pytest.mark.slow
@pytest.mark.model
def test_metric3d_depth_scales_with_fx():
    model = _load("metric3d-vit-large")
    k = torch.tensor([[500.0, 0, 75], [0, 500, 50], [0, 0, 1]])
    x = image()
    one = _map(model(x, intrinsics=k))
    k[0, 0] *= 2
    torch.testing.assert_close(_map(model(x, intrinsics=k)), 2 * one, rtol=1e-5, atol=0)


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.parametrize("ref", ["metric3d-vit-large", "midas-dpt-beit-large-512"])
def test_odd_size(ref):
    out = _map(_load(ref)(image(97, 131)))
    assert out.shape == (97, 131)
    assert torch.isfinite(out).all()


def test_beit_index_cache_bounded(monkeypatch):
    monkeypatch.setattr(midas_beit, "_INDEX", {})
    attn = Attention(8, 2, (4, 4))
    torch.nn.init.normal_(attn.relative_position_bias_table)
    for side in range(2, 2 + midas_beit._INDEX_SIZES + 3):  # every input size adds an index
        midas_beit._rel_pos_bias(attn, (side, side + 1))
    assert len(midas_beit._INDEX) == midas_beit._INDEX_SIZES
    assert ((2, 3), torch.device("cpu")) not in midas_beit._INDEX  # the oldest sizes went first
    bias = midas_beit._rel_pos_bias(attn, (2, 3))  # regenerated: the vendored upstream bias
    assert ((2, 3), torch.device("cpu")) in midas_beit._INDEX
    torch.testing.assert_close(bias, attn._get_rel_pos_bias((2, 3)), rtol=0, atol=0)


@pytest.mark.slow
@pytest.mark.model
def test_marigold_seed():
    x = image(64, 128)  # sides multiples of 64: the UNet needs latents divisible by 8
    a = _load("marigold-depth-v1-1", steps=1, long_side=0)
    first = _map(a(x))
    torch.testing.assert_close(_map(a(x)), first, rtol=0, atol=0)  # same seed: same noise, same depth
    assert _map(a(image(128, 64))).shape == (128, 64)  # a new size draws new noise of its shape
    torch.testing.assert_close(_map(a(x)), first, rtol=0, atol=0)  # back to the first size: the same noise again
    del a
    other = _map(_load("marigold-depth-v1-1", steps=1, long_side=0, seed=1)(x))
    assert (other - first).abs().max() > 1e-3


@pytest.mark.slow
@pytest.mark.model
def test_zoedepth_too_small_for_reflect_padding():
    model = _load("zoedepth-nk")
    with pytest.raises(InputError, match=r"at least \d+ px"):
        model(np.zeros((4, 64, 3), np.uint8))


def _fit(ratio: torch.Tensor) -> tuple[float, float]:
    """Slope and zero crossing (in pixels) of a least-squares line through `ratio[i]` over the pixel index `i`."""
    i = torch.arange(len(ratio), dtype=torch.float64)
    slope, icept = np.polyfit(i.numpy(), ratio.double().numpy(), 1)
    return float(slope), float(-icept / slope)


@pytest.mark.slow
@pytest.mark.model
def test_da3_metric_focal_at_network_size():
    model = _load("da3metric-large")
    x = image()
    free = model(x)
    k = free.intrinsics
    assert k is not None  # 300 px at the network size, returned in pixels of the input
    one = _map(free)
    torch.testing.assert_close(_map(model(x, intrinsics=k)), one, rtol=1e-5, atol=0)  # the same focal length
    k = k.clone()
    k[0, 0] *= 2
    k[1, 1] *= 2
    torch.testing.assert_close(_map(model(x, intrinsics=k)), 2 * one, rtol=1e-5, atol=0)


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.parametrize("ref", ["da3-large-1.1", "da3nested-giant-large-1.1"])
def test_da3_camera_head_and_nested(ref):
    out = _load(ref)(image(97, 131))
    assert out.depth is not None
    assert out.confidence is not None
    assert out.intrinsics is not None
    assert out.depth.shape == out.confidence.shape == (97, 131)
    assert torch.isfinite(out.depth).all()
    assert (out.depth > 0).all()
    k = out.intrinsics  # from the predicted field of view, in pixels of the input
    assert (k[0, 2], k[1, 2], k[2, 2]) == (130 / 2, 96 / 2, 1.0)  # OpenCV centre ((W - 1) / 2, (H - 1) / 2)
    assert k[0, 0] > 0
    assert k[1, 1] > 0
    if ref.startswith("da3nested"):  # metric branch aligned to the relative one; sky from the metric branch
        assert out.mask is not None
        assert out.mask.shape == (97, 131)
        assert out.mask.dtype == torch.bool


@pytest.mark.slow
@pytest.mark.model
def test_moge_given_fx_unprojects_the_points():
    model = _load("moge-2-vits-normal", normals=True)
    x = image(64, 120)
    free = model(x)
    assert free.intrinsics is not None
    assert free.normals is not None
    torch.testing.assert_close(free.normals.norm(dim=-1), torch.ones(64, 120), rtol=0, atol=1e-5)
    k = torch.tensor([[150.0, 0, 60.0], [0, 150.5, 31.0], [0, 0, 1]])  # within the tolerance of MoGe's camera
    given = model(x, intrinsics=k)
    pts = given.points
    assert pts is not None  # only fx is used: square pixels, principal point at the image centre (upstream fov_x)
    torch.testing.assert_close(given.intrinsics, torch.tensor([[150.0, 0, 59.5], [0, 150, 31.5], [0, 0, 1]]))
    for bad in ([[150.0, 0, 50], [0, 150, 28], [0, 0, 1]], [[150.0, 0, 60], [0, 160, 32], [0, 0, 1]]):
        with pytest.raises(InputError, match="MoGe uses only fx"):
            model(x, intrinsics=bad)
    u = (torch.arange(120) - 59.5) / 150  # OpenCV pixel centres around (W - 1) / 2
    v = (torch.arange(64) - 31.5) / 150
    torch.testing.assert_close(pts[..., 0] / pts[..., 2], u.expand(64, 120), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(pts[..., 1] / pts[..., 2], v[:, None].expand(64, 120), rtol=1e-4, atol=1e-6)


@pytest.mark.model
def test_depth_pro_depth_scales_with_fx():
    if not os.environ.get(HEAVY_CPU):
        pytest.skip(f"three depth-pro forwards at a fixed 1536 px (CPU ~150 s); {HEAVY_CPU}=1 runs it")
    model = _load("depth-pro")
    x = image()
    free = model(x)
    k = free.intrinsics
    assert k is not None
    assert (k[0, 2], k[1, 2]) == (74.5, 49.5)
    one = _map(free)
    torch.testing.assert_close(_map(model(x, intrinsics=k)), one, rtol=1e-4, atol=0)  # FOV head skipped, same fx
    k = k.clone()
    k[0, 0] *= 2
    torch.testing.assert_close(_map(model(x, intrinsics=k)), 2 * one, rtol=1e-4, atol=0)


@pytest.mark.slow
@pytest.mark.model
def test_unik3d_rays_from_given_intrinsics():
    model = _load("unik3d-vitl")
    h, w = 64, 200  # aspect 3.1 > 2.5: zero-padded to 80 x 200 (8 px on top), the given cy shifts with it
    fx, fy, cx, cy = 120.0, 100.0, 80.0, 20.0
    pts = model(image(h, w), intrinsics=[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).points
    assert pts is not None
    assert pts.shape == (h, w, 3)
    sx, x0 = _fit((pts[..., 0] / pts[..., 2]).mean(0))  # rays of the given camera, bilinearly resized points
    sy, y0 = _fit((pts[..., 1] / pts[..., 2]).mean(1))
    assert sx == pytest.approx(1 / fx, rel=0.03)
    assert sy == pytest.approx(1 / fy, rel=0.03)
    assert x0 == pytest.approx(cx, abs=1.5)
    assert y0 == pytest.approx(cy, abs=1.5)


@pytest.mark.model
@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["portable", "kernels"])
@pytest.mark.parametrize(("ref", "boost"), [("midas-dpt-beit-large-512", 8), ("zoedepth-nk", 2)])
def test_dpt_head_fp16_finite_at_photo_levels(ref, boost, mode):
    """The fp16 DPT-BEiT head stays finite at a real size and at the activation levels of real photos.

    On photos its 3x3 convs reach ~1.1e5 in float32 (fp16 overflowed to inf / NaN depth before `scale_head`); the
    synthetic texture gives 2-8x less, so the first head conv is scaled up by `boost` (a power of two) to get there.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    card, variant = registry.resolve("depth", ref)
    try:
        registry.weights(card, variant, allow_download=False)
        m = depth.load(ref, mode=mode, device="cuda")
    except (WeightsError, BackendUnavailableError) as e:
        pytest.skip(f"{ref} {mode}: {e}")
    net = m._runner.net
    head = (net.core.core if ref == "zoedepth-nk" else net).scratch.output_conv[0]
    img = image(720, 1280)
    for k in (1, boost):
        with torch.no_grad():
            head.weight.mul_(k)
            head.bias.mul_(k)
        out = _map(m(img))
        assert out.shape[-2:] == (720, 1280)
        assert torch.isfinite(out).all(), f"x{k}: {int((~torch.isfinite(out)).sum())} non-finite pixels"


def test_beit_rel_pos_bias_compiles_in_inference_mode(monkeypatch):
    """`torch.compile` callers run under `inference_mode`: the index is built outside it, positionally (dynamo
    rejects `inference_mode(mode=False)`)."""
    monkeypatch.setattr(midas_beit, "_INDEX", {})
    attn = Attention(8, 2, (4, 4))
    torch.nn.init.normal_(attn.relative_position_bias_table)
    with torch.inference_mode():
        got = torch.compile(midas_beit._rel_pos_bias, backend="eager", fullgraph=True)(attn, (3, 5))
    torch.testing.assert_close(got, attn._get_rel_pos_bias((3, 5)), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_marigold_ddim_step_cached_coefficients(dtype):
    """The coefficients cached per timestep give diffusers' per-step results, bit for bit."""
    from fastforks.depth.models.marigold.net.ddim import DDIM, TRAIN_STEPS  # noqa: PLC0415

    gen = torch.Generator().manual_seed(0)
    v, x = (torch.randn(2, 4, 8, 8, generator=gen).to(dtype) for _ in range(2))
    for steps in (1, 4, 10):
        ddim = DDIM(steps)
        for t in ddim.timesteps:
            prev = t - TRAIN_STEPS // steps
            a_t = ddim.alphas_cumprod[t]
            a_prev = ddim.alphas_cumprod[prev] if prev >= 0 else ddim.final_alpha_cumprod
            x0 = (a_t**0.5) * x - ((1 - a_t) ** 0.5) * v
            eps = (a_t**0.5) * v + ((1 - a_t) ** 0.5) * x
            want = a_prev**0.5 * x0 + (1 - a_prev) ** 0.5 * eps
            torch.testing.assert_close(ddim.step(v, t, x), want, rtol=0, atol=0)


@pytest.mark.parametrize(("h", "w"), [(384, 512), (97, 131), (1000, 30)])
def test_zoedepth_sizes_are_python_ints(h, w):
    """Resize sizes and pad widths are Python ints (dynamo traces NumPy scalars into tensors)."""
    from fastforks.depth.models.zoedepth.model import _pads  # noqa: PLC0415
    from fastforks.depth.models.zoedepth.net.midas_core import Resize  # noqa: PLC0415

    for method in ("lower_bound", "upper_bound", "minimal"):  # (1000, 30): upper_bound rounds W to 0, as upstream
        size = Resize(512, 384, keep_aspect_ratio=True, ensure_multiple_of=32, resize_method=method).get_size(w, h)
        assert all(type(s) is int and s % 32 == 0 for s in size), size
    assert all(type(p) is int for p in _pads(h, w))
    assert _pads(h, w) == (int(np.sqrt(h / 2) * 3), int(np.sqrt(w / 2) * 3))  # the upstream NumPy values
