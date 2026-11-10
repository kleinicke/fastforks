"""SEA-RAFT: the card's variants, the network on random weights, the half-resolution (Spring) path, the engine's input
checks (CPU), and the CUDA engine against the portable fp32 runner away from the golden's 384x512 (GPU).

The goldens run one 384x512 pair with a small motion: every 1/8 grid there is even, nothing is padded, the batch is 1
and the lookup stays near the centre of its window. The GPU tests cover what that leaves out.
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn.functional as F

from fastforks import flow
from fastforks._core import registry
from fastforks._core.bench import synthetic_pair
from fastforks._core.errors import InputError, WeightsError
from fastforks.flow.models.sea_raft import model as family
from fastforks.flow.models.sea_raft.net import MEDIUM, SEARAFT, SMALL

if TYPE_CHECKING:
    from pathlib import Path

VARIANTS = {  # id -> (size, scale, min_size)
    "sea-raft-m": ("medium", 0, 128),
    "sea-raft-m-kitti": ("medium", 0, 128),
    "sea-raft-m-spring": ("medium", -1, 256),
    "sea-raft-m-things": ("medium", 0, 128),
    "sea-raft-s": ("small", 0, 128),
    "sea-raft-s-kitti": ("small", 0, 128),
    "sea-raft-s-spring": ("small", -1, 256),
}


def _pair(height: int, width: int, shift: int = 8, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """A textured pair on the CPU; item `i` of a batch moves by `shift * (i + 1)` px."""
    pairs = [synthetic_pair(height, width, torch.device("cpu"), shift=shift * (i + 1)) for i in range(batch)]
    return torch.cat([a for a, _ in pairs]), torch.cat([b for _, b in pairs])


# ---------------------------------------------------------------------------------------------------------- CPU


def test_card_variants() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("flow").values() if c.family == "sea_raft"}
    assert sorted(cards) == sorted(VARIANTS)
    for vid, (size, scale, min_size) in VARIANTS.items():
        card, variant = cards[vid]
        params = card.params | variant.params
        assert (params["size"], params["scale"]) == (size, scale), vid
        assert variant.io.min_size == min_size, vid  # the halved pair still has to hold the correlation pyramid
        data = variant.licence.get("training_data", card.licence.training_data)
        assert ("spring" in data) == vid.endswith("-spring"), vid
    assert set(flow.list_models()) >= set(VARIANTS)
    assert "sea-raft-m-sintel" not in flow.list_models()  # it was the checkpoint that is now sea-raft-m


@pytest.mark.parametrize(("cfg", "size"), [(SMALL, (128, 160)), (MEDIUM, (136, 168))])
def test_net_shape(cfg: object, size: tuple[int, int]) -> None:
    torch.manual_seed(0)
    net = SEARAFT(cfg).eval()
    a, b = (2 * t - 1 for t in _pair(*size, batch=2))
    with torch.inference_mode():
        out = net(a, b, 2)
    assert out.shape == (2, 2, *size)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


@pytest.fixture(scope="module")
def small_weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A SEA-RAFT (S) checkpoint with random weights, as `load(weights=...)` takes it."""
    from safetensors.torch import save_file  # noqa: PLC0415 - optional for the other tests

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("sea_raft") / "random-s.safetensors"
    save_file(SEARAFT(SMALL).state_dict(), path)
    return path


def test_odd_size_is_padded(small_weights: Path) -> None:
    model = flow.load("sea-raft-s", weights=small_weights, device="cpu", options={"iters": 1})
    a, b = _pair(131, 173)  # neither a multiple of 8: replicate padding, cropped again
    out = model(a, b).flow
    assert out.shape == (1, 2, 131, 173)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("size", [(256, 320), (257, 323)])
def test_spring_runs_at_half_resolution(small_weights: Path, size: tuple[int, int]) -> None:
    """`scale = -1` is upstream's `calc_flow`: halve the pair, run, upsample the flow and double it."""
    h, w = size
    full = flow.load("sea-raft-s", weights=small_weights, device="cpu", options={"iters": 1})
    half = flow.load("sea-raft-s-spring", weights=small_weights, device="cpu", options={"iters": 1})
    a, b = _pair(h, w)
    got = half(a, b).flow
    assert got.shape == (1, 2, h, w)
    small = [F.interpolate(t, scale_factor=0.5, mode="bilinear", align_corners=False) for t in (a, b)]
    assert small[0].shape[-2:] == (h // 2, w // 2)
    want = F.interpolate(full(*small).flow, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
    torch.testing.assert_close(got[..., : 2 * (h // 2), : 2 * (w // 2)], want, rtol=0, atol=1e-5)
    if h % 2:  # an odd size: the halving drops the last row and column, the flow repeats its last ones there
        torch.testing.assert_close(got[..., -1, :], got[..., -2, :], rtol=0, atol=0)
        torch.testing.assert_close(got[..., :, -1], got[..., :, -2], rtol=0, atol=0)
    assert (got - full(a, b).flow).abs().max() > 1e-3  # not the full-resolution pass
    with torch.inference_mode():
        exported = half.module()(a, b)  # export does what the call does
    torch.testing.assert_close(exported, got, rtol=0, atol=1e-5)


def test_spring_min_size(small_weights: Path) -> None:
    a, b = _pair(200, 240)
    flow.load("sea-raft-s", weights=small_weights, device="cpu", options={"iters": 1})(a, b)
    half = flow.load("sea-raft-s-spring", weights=small_weights, device="cpu", options={"iters": 1})
    with pytest.raises(InputError, match="256"):  # 100x120 after the halving: under the 128 px of the pyramid
        half(a, b)


def test_unknown_scale_is_refused() -> None:
    ctx = types.SimpleNamespace(
        params=family.Params(size="small", iters=4, scale=1),
        variant=types.SimpleNamespace(ref="sea-raft-x@1"),
    )
    with pytest.raises(ValueError, match="scale must be 0 or -1"):
        family.Portable(ctx)


@pytest.mark.parametrize(
    ("shape1", "shape2", "dtype", "match"),
    [
        ((3, 100, 128), (3, 100, 128), torch.float32, "multiples of 8"),
        ((3, 128, 132), (3, 128, 132), torch.float32, "multiples of 8"),
        ((3, 56, 128), (3, 56, 128), torch.float32, "at least 64 px"),  # 7 cells: no cell at pyramid level 3
        ((3, 128, 128), (3, 128, 128), torch.float16, "float32"),
        ((4, 128, 128), (4, 128, 128), torch.float32, r"\[3, H, W\]"),
        ((3, 128, 128), (3, 128, 136), torch.float32, r"\[3, H, W\]"),
        ((1, 3, 128, 128), (1, 3, 128, 128), torch.float32, r"\[3, H, W\]"),
    ],
)
def test_engine_refuses_bad_inputs(shape1: tuple, shape2: tuple, dtype: torch.dtype, match: str) -> None:
    """The kernels trust the shapes they are given: the engine checks them before the first launch."""
    engine = pytest.importorskip("fastforks.flow.models.sea_raft.kernels.engine")
    stub = types.SimpleNamespace(cfg=SMALL)
    with pytest.raises(ValueError, match=match):
        engine.Engine._check(stub, torch.zeros(shape1, dtype=dtype), torch.zeros(shape2, dtype=dtype))
    assert engine.Engine._check(stub, torch.zeros(3, 64, 72), torch.zeros(3, 64, 72)) == (64, 72)


# ---------------------------------------------------------------------------------------------------------- GPU

gpu = [pytest.mark.model, pytest.mark.gpu]
REFS = ["sea-raft-s@1", "sea-raft-m@1"]
# fixed: 1.5 x the worst (mean, p99) |kernels - portable fp32| in px over the refs of a case, measured on an RTX 5060 Ti
# and rounded up to 2 digits; the measured (mean, p99) of S / M in the comments
BOUNDS = {
    "sizes": (0.0013, 0.0068),  # 250x333: 0.00084, 0.0045 / 0.00069, 0.0036; 136x168: 0.00079, 0.0041 / 0.00083, 0.0040
    "batch": (0.0018, 0.012),  # 0.0011, 0.0074 / 0.00097, 0.0055
    "shift": (0.092, 0.91),  # 0.0095, 0.093 / 0.061, 0.60: 104 px, most of it in the cells whose match left the image
    "iters": (0.0031, 0.016),  # 0.0020, 0.010 / 0.0016, 0.0087
    "spring": (0.0022, 0.011),  # 0.0013, 0.0070 / 0.0014, 0.0072
}


def _load(ref: str, mode: str, **options: int) -> flow.FlowModel:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    try:
        dtype = {"dtype": "float32"} if mode == "portable" else {}
        return flow.load(ref, mode=mode, device="cuda", options=options or None, **dtype)
    except WeightsError as e:
        pytest.skip(f"{ref} weights not available: {e}")


def _errors(ref: str, a: torch.Tensor, b: torch.Tensor, **options: int) -> tuple[float, float, torch.Tensor]:
    """(mean, p99) of |kernels - portable fp32| in px, and the portable flow."""
    want = _load(ref, "portable", **options)(a.cuda(), b.cuda()).flow
    got = _load(ref, "kernels", **options)(a.cuda(), b.cuda()).flow
    assert got.shape == want.shape
    assert got.dtype == torch.float32
    d = (got - want).abs().flatten().cpu()
    return d.mean().item(), d.kthvalue(max(1, round(0.99 * d.numel()))).values.item(), want


def _check(case: str, mean: float, p99: float) -> None:
    assert mean <= BOUNDS[case][0], f"{case}: mean {mean:.5f} px"
    assert p99 <= BOUNDS[case][1], f"{case}: p99 {p99:.5f} px"


@pytest.mark.parametrize("ref", [pytest.param(r, marks=gpu) for r in REFS])
@pytest.mark.parametrize("size", [(250, 333), (136, 168)])  # padded to 256x336 (odd 1/8 width); 17x21 cells
def test_kernels_match_portable_sizes(ref: str, size: tuple[int, int]) -> None:
    mean, p99, _ = _errors(ref, *_pair(*size))
    _check("sizes", mean, p99)


@pytest.mark.parametrize("ref", [pytest.param(r, marks=gpu) for r in REFS])
def test_kernels_match_portable_batch(ref: str) -> None:
    a, b = _pair(200, 264, batch=2)
    mean, p99, want = _errors(ref, a, b)
    assert (want[0, 0] - want[1, 0]).median() > 6  # 8 px and 16 px motion: two flows, each at its place
    _check("batch", mean, p99)


@pytest.mark.parametrize("ref", [pytest.param(r, marks=gpu) for r in REFS])
def test_kernels_match_portable_large_shift(ref: str) -> None:
    """A 104 px motion: the lookup windows of all four levels sit far from the start and reach over the border."""
    mean, p99, want = _errors(ref, *_pair(384, 640, shift=104))
    assert want[:, 0].median().abs() > 90  # the network follows it (x flow about -104 px)
    _check("shift", mean, p99)


@pytest.mark.parametrize("ref", [pytest.param(r, marks=gpu) for r in REFS])
def test_kernels_match_portable_iters(ref: str) -> None:
    mean, p99, _ = _errors(ref, *_pair(248, 328, shift=20), iters=7)
    _check("iters", mean, p99)


@pytest.mark.parametrize("ref", [pytest.param(r, marks=gpu) for r in ("sea-raft-s-spring@1", "sea-raft-m-spring@1")])
def test_kernels_match_portable_spring(ref: str) -> None:
    mean, p99, _ = _errors(ref, *_pair(540, 962, shift=30))  # 270x481 after the halving, padded to 272x488
    _check("spring", mean, p99)
