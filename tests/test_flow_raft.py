"""RAFT: the engine's input checks (CPU), and the CUDA engine against the portable fp32 runner away from the golden's
384x512 (GPU).

The goldens run one 384x512 pair with a small motion: every 1/8 grid there is even, nothing is padded, the batch is 1,
the iterations are the default 12 and the lookup stays near the centre of its window. The GPU tests cover what that
leaves out; tests/test_kernel_fuzz.py checks the lookup and the upsampling kernels on their own.
"""

from __future__ import annotations

import pytest
import torch

from fastforks import flow
from fastforks._core.bench import synthetic_pair
from fastforks._core.errors import WeightsError


def _pair(height: int, width: int, shift: int = 8, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """A textured pair on the CPU; item `i` of a batch moves by `shift * (i + 1)` px."""
    pairs = [synthetic_pair(height, width, torch.device("cpu"), shift=shift * (i + 1)) for i in range(batch)]
    return torch.cat([a for a, _ in pairs]), torch.cat([b for _, b in pairs])


# ---------------------------------------------------------------------------------------------------------- CPU


@pytest.mark.parametrize(
    ("shape1", "shape2", "dtype", "match"),
    [
        ((1, 3, 132, 128), (1, 3, 132, 128), torch.float32, "multiples of 8"),
        ((1, 3, 128, 100), (1, 3, 128, 100), torch.float32, "multiples of 8"),
        ((1, 3, 120, 128), (1, 3, 120, 128), torch.float32, "at least 128"),  # 15 cells: 1 x 1 at pyramid level 3
        ((1, 3, 0, 128), (1, 3, 0, 128), torch.float32, "at least 128"),
        ((1, 3, 128, 128), (1, 3, 128, 128), torch.float16, "float32"),
        ((1, 3, 128, 128), (1, 3, 128, 128), torch.uint8, "float32"),
        ((1, 4, 128, 128), (1, 4, 128, 128), torch.float32, r"\[B, 3, H, W\]"),
        ((1, 3, 128, 128), (1, 3, 128, 136), torch.float32, r"\[B, 3, H, W\]"),
        ((2, 3, 128, 128), (1, 3, 128, 128), torch.float32, r"\[B, 3, H, W\]"),
        ((3, 128, 128), (3, 128, 128), torch.float32, r"\[B, 3, H, W\]"),
    ],
)
def test_engine_refuses_bad_inputs(shape1: tuple, shape2: tuple, dtype: torch.dtype, match: str) -> None:
    """The kernels trust the shapes they are given: the engine checks them before the first launch."""
    engine = pytest.importorskip("fastforks.flow.models.raft.kernels.engine")
    with pytest.raises(ValueError, match=match):
        engine.check(torch.zeros(shape1, dtype=dtype), torch.zeros(shape2, dtype=dtype))
    engine.check(torch.zeros(2, 3, 128, 136), torch.zeros(2, 3, 128, 136))  # the smallest pair passes


# ---------------------------------------------------------------------------------------------------------- GPU

gpu = [pytest.mark.model, pytest.mark.gpu]
REFS = ["raft-small@1", "raft-large@1"]
# fixed: 1.5 x the worst (mean, p99) |kernels - portable fp32| in px over the refs of a case, measured on an RTX 5060 Ti
# and rounded up to 2 digits; the measured (mean, p99) of small / large in the comments
BOUNDS = {
    "sizes": (0.0038, 0.016),  # 250x333: 0.0024, 0.0094 / 0.0012, 0.0054; 136x168: 0.0025, 0.010 / 0.0014, 0.0056
    "batch": (0.0048, 0.023),  # 0.0032, 0.015 / 0.0014, 0.0063
    "shift": (0.025, 0.12),  # 0.016, 0.079 / 0.0082, 0.038: 104 px
    "iters": (0.0061, 0.026),  # 0.0040, 0.017 / 0.0019, 0.0075
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
