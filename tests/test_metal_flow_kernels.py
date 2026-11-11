"""The real Metal kernel of `flow_metal` on Apple MPS against RAFT's / SEA-RAFT's correlation pyramid (no weights)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fastforks.flow.models.raft import net as raft
from fastforks.flow.models.sea_raft import net as sea_raft
from fastforks.kernels import flow_metal

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS"),
]

MPS = torch.device("mps")
SEEDS = range(4)  # fixed: the seeds the bounds below were measured on (Apple M1 Pro, macOS 27, torch 2.14)
# Bound, fixed: 1.5 x the worst of `SEEDS` measured on that machine, rounded up to 2 digits. corr_window against the
# float64 pyramid + lookup of the same (dtype-rounded) operands on the CPU, max error / max value.
REL = {torch.float32: 2.8e-7, torch.float16: 5.5e-4}  # fixed: measured 1.82e-7, 3.60e-4
# (h, w, channels, radius): RAFT-large / SEA-RAFT (256, 4), RAFT-small (128, 3), the other instantiations; sides
# >= 16 as SEA-RAFT's pyramid halves once more after its last level
SHAPES = [(20, 28, 256, 4), (17, 23, 128, 3), (16, 19, 384, 2), (18, 16, 512, 0)]
LEVELS = 4  # fixed: RAFT's and SEA-RAFT's corr_levels


def _pool(f: torch.Tensor, family: str) -> torch.Tensor:
    """One pyramid step: RAFT average-pools, SEA-RAFT halves bilinearly (the same values: average pooling)."""
    if family == "raft":
        return F.avg_pool2d(f, 2, stride=2)
    return F.interpolate(f, scale_factor=0.5, mode="bilinear", align_corners=False)


def _case(seed: int, shape: tuple[int, int, int, int], dtype: torch.dtype, family: str) -> float:
    """Max error / max value of `CorrWindows` against the family's float64 `_pyramid` + `_lookup` on the CPU."""
    h, w, c, r = shape
    g = torch.Generator().manual_seed(seed)
    f1, f2 = (torch.randn(2, c, h, w, generator=g).to(dtype) for _ in range(2))
    coords = raft._coords(2, h, w, torch.device("cpu")) + torch.randn(2, 2, h, w, generator=g) * 6
    coords[0, :, 0, 0] = torch.tensor([float("nan"), 3.0])  # a non-finite coordinate reads zeros
    coords[1, :, 1, 1] = torch.tensor([-1e30, 1e30])  # far but finite: zeros
    net = raft if family == "raft" else sea_raft
    want = net._lookup(net._pyramid(f1.double(), f2.double(), LEVELS), coords.double(), r)
    levels = [f2.to(MPS)]
    for _ in range(LEVELS - 1):
        levels.append(_pool(levels[-1], family))
    got = flow_metal.CorrWindows(flow_metal.load(MPS), f1.to(MPS), levels, r)(coords.to(MPS))
    assert got.shape == want.shape
    assert got.dtype == dtype
    assert torch.isfinite(got).all()
    assert (got[0, :, 0, 0] == 0).all()
    assert (got[1, :, 1, 1] == 0).all()
    want[0, :, 0, 0] = want[1, :, 1, 1] = 0  # grid_sample's NaN / far reads, checked above
    return ((got.cpu().double() - want).abs().max() / want.abs().max()).item()


@pytest.mark.parametrize("family", ["raft", "sea_raft"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("shape", SHAPES, ids=[f"{h}x{w}c{c}r{r}" for h, w, c, r in SHAPES])
def test_corr_window_matches_pyramid_lookup(family: str, dtype: torch.dtype, shape: tuple[int, int, int, int]):
    worst = max(_case(seed, shape, dtype, family) for seed in SEEDS)
    assert worst <= REL[dtype], f"{family} {shape} {dtype}: {worst:.3g}"
