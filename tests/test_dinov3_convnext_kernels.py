"""DINOv3 ConvNeXt kernels vs portable fp16 beyond the golden pair (B=1, 128x160, synthetic).

ConvNeXt-L (its fc2 GEMMs have K >= `_FP32_ACC_K`: cuBLAS + add instead of the fused fp16-accumulate kernel) at batch 2
(eager first call, then the captured graph), at a size above `_GRAPH_PIXELS` (always eager) and on a real photo
(scikit-image's `coins.png`, the SAM 3 golden input: finite, no fp16 overflow). Bounds: the family's golden GPU
tolerances (mean / p99 of |diff| over the median |portable| value).
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

import synth
from fastforks import features
from fastforks._core.errors import WeightsError
from fastforks.features.models.dinov3_convnext import model as convnext

REF = "dinov3-convnext-large@1"
TOL = tomllib.loads(Path(convnext.__file__).with_name("golden.toml").read_text())["tolerance"]

pytestmark = [pytest.mark.model, pytest.mark.gpu]


@pytest.fixture(scope="module")
def models() -> tuple[features.FeaturesModel, features.FeaturesModel]:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    try:
        port = features.load(REF, mode="portable", device="cuda", dtype="float16")
        fast = features.load(REF, mode="kernels", device="cuda")
    except WeightsError as e:
        pytest.skip(f"{REF} weights not available: {e}")
    assert fast.backend.mode == "kernels"
    assert fast.backend.kernels
    return port, fast


def _close(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    assert bool(torch.isfinite(got).all()), f"{what}: non-finite kernels output"
    err = ((got - want).abs() / want.abs().median()).flatten().cpu().numpy()
    assert err.mean() < TOL["gpu_mean"], f"{what}: mean {err.mean():.4f}"
    assert np.percentile(err, 99) < TOL["gpu_p99"], f"{what}: p99 {np.percentile(err, 99):.4f}"


def _compare(
    models: tuple[features.FeaturesModel, features.FeaturesModel], x: torch.Tensor, what: str, calls: int
) -> None:
    port, fast = models
    want = port(x).last_hidden_state
    for i in range(calls):
        _close(fast(x).last_hidden_state, want, f"{what} call {i + 1}")


def _uint8(*images: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous().cuda()


def test_batch2(models):
    x = _uint8(synth.image(224, 224, seed=0), synth.image(224, 224, seed=1))
    _compare(models, x, "B=2 224", calls=2)  # eager, then graph


def test_above_graph_pixels(models):
    h, w = 1024, 1056
    assert h * w > convnext._GRAPH_PIXELS  # the eager branch
    _compare(models, _uint8(synth.image(h, w)), f"{h}x{w}", calls=1)


def test_real_image(models):
    path = Path(os.environ.get("FASTFORKS_GOLDEN_PATH", "")) / "sam3@1.npz"
    if not path.is_file():
        pytest.skip("coins.png (SAM 3 golden input) not available (set FASTFORKS_GOLDEN_PATH)")
    img = np.load(path)["input"].round().clip(0, 255).astype(np.uint8)
    _compare(models, _uint8(img), "coins.png", calls=2)
