"""ConvNeXt features: kernels vs portable fp16 beyond the golden pair (B=1, 128x160, synthetic), and the output layout.

ConvNeXt-L at batch 2 (eager first call, then the captured graph) and at a size above `_GRAPH_PIXELS` (always eager).
Bounds: the family's golden GPU tolerances (mean / p99 of |diff| over the median |portable| value). The layout test
runs the smallest variant on the CPU.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

import synth
from fastforks import features
from fastforks._core.errors import WeightsError
from fastforks.features.models.convnext_features import model as convnext

REF = "convnext_large.fb_in22k@1"
TOL = tomllib.loads(Path(convnext.__file__).with_name("golden.toml").read_text())["tolerance"]

pytestmark = [pytest.mark.model, pytest.mark.slow]


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
    want = port(x)
    for i in range(calls):
        got = fast(x)
        _close(got.last_hidden_state, want.last_hidden_state, f"{what} call {i + 1} tokens")
        _close(got.pooler_output, want.pooler_output, f"{what} call {i + 1} pooled")


def _uint8(*images: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()


@pytest.mark.gpu
def test_batch2(models):
    x = _uint8(synth.image(224, 224, seed=0), synth.image(224, 224, seed=1)).cuda()
    _compare(models, x, "B=2 224", calls=2)  # eager, then graph


@pytest.mark.gpu
def test_above_graph_pixels(models):
    h, w = 1024, 1056
    assert h * w > convnext._GRAPH_PIXELS  # the eager branch
    _compare(models, _uint8(synth.image(h, w)).cuda(), f"{h}x{w}", calls=1)


def test_layout_cpu():
    """Stride-32 tokens without a prefix token; `pooler_output` is the head norm of their mean."""
    try:
        m = features.load("convnext_tiny.fb_in22k@1", mode="portable", device="cpu")
    except WeightsError as e:
        pytest.skip(f"weights not available: {e}")
    out = m(_uint8(synth.image(96, 160))[0])
    assert (m.num_prefix_tokens, m.hidden_size, m.num_layers) == (0, 768, 4)
    assert out.grid == (3, 5)
    assert out.last_hidden_state.shape == (15, 768)
    norm = m._runner.fast.net.head.norm
    want = torch.nn.functional.layer_norm(out.last_hidden_state.mean(0), (768,), norm.weight, norm.bias, norm.eps)
    torch.testing.assert_close(out.pooler_output, want, atol=1e-4, rtol=0)
