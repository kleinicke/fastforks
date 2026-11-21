"""The compiled interactive session must retain the existing GPU golden accuracy."""

import importlib.util

import pytest
import torch

from fastforks import volume
from test_golden import golden, tolerance


@pytest.mark.gpu
@pytest.mark.skipif(
    importlib.util.find_spec("triton") is None,
    reason="torch.compile on CUDA needs Triton (not installed: no official Windows wheels)",
)
@pytest.mark.slow
@pytest.mark.model
def test_compiled_session_matches_real_ct_golden() -> None:
    """Compile the patch network but keep the high-level point-prompt session working."""
    ref = "nninteractive@1"
    want = golden(ref)
    model = volume.load(ref, mode="portable", device="cuda")
    assert isinstance(model, volume.VolumeInteractive)
    image = torch.from_numpy(want["input"])
    point = tuple(float(v) for v in want["point"])
    target = torch.from_numpy(want["mask"]).bool().to("cuda")
    with torch.inference_mode():
        eager = model(image, point=point).mask
        model.compile()
        compiled = model(image, point=point).mask
    tol = tolerance(ref, "portable")
    for result in (eager, compiled):
        err = (result != target).float().flatten()
        assert float(err.mean()) < tol["gpu_mean"]
        assert float(torch.quantile(err, 0.99)) < tol["gpu_p99"]
