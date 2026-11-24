"""GeoCalib: the card, the pre-processing (resize / crop), the device LM step and the exported LM fit."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from fastforks import calibration
from fastforks._core import registry
from fastforks._core.patch import override
from fastforks.calibration.models.geocalib.model import _FIELDS, _GeoCalib, _no_uncertainty, preprocess
from fastforks.calibration.models.geocalib.net import lm_optimizer
from fastforks.calibration.models.geocalib.net.camera import camera_models
from fastforks.calibration.models.geocalib.net.gravity import Gravity
from fastforks.calibration.models.geocalib.net.lm_optimizer import LMOptimizer, _cholesky_solve_export, optimizer_step
from fastforks.calibration.models.geocalib.net.misc import J_vecnorm, SphericalManifold
from fastforks.calibration.models.geocalib.net.perspective_fields import get_perspective_field


def test_card() -> None:
    card, variant = registry.cards("calibration")["geocalib-pinhole"]
    assert variant.outputs == ("intrinsics", "gravity", "roll", "pitch", "vfov")
    assert card.params["camera_model"] == "pinhole"
    assert card.licence.code == "Apache-2.0"
    assert "geocalib-pinhole" in calibration.list_models()


def test_vecnorm_jacobian_exports_batch_wide_zero_guard() -> None:
    """One zero vector applies the upstream epsilon to the whole batch, including nonzero vectors."""

    class Jacobian(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return J_vecnorm(x)

    normal = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    zero = normal.clone()
    zero[0] = 0
    exported = torch.export.export(Jacobian(), (normal,)).module()
    for x in (normal, zero):
        norm = torch.norm(x, dim=-1, keepdim=True).unsqueeze(-1)
        if (norm == 0).any():
            norm = norm + 1e-6
        expected = torch.eye(3) / norm - torch.einsum("...i,...j->...ij", x, x) / norm**3
        torch.testing.assert_close(J_vecnorm(x), expected, rtol=0, atol=0)
        torch.testing.assert_close(exported(x), expected, rtol=0, atol=0)


def test_householder_exports_small_sigma_guard() -> None:
    """The exported transform applies the same small-sigma adjustment as eager."""

    class Householder(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            return SphericalManifold.householder_vector(x)

    normal = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, -3.0]])
    small = normal.clone()
    small[0] = torch.tensor([0.0, 0.0, 1.0])
    exported = torch.export.export(Householder(), (normal,)).module()
    for x in (normal, small):
        expected = Householder()(x)
        actual = exported(x)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_preprocess_shapes() -> None:
    """Shorter side to 320, crop to multiples of 32; scales and crop map back to the input."""
    x, scales, crop = preprocess(torch.rand(1, 3, 1920, 1440), 320, 32)
    assert x.shape == (1, 3, 416, 320)  # 426 -> 416
    torch.testing.assert_close(scales, torch.tensor([320 / 1440, 426 / 1920]))
    torch.testing.assert_close(crop, torch.tensor([0.0, -10.0]))
    x, _, crop = preprocess(torch.rand(2, 3, 300, 400), 320, 32)  # upscaled: no blur
    assert x.shape == (2, 3, 320, 416)
    assert crop.tolist() == [-10.0, 0.0]


def test_optimizer_step_solves_and_fails_to_zero() -> None:
    torch.manual_seed(0)
    a = torch.randn(2, 3, 3)
    h = a @ a.transpose(1, 2) + torch.eye(3)
    g = torch.randn(2, 3)
    lamb = torch.zeros(2)
    d = optimizer_step(g, h, lamb)
    h_damped = h + h.diagonal(dim1=-2, dim2=-1).mul(0).clamp(min=1e-6).diag_embed()
    torch.testing.assert_close(d, torch.linalg.solve(h_damped, g), atol=1e-5, rtol=1e-5)
    bad = h.clone()
    bad[1] = -torch.eye(3)  # not positive definite: the whole batch takes a zero step (as upstream)
    assert not optimizer_step(g, bad, lamb).any()


def test_exported_optimizer_step_has_no_cholesky_and_fails_to_zero() -> None:
    """The exported step solves without a Cholesky operator (ONNX has none) and keeps the batch-wide zero step."""

    class Step(torch.nn.Module):
        def forward(self, g: torch.Tensor, h: torch.Tensor, lamb: torch.Tensor) -> torch.Tensor:
            return optimizer_step(g, h, lamb)

    torch.manual_seed(0)
    a = torch.randn(4, 3, 3)
    h = a @ a.transpose(1, 2) + 0.1 * torch.eye(3)
    g = torch.randn(4, 3)
    lamb = torch.tensor([0.0, 0.1, 1.0, 100.0])
    program = torch.export.export(Step(), (g, h, lamb), strict=False)
    targets = {str(node.target) for node in program.graph.nodes if node.op == "call_function"}
    assert not [t for t in targets if "cholesky" in t or "linalg" in t], targets
    exported = program.module()
    torch.testing.assert_close(exported(g, h, lamb), optimizer_step(g, h, lamb), atol=1e-5, rtol=1e-5)
    for bad_system in (-torch.eye(3), torch.full((3, 3), float("nan"))):
        bad = h.clone()
        bad[2] = bad_system
        assert not optimizer_step(g, bad, torch.zeros(4)).any()  # eager: LAPACK reports the failed pivot
        assert not exported(g, bad, torch.zeros(4)).any()
    h[2] = 0  # a zero system is damped to eps on the diagonal: solvable, a large step in both
    torch.testing.assert_close(exported(g, h, lamb), optimizer_step(g, h, lamb), atol=1e-5, rtol=1e-5)
    # a 4 x 4 system (simple_radial): the unrolled solve follows the system's size
    a = torch.randn(2, 4, 4)
    h, g = a @ a.transpose(1, 2) + 0.1 * torch.eye(4), torch.randn(2, 4)
    solution, failed = _cholesky_solve_export(g, h)
    assert not failed.any()
    expected = torch.cholesky_solve(g[..., None], torch.linalg.cholesky(h))[..., 0]
    torch.testing.assert_close(solution, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(("num_steps", "stops_early"), [(30, True), (2, False)])
def test_exported_fit_equals_the_early_stopped_fit(monkeypatch, num_steps, stops_early) -> None:
    """The exported fixed-length LM loop gives the eager result, whether upstream's loop stops early or runs out."""
    opt: Any = LMOptimizer({"camera_model": "pinhole", "num_steps": num_steps}).eval()
    override(opt, _no_uncertainty, "estimate_uncertainty")
    runner: Any = SimpleNamespace(net=SimpleNamespace(optimizer=opt))

    class Fit(torch.nn.Module):
        def forward(self, up: torch.Tensor, latitude: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            confidence = torch.ones_like(latitude[:, 0])
            data = dict(zip(_FIELDS, (up, confidence, latitude, confidence), strict=True))
            out = _GeoCalib._fit(runner, data)  # the eager / export switch under test
            return out["camera"]._data, out["gravity"]._data

    size = {"height": torch.full((2,), 48.0), "width": torch.full((2,), 64.0)}
    camera = camera_models["pinhole"].from_dict(size | {"vfov": torch.tensor([0.9, 1.3])}).float()
    gravity = Gravity.from_rp(torch.tensor([0.2, -0.1]), torch.tensor([0.3, 0.15])).float()
    fields = get_perspective_field(camera, gravity)
    steps = []
    monkeypatch.setattr(lm_optimizer, "optimizer_step", lambda *a: steps.append(1) or optimizer_step(*a))
    want = Fit()(*fields)
    assert (len(steps) < num_steps) == stops_early, len(steps)
    got = torch.export.export(Fit(), fields, strict=False).module()(*fields)
    for g, w in zip(got, want, strict=True):
        torch.testing.assert_close(g, w, atol=1e-4, rtol=1e-5)
    if stops_early:  # the fit recovered the camera the fields were rendered from
        torch.testing.assert_close(want[0], camera._data, atol=1e-2, rtol=1e-4)
