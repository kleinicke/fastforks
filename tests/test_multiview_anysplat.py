"""AnySplat: the card, the voxel fusion against a per-voxel loop of upstream's formula, and the Gaussians PLY export."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from fastforks import multiview
from fastforks._core import registry
from fastforks._core.errors import InputError
from fastforks._core.outputs import NumpyOutputs
from fastforks.multiview._gaussians import pop_gaussians
from fastforks.multiview.models.anysplat import model


def test_card() -> None:
    card, variant = registry.cards("multiview")["anysplat"]
    assert "gaussians" in variant.outputs
    assert card.licence.weights == "MIT"
    assert card.options["voxel_size"] == pytest.approx(0.002)


def test_voxelize_matches_loop() -> None:
    """Upstream `voxelizaton_with_fusion`: per voxel, the points and features weighted by the softmax of `conf`."""
    g = torch.Generator().manual_seed(0)
    pts = torch.rand(200, 3, generator=g) * 0.015
    feats = torch.randn(200, 5, generator=g)
    conf = torch.randn(200, generator=g) * 3
    vp, vf = model.voxelize(pts, feats, conf, 0.005)
    keys = (pts / 0.005).round().int()
    uniq = torch.unique(keys, dim=0)
    assert vp.shape == (len(uniq), 3)
    for i, key in enumerate(uniq):
        sel = (keys == key).all(1)
        e = torch.exp(conf[sel] - conf[sel].max())
        w = (e / (e.sum() + 1e-6))[:, None]
        torch.testing.assert_close(vp[i], (pts[sel] * w).sum(0), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(vf[i], (feats[sel] * w).sum(0), atol=1e-6, rtol=1e-5)


def test_gaussians_fields_and_ply(tmp_path) -> None:
    raw = torch.randn(10, model.GS_DIM + 1)
    out = model.gaussians(raw, torch.randn(10, 3), 0.0)
    g = pop_gaussians(out)
    assert out == {}
    assert isinstance(g, multiview.Gaussians)
    assert len(g) == 10
    assert g.sh_degree == model.SH_DEGREE
    torch.testing.assert_close(g.rotations.norm(dim=1), torch.ones(10))
    assert bool((g.opacities > 0).all() and (g.opacities < 1).all())
    g.save(tmp_path / "s.ply")
    data = (tmp_path / "s.ply").read_bytes()
    head, body = data.split(b"end_header\n", 1)
    names = [line.split()[-1] for line in head.decode().splitlines() if line.startswith("property")]
    assert len(names) == 3 + 3 + 3 * model.SH + 1 + 3 + 4
    rows = np.frombuffer(body, "<f4").reshape(10, len(names))
    np.testing.assert_allclose(rows[:, names.index("rot_0")], g.rotations[:, 0].numpy(), rtol=1e-6)
    np.testing.assert_allclose(1 / (1 + np.exp(-rows[:, names.index("opacity")])), g.opacities.numpy(), rtol=1e-5)
    np.testing.assert_allclose(np.exp(rows[:, names.index("scale_0")]), g.scales[:, 0].numpy(), rtol=1e-5)
    with pytest.raises(InputError):
        g.save(tmp_path / "s.npz")


@dataclass(frozen=True)
class _Result(NumpyOutputs):
    gaussians: multiview.Gaussians | None


def test_result_moves_dataclass_fields() -> None:
    """`.to()` / `.cpu()` of any result move the tensors of a dataclass field (`MultiViewResult.gaussians`)."""
    g = pop_gaussians(model.gaussians(torch.randn(4, model.GS_DIM + 1), torch.randn(4, 3), 0.0))
    moved = _Result(g).to("meta")
    assert isinstance(moved.gaussians, multiview.Gaussians)
    assert {getattr(moved.gaussians, k).device.type for k in multiview.Gaussians.__dataclass_fields__} == {"meta"}
    assert _Result(None).cpu().gaussians is None
    assert _Result(g).numpy() == {}
