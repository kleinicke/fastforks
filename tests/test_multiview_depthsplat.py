"""DepthSplat: the card, the SH rotation and quaternion helpers, and the intrinsics normalisation."""

from __future__ import annotations

import math
import types

import pytest
import torch

from fastforks._core import registry
from fastforks.multiview._gaussians import pop_gaussians
from fastforks.multiview.models.depthsplat import model


def _rotations(n: int, seed: int = 0) -> torch.Tensor:
    q = torch.linalg.qr(torch.randn(n, 3, 3, generator=torch.Generator().manual_seed(seed), dtype=torch.float64))[0]
    return q * torch.linalg.det(q).sign()[:, None, None]


def test_card() -> None:
    card, variant = registry.cards("multiview")["depthsplat-small-re10k"]
    assert "gaussians" in variant.outputs
    assert card.licence.weights == "MIT"
    assert card.options["long_side"] == 256


def test_network_size() -> None:
    assert model.network_size(120, 160, 256) == (192, 256)
    assert model.network_size(256, 256, 256) == (256, 256)
    assert model.network_size(4, 1000, 256) == (16, 256)


def test_sh_rotation_is_a_representation() -> None:
    """`D(R1 R2) = D(R1) D(R2)`, `D` orthogonal, `D_1 = R` and the degree-2 block maps `Y_2(s)` to `Y_2(R s)`."""
    r = _rotations(6)
    d = model.sh_rotation(r)
    torch.testing.assert_close(model.sh_rotation(r[:3] @ r[3:]), d[:3] @ d[3:])
    torch.testing.assert_close(d @ d.transpose(1, 2), torch.eye(model.SH, dtype=torch.float64).expand(6, -1, -1))
    torch.testing.assert_close(d[:, 1:4, 1:4], r)
    s = torch.nn.functional.normalize(torch.randn(6, 3, dtype=torch.float64), dim=1)
    torch.testing.assert_close(d[:, 4:, 4:] @ model._y2(s)[..., None], model._y2((r @ s[..., None])[..., 0])[..., None])


def _matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        -1,
    ).unflatten(-1, (3, 3))  # fmt: skip


def test_quaternions() -> None:
    """`matrix_quaternion` inverts the rotation matrix of a quaternion; the product composes rotations."""
    r = torch.cat((_rotations(8), torch.diag_embed(torch.tensor([[1.0, -1, -1], [-1, 1, -1], [-1, -1, 1]]).double())))
    q = model.matrix_quaternion(r)
    torch.testing.assert_close(_matrix(q), r)
    torch.testing.assert_close(_matrix(model.quaternion_product(q[:4], q[4:8])), r[:4] @ r[4:8])


def test_normalized_intrinsics_are_resize_invariant() -> None:
    k = torch.tensor([[[100.0, 0, 79.5], [0, 90, 59.5], [0, 0, 1]]])
    kn = model.normalized_intrinsics(k, 120, 160)
    torch.testing.assert_close(kn[0, :2, 2], torch.tensor([0.5, 0.5]))
    k2 = k.clone()
    k2[:, :2] *= 2
    k2[:, :2, 2] += 0.5  # the same camera at twice the size (pixel centres at integers)
    torch.testing.assert_close(model.normalized_intrinsics(k2, 240, 320), kn)


def test_gaussians_fields() -> None:
    v, h, w = 2, 4, 6
    raw = torch.randn(v, model.RAW_DIM, h, w)
    k = model.normalized_intrinsics(torch.tensor([[50.0, 0, 2.5], [0, 50, 1.5], [0, 0, 1]]).expand(v, 3, 3), h, w)
    c2w = torch.eye(4).repeat(v, 1, 1)
    c2w[:, :3, :3] = _rotations(v).float()
    depth = torch.full((v, h, w), 2.0)
    out = model.gaussians(raw, depth, torch.rand(v, 3, h, w), k, c2w)
    g = pop_gaussians(out)
    assert g is not None
    assert len(g) == v * h * w
    assert g.sh_degree == model.SH_DEGREE
    torch.testing.assert_close(g.rotations.norm(dim=1), torch.ones(v * h * w))
    assert bool((g.rotations[:, 0] >= 0).all())
    # every centre lies at depth 2 along its camera's optical axis
    z = ((g.means.view(v, -1, 3) - c2w[:, None, :3, 3]) @ c2w[:, :3, :3])[..., 2]
    torch.testing.assert_close(z, torch.full_like(z, 2.0))


@pytest.mark.parametrize("deg", [1, 2])
def test_sh_mask(deg: int) -> None:
    """Upstream `sh_mask`: 1 for degree 0, `0.1 * 0.25 ** l` for degree `l` (applied to the raw coefficients)."""
    raw = torch.zeros(1, model.RAW_DIM, 1, 1)
    raw[0, 10 + deg**2] = 1.0  # red, first coefficient of degree `deg`
    k = model.normalized_intrinsics(torch.tensor([[[1.0, 0, 0], [0, 1, 0], [0, 0, 1]]]), 1, 1)
    out = model.gaussians(raw, torch.ones(1, 1, 1), torch.full((1, 3, 1, 1), 0.5), k, torch.eye(4)[None])
    assert out["gaussians/harmonics"][0, 0, deg**2].item() == pytest.approx(0.1 * 0.25**deg)
    assert math.isclose(out["gaussians/harmonics"][0, 0, 0].item(), 0.0, abs_tol=1e-7)


class _StubNet(torch.nn.Module):
    """Depth from the scene's scale (near and the baseline), so a missing normalisation shows."""

    def forward(
        self, x: torch.Tensor, _k: torch.Tensor, c2w: torch.Tensor, near: torch.Tensor, far: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, v, _, h, w = x.shape
        base = (c2w[0, :, :3, 3] - c2w[0, :1, :3, 3]).norm(dim=1)[:, None, None]
        depth = (near[0, :, None, None] + far[0, :, None, None] + base) * x[0, :, 0] + 0.1
        raw = torch.zeros(v, model.RAW_DIM, h, w)
        raw[:, 3:6] = x[0, :, :1]
        return depth, torch.ones(v, 1, h, w), raw


def test_scene_scale_normalised() -> None:
    """The scene is scaled to near = `NEAR`: scaling cameras and depth range scales depth, centres and scales."""
    runner = object.__new__(model.Portable)
    runner.ctx = types.SimpleNamespace(params=types.SimpleNamespace(long_side=16), device=torch.device("cpu"))
    runner.net, runner.half = _StubNet(), False
    imgs = torch.rand(2, 3, 16, 16, generator=torch.Generator().manual_seed(0))
    k = torch.tensor([[20.0, 0, 7.5], [0, 20, 7.5], [0, 0, 1]]).expand(2, 3, 3)
    c2w = torch.eye(4).repeat(2, 1, 1)
    c2w[1, 0, 3] = 0.3
    a = runner(imgs, k, c2w, torch.tensor([2.0, 20.0]))
    c2w[:, :3, 3] *= 500
    b = runner(imgs, k, c2w, torch.tensor([1000.0, 10000.0]))
    for key in ("depth", "gaussians/means", "gaussians/scales"):
        torch.testing.assert_close(b[key], a[key] * 500, rtol=1e-5, atol=0)
