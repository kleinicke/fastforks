"""ReSplat: the card, the network size, the kNN, the SH rotation, quaternions and the plain-PyTorch rasterizer."""

from __future__ import annotations

import torch

from fastforks._core import registry
from fastforks.multiview.models.resplat import model, render
from fastforks.multiview.models.resplat.net.point_transformer import knn_query


def _rotations(n: int, seed: int = 0) -> torch.Tensor:
    q = torch.linalg.qr(torch.randn(n, 3, 3, generator=torch.Generator().manual_seed(seed), dtype=torch.float64))[0]
    return q * torch.linalg.det(q).sign()[:, None, None]


def _params(**kw: object) -> model.Params:
    base = {
        "vit": "vits", "channels": 256, "latent": 4, "fixed_latent": True, "gaussians_per_point": 1,
        "same_points": False, "lowest": 8, "size_multiple": 32, "train_size": (256, 448), "near": 0.01, "far": 200.0,
        "long_side": 0, "refine": 2,
    }  # fmt: skip
    return model.Params(**(base | kw))  # type: ignore[arg-type]


def test_card() -> None:
    cards = registry.cards("multiview")
    card, variant = cards["resplat-small-dl3dv"]
    assert "gaussians" in variant.outputs
    assert card.licence.weights == "MIT"
    base, base_variant = cards["resplat-base-re10k"]
    assert base.variant_licence(base_variant).weights == "CC-BY-NC-4.0"


def test_network_size() -> None:
    assert model.network_size(256, 448, _params()) == (256, 448)
    assert model.network_size(120, 160, _params(long_side=256)) == (192, 256)
    assert model.network_size(4, 1000, _params(long_side=128, size_multiple=16)) == (16, 128)


def test_knn_matches_exact_distances() -> None:
    """The pooled search finds the exact neighbour distances (a far dense cluster, sparse points)."""
    g = torch.Generator().manual_seed(0)
    p = torch.cat((torch.randn(1500, 3, generator=g) * 1e-3 + 50.0, torch.randn(500, 3, generator=g)))
    d = ((p[:, None] - p[None]) ** 2).sum(-1)
    want = d.topk(8, dim=1, largest=False)[0]  # the fp32 grid at 50 makes ties: compare distances, not indices
    for pool, margin in ((64, 16), (4096, 2), (4096, 16)):  # 64 / margin 2: many queries redone over all points
        assert torch.equal(d.gather(1, knn_query(8, p, rows=32, pool=pool, margin=margin)).sort(1)[0], want)


def test_sh_rotation_keeps_colours() -> None:
    """Coefficients `c @ D(R)` evaluated at `R d` give the colour of `c` at `d`, for every degree."""
    r = _rotations(3)
    c = torch.randn(3, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    d = torch.nn.functional.normalize(torch.randn(3, 50, 3, dtype=torch.float64), dim=-1)
    for i in range(3):
        want = render.sh_basis(d[i]) @ c[i]
        got = render.sh_basis(d[i] @ r[i].T) @ (c[i] @ model.sh_rotation(r[i]))
        torch.testing.assert_close(got, want)


def test_quaternion_matrix() -> None:
    """Upstream's `x y z w` convention, and the inverse of `matrix_quaternion` (`w x y z`)."""
    r = _rotations(8)
    q = model.matrix_quaternion(r)[:, [1, 2, 3, 0]]
    torch.testing.assert_close(model.quaternion_matrix(q), r, atol=1e-7, rtol=0)


def test_render_single_gaussian() -> None:
    """One opaque, wide Gaussian in front of the camera covers the centre with its degree-0 colour."""
    sh = torch.zeros(1, 3, 16)
    sh[0, :, 0] = torch.tensor([0.2, -0.1, 0.3]) / 0.2820947917738781
    k = torch.tensor([[[1.0, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]])
    img = render.render(
        torch.tensor([[0.0, 0, 2]]), torch.eye(3)[None] * 0.25, torch.tensor([1.0]), sh, torch.eye(4)[None], k,
        32, 32, 0.01, 100.0,
    )  # fmt: skip
    torch.testing.assert_close(img[0, :, 16, 16], torch.tensor([0.7, 0.4, 0.8]) * 0.99, atol=5e-3, rtol=0)
