"""The export-friendly nearest rotation against the SVD it replaces."""

from __future__ import annotations

import torch

from fastforks._core.export_linalg import nearest_rotation


def _svd_rotation(b: torch.Tensor) -> torch.Tensor:
    u, _, v = torch.svd(b.transpose(-1, -2))
    det = torch.det(v @ u.transpose(-2, -1))
    return torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2) @ u.transpose(-2, -1)


def test_nearest_rotation_matches_the_svd() -> None:
    g = torch.Generator().manual_seed(0)
    for b in (torch.randn(4000, 3, 3, generator=g), 50 * torch.randn(500, 3, 3, generator=g)):
        r = nearest_rotation(b)
        assert (r - _svd_rotation(b)).abs().max() < 1e-4
        assert (torch.linalg.det(r) - 1).abs().max() < 1e-5
        assert (r @ r.transpose(-1, -2) - torch.eye(3)).abs().max() < 1e-5


def test_nearest_rotation_exports() -> None:
    class M(torch.nn.Module):
        def forward(self, b: torch.Tensor) -> torch.Tensor:
            return nearest_rotation(b)

    ep = torch.export.export(M(), (torch.randn(2, 3, 3),))
    targets = {str(n.target) for n in ep.graph.nodes if n.op == "call_function"}
    assert not any(op in t for t in targets for op in ("svd", "eig", "det", "qr"))
