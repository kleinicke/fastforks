"""`estimate_absolute_pose`: synthetic scenes with known R, t and outliers; portable vs kernels."""

from __future__ import annotations

import pytest
import torch

from fastforks._core.errors import InputError
from fastforks.matching import AbsolutePose, estimate_absolute_pose
from fastforks.matching import absolute_pose as ap

F = 800.0


def scene(  # noqa: PLR0913, PLR0917 - one scene per case
    p: int, n: int, outliers: float, noise: float = 0.3, seed: int = 0, device: str = "cpu"
) -> tuple:
    """`p` random cameras of `n` 2D-3D matches: pixels, world points, K, true R, t and the inlier mask."""
    gen = torch.Generator(device).manual_seed(seed)
    dev = torch.device(device)
    k = torch.tensor([[F, 0, 400.0], [0, F, 300.0], [0, 0, 1]], device=dev).expand(p, 3, 3)
    aa = torch.randn(p, 3, device=dev, generator=gen)
    z = torch.zeros_like(aa[:, 0])
    skew = torch.stack([z, -aa[:, 2], aa[:, 1], aa[:, 2], z, -aa[:, 0], -aa[:, 1], aa[:, 0], z], 1).view(p, 3, 3)
    r = torch.linalg.matrix_exp(skew)
    t = torch.randn(p, 3, device=dev, generator=gen) * 2
    # Points in front of the camera, mapped back to the world: X = R^T (x_cam - t).
    xc = torch.randn(p, n, 3, device=dev, generator=gen) * torch.tensor([2.0, 2.0, 1.0], device=dev)
    xc = xc + torch.tensor([0.0, 0.0, 6.0], device=dev)
    x = torch.einsum("pji,pnj->pni", r, xc - t[:, None])
    c = torch.tensor([400.0, 300.0], device=dev)
    p2 = xc[..., :2] / xc[..., 2:] * F + c + torch.randn(p, n, 2, device=dev, generator=gen) * noise
    bad = torch.rand(p, n, device=dev, generator=gen) < outliers
    wild = torch.rand(p, n, 2, device=dev, generator=gen) * torch.tensor([800.0, 600.0], device=dev)
    return p2 * ~bad[..., None] + wild * bad[..., None], x, k, r, t, ~bad


def pose_error(pose: AbsolutePose, r: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotation error in degrees and translation error (absolute) per camera (batched or single)."""
    rr, tt = pose.R.reshape(-1, 3, 3), pose.t.reshape(-1, 3)
    e_r = torch.rad2deg(torch.arccos(((torch.einsum("pji,pji->p", rr, r.reshape(-1, 3, 3)) - 1) / 2).clamp(-1, 1)))
    return e_r, (tt - t.reshape(-1, 3)).norm(dim=1)


def test_p3p_exact():
    """Every exact three-point sample has the true pose among its solutions."""
    p2, x, k, r, t, _ = scene(64, 3, outliers=0.0, noise=0.0)
    y = torch.cat(((p2 - k[:, :2, 2][:, None]) / F, torch.ones_like(p2[..., :1])), -1).double()
    y = y / y.norm(dim=-1, keepdim=True)
    m, ok = ap._p3p(y, x.double())
    assert ok.any(1).all()
    truth = torch.cat((r.reshape(64, 9), t), 1).double()[:, None]
    err = (m - truth).abs().amax(-1)
    assert torch.where(ok, err, torch.full_like(err, 1e9)).amin(1).max() < 1e-3  # float32 pixels
    # The solutions reproject their own sample exactly.
    xc = x.double()[:, None] @ m[..., :9].view(64, 4, 3, 3).transpose(-1, -2) + m[..., None, 9:]
    proj = xc[..., :2] / xc[..., 2:] * F + k[:, :2, 2][:, None, None]
    assert torch.where(ok[..., None, None], (proj - p2[:, None].double()).abs(), torch.zeros_like(proj)).max() < 1e-3


@pytest.mark.parametrize(("n", "outliers"), [(200, 0.0), (1000, 0.5)])
def test_estimate_absolute_pose(n, outliers):
    p2, x, k, r, t, inl = scene(3, n, outliers=outliers)
    pose = estimate_absolute_pose(p2, x, k, mode="portable")
    assert pose.ok.all()
    e_r, e_t = pose_error(pose, r, t)
    assert e_r.max() < 0.2
    assert e_t.max() < 0.02
    assert pose.inliers.shape == (3, n)
    assert (pose.inliers & ~inl).sum() <= 0.01 * n * 3  # few outliers slip in
    assert (pose.inliers & inl).sum() >= 0.95 * inl.sum()


def test_unbatched_and_mask():
    p2, x, k, r, t, _ = scene(1, 300, outliers=0.3)
    pose = estimate_absolute_pose(p2[0], x[0], k[0], mode="portable")
    assert pose.R.shape == (3, 3)
    assert pose.t.shape == (3,)
    assert pose.inliers.shape == (300,)
    assert bool(pose.ok)
    mask = torch.arange(300) < 150
    masked = estimate_absolute_pose(p2[0], x[0], k[0], mask=mask, mode="portable")
    assert bool(masked.ok)
    assert not masked.inliers[150:].any()
    assert pose_error(masked, r, t)[0].max() < 0.3


def test_degenerate_inputs():
    p2, x, k, _, _, _ = scene(1, 10, outliers=0.0)
    with pytest.raises(InputError, match="points2d and points3d"):
        estimate_absolute_pose(p2[0], x[0, :5], k[0], mode="portable")
    with pytest.raises(InputError, match="at least 4"):
        estimate_absolute_pose(p2[0, :3], x[0, :3], k[0], mode="portable")
    with pytest.raises(InputError, match="threshold"):
        estimate_absolute_pose(p2[0], x[0], k[0], threshold=0.0, mode="portable")
    # All-outlier input: not ok, identity pose.
    pose = estimate_absolute_pose(p2[0], x[0].roll(1, 0), k[0], mode="portable", threshold=0.01)
    assert not bool(pose.ok)
    assert torch.equal(pose.R, torch.eye(3))


def test_refine_lowers_cost():
    """The LM step from a perturbed true pose lowers the truncated reprojection cost and returns to the truth."""
    p2, x, k, r, t, _ = scene(2, 500, outliers=0.3)
    c = k[:, :2, 2][:, None]
    scorer = ap._TorchScorer((p2 - c) / F, x, torch.ones(2, 500, dtype=torch.bool), torch.full((2,), (2.0 / F) ** 2))
    pairs = torch.arange(2)
    bump = torch.linalg.matrix_exp(ap.rp._cross(torch.tensor([[0.001, -0.0005, 0.0003], [0.0, 0.001, 0.0]])))
    m = torch.cat(((r @ bump).reshape(2, 9), t + 0.003), 1)
    before = scorer.score(pairs, m[:, None], 500)[0][:, 0]
    refined = scorer.refine(pairs, m, 10)
    after = scorer.score(pairs, refined[:, None], 500)[0][:, 0]
    assert (after > before).all()
    e_r, e_t = pose_error(AbsolutePose(refined[:, :9].view(2, 3, 3), refined[:, 9:], None, None), r, t)
    assert e_r.max() < 0.05
    assert e_t.max() < 0.01


@pytest.mark.gpu
def test_kernels_match_portable():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    p2, x, k, r, t, _ = scene(6, 2000, outliers=0.5, device="cuda")
    portable = estimate_absolute_pose(p2, x, k, mode="portable")
    kernels = estimate_absolute_pose(p2, x, k, mode="kernels")
    assert portable.ok.all()
    assert kernels.ok.all()
    for pose in (portable, kernels):
        e_r, e_t = pose_error(pose, r, t)
        assert e_r.max() < 0.2
        assert e_t.max() < 0.02
    assert (portable.inliers == kernels.inliers).float().mean() > 0.95


@pytest.mark.gpu
def test_kernel_p3p_and_scores_match_portable():
    """The P3P kernel returns the torch solutions; the scoring kernels and the polish agree with torch."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from fastforks.kernels import ransac  # noqa: PLC0415 - optional at import time

    p2, x, k, _, _, _ = scene(4, 1500, outliers=0.3, device="cuda")
    c = k[:, :2, 2][:, None]
    valid = torch.ones(4, 1500, dtype=torch.bool, device="cuda")
    thr2 = torch.full((4,), (2.0 / F) ** 2, device="cuda")
    torch_scorer = ap._TorchScorer((p2 - c) / F, x, valid, thr2)
    kernel_scorer = ap._KernelScorer(ransac.load(x.device), (p2 - c) / F, x, valid, thr2)
    pairs = torch.arange(4, device="cuda")
    gen = torch.Generator("cuda").manual_seed(1)
    idx = (torch.rand(4, 100, 3, device="cuda", generator=gen) * 1500).int()
    ma, oka = torch_scorer.hypotheses(pairs, idx)
    mb, okb = kernel_scorer.hypotheses(pairs, idx)
    # Same solution sets (the order of the four may differ): every real torch solution has a kernel twin.
    d = (ma.view(4, 100, 4, 1, 12) - mb.view(4, 100, 1, 4, 12)).abs().amax(-1).amin(-1).view(4, 400)
    assert (torch.where(oka, d, torch.zeros_like(d)) < 1e-3).all()
    assert (oka.sum() - okb.sum()).abs() <= 0.02 * oka.numel()
    for n in (256, 1500):  # the many- and the few-candidate scoring kernels
        for cand in (ma[:, :64], ma[:, :8]):
            sa, ca = torch_scorer.score(pairs, cand, n)
            sb, cb = kernel_scorer.score(pairs, cand, n)
            assert torch.allclose(sa, sb, atol=0.5, rtol=1e-3)
            assert (ca - cb).abs().max() <= 1
    best = ma[torch.arange(4, device="cuda"), torch_scorer.score(pairs, ma, 1500)[0].argmax(1)]
    score, count = (v[:, 0] for v in torch_scorer.score(pairs, best[:, None], 1500))
    fa, ia, oa = torch_scorer.finish(best, score, count)
    fb, ib, ob = kernel_scorer.finish(best, score, count)
    assert oa.all()
    assert ob.all()
    assert (fa - fb).abs().max() < 1e-3
    assert (ia == ib).float().mean() > 0.99
