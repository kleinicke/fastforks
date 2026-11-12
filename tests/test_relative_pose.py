"""`estimate_pose` and `estimate_fundamental`: synthetic scenes with known R, t and outliers; portable vs kernels."""

from __future__ import annotations

import pytest
import torch

from fastforks._core.errors import InputError
from fastforks._core.kernels import KernelUnavailableError
from fastforks.matching import MatchingResult, RelativePose, estimate_fundamental, estimate_pose
from fastforks.matching import relative_pose as rp

F = 800.0


def scene(  # noqa: PLR0913, PLR0917 - one scene per case
    p: int, n: int, outliers: float, noise: float = 0.3, seed: int = 0, device: str = "cpu"
) -> tuple:
    """`p` random pairs of `n` matches: pixels, K, true R, t (camera 1 to 2) and the inlier mask."""
    gen = torch.Generator(device).manual_seed(seed)
    dev = torch.device(device)
    k = torch.tensor([[F, 0, 400.0], [0, F, 300.0], [0, 0, 1]], device=dev).expand(p, 3, 3)
    aa = torch.randn(p, 3, device=dev, generator=gen) * 0.3
    z = torch.zeros_like(aa[:, 0])
    skew = torch.stack([z, -aa[:, 2], aa[:, 1], aa[:, 2], z, -aa[:, 0], -aa[:, 1], aa[:, 0], z], 1).view(p, 3, 3)
    r = torch.linalg.matrix_exp(skew)
    t = torch.randn(p, 3, device=dev, generator=gen)
    t = t / t.norm(dim=1, keepdim=True)
    x = torch.randn(p, n, 3, device=dev, generator=gen) * torch.tensor([2.0, 2.0, 1.0], device=dev)
    x = x + torch.tensor([0.0, 0.0, 6.0], device=dev)
    x2 = torch.einsum("pij,pnj->pni", r, x) + t[:, None]
    c = torch.tensor([400.0, 300.0], device=dev)
    p1 = x[..., :2] / x[..., 2:] * F + c + torch.randn(p, n, 2, device=dev, generator=gen) * noise
    p2 = x2[..., :2] / x2[..., 2:] * F + c + torch.randn(p, n, 2, device=dev, generator=gen) * noise
    bad = torch.rand(p, n, device=dev, generator=gen) < outliers
    wild = torch.rand(p, n, 2, device=dev, generator=gen) * torch.tensor([800.0, 600.0], device=dev)
    return p1, p2 * ~bad[..., None] + wild * bad[..., None], k, r, t, ~bad


def pose_error(pose: RelativePose, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """max(rotation, translation-direction angle) in degrees per pair."""
    e_t = torch.rad2deg(torch.arccos((pose.t * t).sum(-1).abs().clamp(max=1)))
    e_r = torch.rad2deg(torch.arccos(((torch.einsum("pji,pji->p", pose.R, r) - 1) / 2).clamp(-1, 1)))
    return torch.maximum(e_t, e_r)


def true_fundamental(k: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """`F = K^-T [t]_x R K^-1` `[P, 3, 3]`, unit norm."""
    ki = torch.linalg.inv(k)
    f = ki.transpose(1, 2) @ rp._cross(t) @ r @ ki
    return f / f.flatten(1).norm(dim=1)[:, None, None]


def f_error(f: torch.Tensor, f_true: torch.Tensor) -> torch.Tensor:
    """Distance of unit-norm `[P, 3, 3]` matrices up to sign."""
    a, b = f.flatten(1), f_true.flatten(1)
    return torch.minimum((a - b).norm(dim=1), (a + b).norm(dim=1))


def test_seven_point_exact():
    """Noise-free seven-point samples: one of the solutions is the true F (in Hartley-normalised coordinates)."""
    p1, p2, k, r, t, _ = scene(64, 7, outliers=0.0, noise=0.0)
    x1, x2 = rp._normalise(p1, k).double(), rp._normalise(p2, k).double()
    f, ok = rp._seven_point(x1, x2)
    f_true = (rp._cross(t) @ r).double().reshape(64, 1, 9)
    f_true = f_true / f_true.norm(dim=-1, keepdim=True)
    dist = torch.minimum((f - f_true).norm(dim=-1), (f + f_true).norm(dim=-1))
    assert ok.any(1).all()
    assert torch.where(ok, dist, 9.0).amin(1).max() < 1e-3  # the scene is float32
    # Every solution satisfies the seven constraints and is singular.
    res = torch.einsum("bni,bsi->bsn", rp._kron(x1, x2), f).abs()
    assert torch.where(ok[..., None], res, 0.0).max() < 1e-9
    assert torch.where(ok, torch.linalg.det(f.view(64, 3, 3, 3)).abs(), 0.0).max() < 1e-9


@pytest.mark.parametrize(("n", "outliers"), [(300, 0.0), (1000, 0.5)])
def test_estimate_fundamental(n, outliers):
    p1, p2, k, r, t, inlier = scene(3, n, outliers=outliers)
    fund = estimate_fundamental(p1, p2, threshold=1.0)
    assert fund.F.shape == (3, 3, 3)
    assert fund.ok.all()
    assert f_error(fund.F, true_fundamental(k, r, t)).max() < 2e-3
    assert (fund.inliers & inlier).sum(1).min() > 0.9 * inlier.sum(1).min()
    assert (fund.inliers & ~inlier).sum(1).max() < 0.05 * n
    one = estimate_fundamental(p1[0], p2[0], mask=inlier[0])
    assert one.F.shape == (3, 3)
    assert one.ok
    assert not (one.inliers & ~inlier[0]).any()
    with pytest.raises(InputError):
        estimate_fundamental(p1[:, :6], p2[:, :6])


def test_five_point_exact():
    """Exact 5-point data: the true E is among the real solutions, all solutions satisfy the constraints."""
    p1, p2, k, r, t, _ = scene(32, 5, outliers=0.0, noise=0.0)
    x1, x2 = rp._normalise(p1, k), rp._normalise(p2, k)
    e, ok = rp._five_point(x1.double(), x2.double())
    kron = rp._kron(x1, x2).double()
    assert ok.any(dim=1).all()
    assert torch.einsum("bni,bci->bcn", kron, e)[ok].abs().max() < 1e-6
    tx = torch.zeros(32, 3, 3)
    tx[:, 0, 1], tx[:, 0, 2], tx[:, 1, 0] = -t[:, 2], t[:, 1], t[:, 2]
    tx[:, 1, 2], tx[:, 2, 0], tx[:, 2, 1] = -t[:, 0], -t[:, 1], t[:, 0]
    true = (tx @ r).reshape(32, 9)
    true = true / true.norm(dim=1, keepdim=True)
    d = torch.minimum((e.float() - true[:, None]).norm(dim=-1), (e.float() + true[:, None]).norm(dim=-1))
    assert torch.where(ok, d, 9.0).amin(1).max() < 1e-3


@pytest.mark.parametrize(("n", "outliers"), [(300, 0.0), (1000, 0.5)])
def test_estimate_pose(n, outliers):
    p1, p2, k, r, t, inlier = scene(4, n, outliers)
    pose = estimate_pose(p1, p2, k, k)
    assert pose.ok.all()
    assert pose_error(pose, r, t).max() < 1.0
    assert pose.inliers.shape == (4, n)
    precision = (pose.inliers & inlier).sum(1) / pose.inliers.sum(1)
    assert precision.min() > 0.98
    assert ((pose.inliers & inlier).sum(1) / inlier.sum(1)).min() > 0.7


def test_unbatched_and_mask():
    p1, p2, k, r, t, inlier = scene(1, 400, outliers=0.3)
    pose = estimate_pose(p1[0], p2[0], k[0], k[0])
    assert pose.R.shape == (3, 3)
    assert pose.t.shape == (3,)
    assert pose.inliers.shape == (400,)
    assert (
        pose_error(RelativePose(pose.R[None], pose.t[None], pose.E[None], pose.inliers[None], pose.ok[None]), r, t)
        < 1.0
    )
    # masked-out matches are never inliers; the result still holds with the outliers masked away
    masked = estimate_pose(p1[0], p2[0], k[0], k[0], mask=inlier[0])
    assert masked.ok
    assert not masked.inliers[~inlier[0]].any()


def test_degenerate_inputs():
    p1, p2, k, *_ = scene(2, 50, outliers=0.0)
    with pytest.raises(InputError):
        estimate_pose(p1[:, :4], p2[:, :4], k, k)
    with pytest.raises(InputError):
        estimate_pose(p1, p2[:, :40], k, k)
    with pytest.raises(InputError):
        estimate_pose(p1, p2, k, k, threshold=0.0)
    with pytest.raises(InputError):
        estimate_pose(p1, p2, k, k, mask=torch.zeros(2, 50, dtype=torch.bool))
    # all outliers: no pose, flagged, identity returned
    bad = torch.rand(2, 50, 2) * 500
    pose = estimate_pose(p1, bad, k, k)
    assert pose.R.shape == (2, 3, 3)
    assert pose.inliers.sum() <= 2 * 10
    if not pose.ok.all():
        assert torch.allclose(pose.R[~pose.ok], torch.eye(3).expand(int((~pose.ok).sum()), 3, 3))
    if not torch.cuda.is_available():
        with pytest.raises(KernelUnavailableError):
            estimate_pose(p1, p2, k, k, mode="kernels")


@pytest.mark.gpu
def test_kernels_match_portable_fundamental():
    p1, p2, k, r, t, _ = scene(6, 1500, outliers=0.5, device="cuda")
    f_true = true_fundamental(k, r, t)
    portable = estimate_fundamental(p1, p2, mode="portable")
    kernels = estimate_fundamental(p1, p2, mode="kernels")
    assert portable.ok.all()
    assert kernels.ok.all()
    assert f_error(portable.F, f_true).max() < 1e-2
    assert f_error(kernels.F, f_true).max() < 1e-2
    # Different random hypotheses: one rank-2 refit from the best of them leaves the two backends a few 1e-3 apart.
    assert (portable.inliers == kernels.inliers).float().mean() > 0.9


@pytest.mark.gpu
def test_kernels_match_portable():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    p1, p2, k, r, t, _ = scene(6, 2000, outliers=0.5, device="cuda")
    portable = estimate_pose(p1, p2, k, k, mode="portable")
    kernels = estimate_pose(p1, p2, k, k, mode="kernels")
    assert portable.ok.all()
    assert kernels.ok.all()
    assert pose_error(kernels, r, t).max() < 1.0
    assert (pose_error(kernels, r, t) - pose_error(portable, r, t)).abs().max() < 0.5
    agree = (portable.inliers == kernels.inliers).float().mean()
    assert agree > 0.9  # different random hypotheses: the inlier sets differ at the threshold margin


def test_matches_from_result():
    """`MatchingResult.matches` draws certain pixels and their warped positions, in the form `estimate_pose` takes."""
    h, w = 20, 30
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    warp = torch.stack((x + 3.0, y - 2.0))  # a pure shift
    certainty = torch.zeros(h, w)
    certainty[5:15, 10:25] = 0.9
    result = MatchingResult(warp, certainty, "test@1", "portable")
    p1, p2 = result.matches(100)
    assert p1.shape == p2.shape == (100, 2)
    assert (certainty[p1[:, 1].long(), p1[:, 0].long()] > 0).all()
    assert torch.equal(p2, p1 + torch.tensor([3.0, -2.0]))
    assert len(torch.unique(p1, dim=0)) == 100
    p1, p2 = result.matches(1000)  # capped at the 150 certain pixels
    assert p1.shape == (150, 2)
    b1, b2 = MatchingResult(
        warp[None].expand(2, 2, h, w), certainty[None].expand(2, h, w), "test@1", "portable"
    ).matches(10)
    assert b1.shape == b2.shape == (2, 10, 2)
    assert torch.equal(b1[0], result.matches(10)[0])
    none = MatchingResult(warp, None, "test@1", "portable").matches(5)[0]
    assert none.shape == (5, 2)
    with pytest.raises(InputError):
        MatchingResult(warp, torch.zeros(h, w), "test@1", "portable").matches(5)


def _scorer_inputs(device: str, p: int = 4, n: int = 600) -> tuple:
    """A scene in normalised coordinates with a perturbed essential matrix to refine."""
    p1, p2, k, r, t, _ = scene(p, n, outliers=0.5, device=device)
    x1, x2 = rp._normalise(p1, k), rp._normalise(p2, k)
    valid = torch.ones(p, n, dtype=torch.bool, device=x1.device)
    thr2 = torch.full((p,), (3.0 / F) ** 2, device=x1.device)
    gen = torch.Generator(x1.device).manual_seed(1)
    r_p = r @ rp._rodrigues(torch.randn(p, 3, device=x1.device, generator=gen) * 0.003)
    e = (rp._cross(t + torch.randn(p, 3, device=x1.device, generator=gen) * 0.01) @ r_p).reshape(p, 9)
    e_true = (rp._cross(t) @ r).reshape(p, 9)
    return x1, x2, valid, thr2, e, e_true


def test_refine_lowers_cost():
    x1, x2, valid, thr2, e, e_true = _scorer_inputs("cpu")
    scorer = rp._TorchScorer(x1, x2, valid, thr2)
    pairs = torch.arange(4)
    before, true = scorer.cost(pairs, e), scorer.cost(pairs, e_true)
    after = scorer.cost(pairs, scorer.refine(pairs, e, 10))
    assert (after < before).all()
    assert (after < true * 1.05).all()  # within noise of the true pose
    # the kernel's terms agree with the torch chain's cost definition
    jtj, jtr, cost = scorer.lm_terms(pairs, e, torch.randn(4, 9, 5))
    assert torch.allclose(cost, before)
    assert torch.allclose(jtj, jtj.transpose(1, 2))
    assert jtr.shape == (4, 5)


@pytest.mark.gpu
def test_kernel_refine_matches_portable():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from fastforks.kernels import ransac  # noqa: PLC0415 - optional at import time

    x1, x2, valid, thr2, e, _ = _scorer_inputs("cuda", n=2000)
    pairs = torch.arange(4, device="cuda")
    torch_scorer = rp._TorchScorer(x1, x2, valid, thr2)
    kernel_scorer = rp._KernelScorer(ransac.load(x1.device), x1, x2, valid, thr2)
    a, b = torch_scorer.refine(pairs, e, 10), kernel_scorer.refine(pairs, e, 10)
    assert torch.allclose(torch_scorer.cost(pairs, a), kernel_scorer.cost(pairs, b), rtol=1e-3)
    assert torch.minimum((a - b).norm(dim=1), (a + b).norm(dim=1)).max() < 1e-3


@pytest.mark.gpu
def test_kernel_finish_matches_portable():
    """The fused polish kernel (refits, recoverPose, LM, mask) agrees with the torch chain."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from fastforks.kernels import ransac  # noqa: PLC0415 - optional at import time

    x1, x2, valid, thr2, e, e_true = _scorer_inputs("cuda", n=2000)
    pairs = torch.arange(4, device="cuda")
    torch_scorer = rp._TorchScorer(x1, x2, valid, thr2)
    kernel_scorer = rp._KernelScorer(ransac.load(x1.device), x1, x2, valid, thr2)
    score, count = (v[:, 0] for v in torch_scorer.score(pairs, e[:, None], 2000))
    ra, ta, ea, ia, oka = torch_scorer.finish(e, score, count, rp.MIN_MATCHES)
    rb, tb, eb, ib, okb = kernel_scorer.finish(e, score, count, rp.MIN_MATCHES)
    assert oka.all()
    assert okb.all()
    assert torch.minimum((ea - eb).norm(dim=1), (ea + eb).norm(dim=1)).max() < 1e-3
    assert (ra - rb).abs().max() < 1e-3
    assert (ta - tb).abs().max() < 1e-3
    assert (ia == ib).float().mean() > 0.99
    # both close to the true pose
    e_true = e_true / e_true.norm(dim=1, keepdim=True)
    assert torch.minimum((eb - e_true).norm(dim=1), (eb + e_true).norm(dim=1)).max() < 2e-2
