"""`mode="mlx"` of the pose estimators (`kernels/ransac_mlx`): agreement with portable, shapes, degenerate inputs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from fastforks._core.errors import InputError
from fastforks.matching import estimate_absolute_pose, estimate_fundamental, estimate_pose
from fastforks.matching import relative_pose as rp

sys.path.insert(0, str(Path(__file__).parent))
import test_absolute_pose as tap
from test_relative_pose import f_error, pose_error, scene, true_fundamental
from test_torch_free import _blocked, _ok

pytestmark = [
    pytest.mark.mlx,
    pytest.mark.skipif(
        sys.platform != "darwin" or importlib.util.find_spec("mlx") is None, reason="needs Apple silicon and mlx"
    ),
]


def test_essential_matches_portable():
    p1, p2, k, r, t, inlier = scene(6, 2000, outliers=0.5)
    portable = estimate_pose(p1, p2, k, k, threshold=0.5, mode="portable")
    mlx = estimate_pose(p1, p2, k, k, threshold=0.5, mode="mlx")
    assert isinstance(mlx.R, torch.Tensor)
    assert mlx.R.dtype == torch.float32
    assert mlx.ok.all()
    assert pose_error(mlx, r, t).max() < 1.0
    assert (pose_error(mlx, r, t) - pose_error(portable, r, t)).abs().max() < 0.5
    assert (portable.inliers == mlx.inliers).float().mean() > 0.9
    assert ((mlx.inliers & inlier).sum(1) / mlx.inliers.sum(1)).min() > 0.98
    # E, R and t are consistent: E ~ [t]x R, R a rotation
    assert torch.allclose(mlx.R @ mlx.R.transpose(-1, -2), torch.eye(3).expand(6, 3, 3), atol=1e-4)
    e = rp._cross(mlx.t) @ mlx.R
    e = e / e.flatten(1).norm(dim=1)[:, None, None]
    sign = torch.sign((e * mlx.E).sum((1, 2)))[:, None, None]
    assert (e * sign - mlx.E).abs().max() < 1e-3


def test_fundamental_matches_portable():
    p1, p2, k, r, t, _ = scene(6, 1500, outliers=0.5)
    f_true = true_fundamental(k, r, t)
    portable = estimate_fundamental(p1, p2, mode="portable")
    mlx = estimate_fundamental(p1, p2, mode="mlx")
    assert mlx.ok.all()
    # the unit-norm F distance is ill-conditioned on some pairs (portable is off by as much there): compare to it
    assert (f_error(mlx.F, f_true) - f_error(portable.F, f_true)).max() < 1e-2
    assert (portable.inliers == mlx.inliers).float().mean() > 0.9
    assert torch.linalg.det(mlx.F.double()).abs().max() < 1e-6  # rank 2


def test_absolute_matches_portable():
    p2, x, k, r, t, inlier = tap.scene(6, 1000, outliers=0.5)
    portable = estimate_absolute_pose(p2, x, k, mode="portable")
    mlx = estimate_absolute_pose(p2, x, k, mode="mlx")
    assert mlx.ok.all()
    e_r, e_t = tap.pose_error(mlx, r, t)
    assert e_r.max() < 0.5
    assert e_t.max() < 0.05
    assert (portable.inliers == mlx.inliers).float().mean() > 0.95
    assert not (mlx.inliers & ~inlier).any(1).float().mean() > 0.5


def test_unbatched_mask_and_numpy():
    p1, p2, k, _, _, inlier = scene(1, 400, outliers=0.3)
    pose = estimate_pose(p1[0], p2[0], k[0], k[0], mode="mlx")
    assert pose.R.shape == (3, 3)
    assert pose.t.shape == (3,)
    assert pose.E.shape == (3, 3)
    assert pose.inliers.shape == (400,)
    assert pose.ok.shape == ()
    masked = estimate_pose(p1[0], p2[0], k[0], k[0], mask=inlier[0], mode="mlx")
    assert masked.ok
    assert not masked.inliers[~inlier[0]].any()
    fm = estimate_fundamental(p1[0].numpy(), p2[0].numpy(), mode="mlx")
    assert fm.F.shape == (3, 3)
    assert fm.inliers.shape == (400,)
    p2a, xa, ka, *_ = tap.scene(1, 300, outliers=0.2)
    ab = estimate_absolute_pose(p2a[0], xa[0], ka[0], mode="mlx")
    assert ab.R.shape == (3, 3)
    assert ab.t.shape == (3,)


def test_batch_shapes_and_varied_counts():
    """A batch with masks of different sizes per pair and a pair too small to solve."""
    p1, p2, k, r, t, _ = scene(5, 600, outliers=0.2)
    mask = torch.ones(5, 600, dtype=torch.bool)
    mask[1, 50:] = False
    mask[2, 7:] = False  # 7 matches: enough to try, rarely a pose
    mask[3, 300:] = False
    pose = estimate_pose(p1, p2, k, k, mask=mask, mode="mlx")
    assert pose.R.shape == (5, 3, 3)
    assert pose.inliers.shape == (5, 600)
    assert not pose.inliers[~mask].any()
    good = torch.tensor([0, 1, 3, 4])
    assert pose.ok[good].all()
    assert (
        pose_error(
            type(pose)(*(getattr(pose, f)[good] for f in ("R", "t", "E", "inliers", "ok"))), r[good], t[good]
        ).max()
        < 2
    )


def test_degenerate_inputs():
    p1, p2, k, *_ = scene(2, 50, outliers=0.0)
    with pytest.raises(InputError):
        estimate_pose(p1[:, :4], p2[:, :4], k, k, mode="mlx")
    with pytest.raises(InputError):
        estimate_pose(p1, p2[:, :40], k, k, mode="mlx")
    with pytest.raises(InputError):
        estimate_pose(p1, p2, k, k, threshold=0.0, mode="mlx")
    with pytest.raises(InputError):
        estimate_fundamental(p1[:, :6], p2[:, :6], mode="mlx")
    # all outliers: flagged, identity R and zero E / F
    bad = torch.rand(2, 50, 2, generator=torch.Generator().manual_seed(0)) * 500
    pose = estimate_pose(p1, bad, k, k, mode="mlx")
    assert pose.inliers.sum() <= 2 * 10
    if not pose.ok.all():
        assert torch.allclose(pose.R[~pose.ok], torch.eye(3).expand(int((~pose.ok).sum()), 3, 3))
    # all points identical: no finite garbage, no crash
    same = torch.full((1, 40, 2), 100.0)
    fm = estimate_fundamental(same, same, mode="mlx")
    assert torch.isfinite(fm.F).all()
    pose = estimate_pose(same, same, k[:1], k[:1], mode="mlx")
    assert torch.isfinite(pose.R).all()
    assert torch.isfinite(pose.t).all()


@pytest.mark.parametrize("seed", range(3))
def test_accuracy_over_seeds(seed):
    """Over seeds the MLX errors stay at the portable errors (fp32 kernels vs the fp64 solver path)."""
    p1, p2, k, r, t, _ = scene(8, 1500, outliers=0.6, noise=0.5, seed=seed)
    portable = pose_error(estimate_pose(p1, p2, k, k, mode="portable"), r, t)
    mlx = pose_error(estimate_pose(p1, p2, k, k, mode="mlx"), r, t)
    assert mlx.median() < portable.median() * 1.5 + 0.05


def test_mlx_pose_without_torch():
    """NumPy in, NumPy out (`set_outputs("numpy")`), and torch is never imported."""
    code = (
        "import numpy as np\n"
        "import fastforks\n"
        "from fastforks.matching import estimate_pose, estimate_fundamental, estimate_absolute_pose\n"
        "fastforks.set_outputs('numpy')\n"
        "g = np.random.default_rng(0)\n"
        "X = g.normal(size=(300, 3)) * [2, 2, 1] + [0, 0, 6]\n"
        "K = np.array([[800, 0, 400], [0, 800, 300], [0, 0, 1.0]])\n"
        "a = 0.2; R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])\n"
        "t = np.array([1.0, 0.1, 0.2])\n"
        "X2 = X @ R.T + t\n"
        "p1 = X[:, :2] / X[:, 2:] * 800 + [400, 300]\n"
        "p2 = X2[:, :2] / X2[:, 2:] * 800 + [400, 300]\n"
        "pose = estimate_pose(p1, p2, K, K, mode='mlx')\n"
        "assert isinstance(pose.R, np.ndarray) and pose.ok\n"
        "assert np.abs(pose.R - R).max() < 1e-2, pose.R\n"
        "fm = estimate_fundamental(p1, p2, mode='mlx'); assert isinstance(fm.F, np.ndarray) and fm.ok\n"
        "ab = estimate_absolute_pose(p2, X, K, mode='mlx'); assert isinstance(ab.R, np.ndarray) and ab.ok\n"
        "assert np.abs(ab.t - t).max() < 1e-2, ab.t\n"
        "done()\n"
    )
    _ok(_blocked(code))


def test_auto_picks_mlx_and_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    p1, p2, k, *_ = scene(2, 500, outliers=0.3)
    monkeypatch.setenv("FASTFORKS_MLX", "1")  # conftest turns MLX routing off for the other tests
    rp._apple_mlx.cache_clear()
    rp._announce_mlx.cache_clear()
    messages: list[str] = []
    monkeypatch.setattr(rp.log, "info", lambda msg, *args: messages.append(msg % args))
    try:
        pose = estimate_pose(p1, p2, k, k)
        assert pose.ok.all()
        assert any("MLX" in m for m in messages)
        assert rp._mlx("auto", p1, "essential")
        assert not rp._mlx("portable", p1, "essential")
    finally:
        rp._apple_mlx.cache_clear()


def test_numpy_float64_inputs():
    p1, p2, k, *_ = scene(2, 400, outliers=0.2)
    pose = estimate_pose(p1.double().numpy(), p2.double().numpy(), k.double().numpy(), k.double().numpy(), mode="mlx")
    assert np.asarray(pose.ok).all()
