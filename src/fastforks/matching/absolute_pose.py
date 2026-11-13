# ruff: noqa: N803, N806 - R, t, X, y follow the P3P / pose notation
"""Absolute pose from 2D-3D matches: batched P3P RANSAC on the GPU (or CPU), `R`, `t` and inliers.

```python
from fastforks.matching import estimate_absolute_pose

pose = estimate_absolute_pose(points2d, points3d, K)  # pixels [N, 2], world points [N, 3] (or [P, N, ...]), [3, 3]
pose.R, pose.t, pose.inliers  # x_cam = R X + t; [3, 3], [3], [N] bool
```

The same RANSAC as `estimate_pose` (`relative_pose`: MSAC, confidence 0.99999, at most 1000 iterations, all pairs
of a batch in parallel) with three-point samples solved by Lambda Twist P3P (up to four poses each), the truncated
reprojection error in normalised coordinates (`threshold` px divided by the focal length; points behind the camera
are outliers) and a final Levenberg-Marquardt of `[R | t]` on the inliers. The solver, scoring and refinement are
plain PyTorch or the `fastforks.kernels.ransac` CUDA kernels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from fastforks._core.errors import InputError
from fastforks.matching import relative_pose as rp

if TYPE_CHECKING:
    from fastforks._core.kernels import CubinModule

__all__ = ["AbsolutePose", "estimate_absolute_pose"]

_TINY = 1e-300  # denominators below this are degenerate
MIN_MATCHES = 4  # P3P samples three; a pose needs a fourth match to be unambiguous
SAMPLE = 3
SOLUTIONS = 4


@dataclass(frozen=True)
class AbsolutePose:
    """Pose of the camera in the world frame of the 3D points (`x_cam = R X + t`), per pair for a batch.

    Attributes:
        R: Rotation `[3, 3]` (`[P, 3, 3]` for a batch).
        t: Translation `[3]` (`[P, 3]`), in the units of the 3D points.
        inliers: `[N]` bool (`[P, N]`): matches that reproject within `threshold` of the final pose (false where
            `mask` was).
        ok: `[]` bool (`[P]`): a pose with at least 4 inliers was found; `R`, `t` are the identity and zero otherwise.
    """

    R: torch.Tensor
    t: torch.Tensor
    inliers: torch.Tensor
    ok: torch.Tensor


def estimate_absolute_pose(  # noqa: PLR0913 - the matches, the intrinsics and the options
    points2d: torch.Tensor,
    points3d: torch.Tensor,
    k: torch.Tensor,
    *,
    threshold: float = 2.0,
    mask: torch.Tensor | None = None,
    mode: Literal["auto", "portable", "kernels"] = "auto",
) -> AbsolutePose:
    """P3P RANSAC + refinement of pixel-to-world matches, one camera or a batch at once.

    Args:
        points2d: Pixel keypoints, `[N, 2]` or `[P, N, 2]` (float; any device, the pose is computed there).
        points3d: The matching world points, `[N, 3]` or `[P, N, 3]`.
        k: Intrinsics, `[3, 3]` or `[P, 3, 3]`.
        threshold: Inlier threshold in pixels (reprojection error).
        mask: `[N]` / `[P, N]` bool, the matches to use (cameras of a batch with fewer matches pad with false).
        mode: `"portable"` is plain PyTorch (any device); `"kernels"` the CUDA kernels (raise
            `KernelUnavailableError` where they cannot run); `"auto"` the kernels on a CUDA device that has them.

    Returns:
        The `AbsolutePose`, unbatched if the inputs were.

    Raises:
        InputError: Shapes do not match, fewer than 4 matches, or a non-positive threshold.
        KernelUnavailableError: `mode="kernels"` without a usable kernel binary / CUDA device.
    """
    batched = points2d.dim() == 3  # noqa: PLR2004 - [P, N, 2]
    p2, p3 = (p.reshape(-1, *p.shape[-2:]).to(torch.float32) for p in (points2d, points3d))
    if points2d.shape[:-1] != points3d.shape[:-1] or p2.shape[-1] != 2 or p3.shape[-1] != 3 or p2.dim() != 3:  # noqa: PLR2004
        got = f"{tuple(points2d.shape)} {tuple(points3d.shape)}"
        msg = f"points2d and points3d must be [N, 2] and [N, 3] (or [P, N, 2] and [P, N, 3]), got {got}"
        raise InputError(msg)
    _, _, _, valid = rp._check(p2, p2, threshold, mask, MIN_MATCHES)  # noqa: SLF001 - shared checks
    k = k.reshape(-1, 3, 3).to(torch.float32).expand(p2.shape[0], 3, 3)
    solver = rp._solver(mode, p2.device)  # noqa: SLF001
    pose = _estimate(p2, p3, k, valid, threshold, solver)
    if batched:
        return pose
    return AbsolutePose(pose.R[0], pose.t[0], pose.inliers[0], pose.ok[0])


def _estimate(  # noqa: PLR0913, PLR0917 - the RANSAC state
    p2: torch.Tensor,
    p3: torch.Tensor,
    k: torch.Tensor,
    valid: torch.Tensor,
    threshold: float,
    solver: CubinModule | None,
) -> AbsolutePose:
    """RANSAC + refinement of `[P, N, 2]` pixels and `[P, N, 3]` points, batched over the cameras."""
    dev = p2.device
    P, N = p2.shape[:2]
    gen = torch.Generator(dev).manual_seed(rp._SEED)  # noqa: SLF001
    order = torch.argsort(torch.rand(P, N, device=dev, generator=gen) + (~valid).float(), dim=1)
    x = rp._normalise(torch.gather(p2, 1, order[..., None].expand(P, N, 2)), k)[..., :2]  # noqa: SLF001
    X = torch.gather(p3, 1, order[..., None].expand(P, N, 3))
    counts = valid.sum(1)
    valid = torch.arange(N, device=dev)[None] < counts[:, None]
    focal = (k[:, 0, 0] + k[:, 1, 1]) / 2
    thr2 = (threshold / focal) ** 2
    scorer = _KernelScorer(solver, x, X, valid, thr2) if solver is not None else _TorchScorer(x, X, valid, thr2)
    rounds = rp._ROUND if solver is not None else rp._ROUND_PORTABLE  # noqa: SLF001
    best, best_score, best_count = rp._ransac(scorer, counts, SAMPLE, rounds, rp._TOP, gen)  # noqa: SLF001
    best, inliers, ok = scorer.finish(best, best_score, best_count)
    inliers_out = torch.zeros(P, N, dtype=torch.bool, device=dev).scatter_(1, order, inliers)
    R = torch.where(ok[:, None, None], best[:, :9].view(P, 3, 3), torch.eye(3, device=dev))
    t = torch.where(ok[:, None], best[:, 9:], torch.zeros(3, device=dev))
    return AbsolutePose(R, t, inliers_out, ok)


class _TorchScorer:
    """P3P hypotheses, reprojection scores and the LM refinement in plain PyTorch."""

    width = 12  # floats per model: R row-major, t

    def __init__(self, x: torch.Tensor, X: torch.Tensor, valid: torch.Tensor, thr2: torch.Tensor) -> None:
        self.x, self.X, self.valid, self.thr2 = x, X, valid, thr2

    def hypotheses(self, pairs: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Poses `[A, H * 4, 12]` of the samples `idx` `[A, H, 3]` of `pairs`, and which are real."""
        A = idx.shape[0]
        x = self.x[pairs[:, None, None], idx].reshape(-1, SAMPLE, 2).double()
        y = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        y = y / y.norm(dim=-1, keepdim=True)
        X = self.X[pairs[:, None, None], idx].reshape(-1, SAMPLE, 3).double()
        m, ok = _p3p(y, X)
        return m.float().reshape(A, -1, 12), ok.reshape(A, -1)

    def project(self, pairs: torch.Tensor, m: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Squared reprojection error `[A, C, n]` of the poses `m` `[A, C, 12]` and the camera depth `[A, C, n]`."""
        A, C = m.shape[:2]
        R, t = m[..., :9].view(A, C, 3, 3), m[..., 9:]
        c = self.X[pairs, None, :n] @ R.transpose(-1, -2) + t[:, :, None]  # [A, C, n, 3]
        z = c[..., 2]
        err = ((c[..., :2] / z.clamp_min(1e-12)[..., None] - self.x[pairs, None, :n]) ** 2).sum(-1)
        return err, z

    def score(self, pairs: torch.Tensor, m: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """MSAC scores and inlier counts `[A, C]` of `m` `[A, C, 12]` on the first `n` matches of `pairs`."""
        A, C = m.shape[:2]
        n = min(n, self.valid.shape[1])
        step = max(1, rp._SCORE_ELEMENTS // (C * 3 * n))  # noqa: SLF001 - cameras per chunk
        scores, counts = [], []
        for a in range(0, A, step):
            err, z = self.project(pairs[a : a + step], m[a : a + step], n)
            inlier = self.valid[pairs[a : a + step], None, :n] & (z > 0)
            gain = (1 - err / self.thr2[pairs[a : a + step], None, None]).clamp_min(0) * inlier
            scores.append(gain.sum(2))
            counts.append((gain > 0).sum(2, dtype=torch.int32))
        return torch.cat(scores), torch.cat(counts)

    def inliers(self, m: torch.Tensor, pairs: torch.Tensor | None = None) -> torch.Tensor:
        """Inlier mask `[A, N]` of one pose per camera `[A, 12]` (all cameras when `pairs` is None)."""
        if pairs is None:
            pairs = torch.arange(m.shape[0], device=m.device)
        err, z = self.project(pairs, m[:, None], self.valid.shape[1])
        return (err[:, 0] < self.thr2[pairs, None]) & (z[:, 0] > 0) & self.valid[pairs]

    def refine(self, pairs: torch.Tensor, m: torch.Tensor, iterations: int) -> torch.Tensor:
        """Levenberg-Marquardt of `[R | t]` `[A, 12]` by `(omega, tau)` on the truncated reprojection cost."""
        A = m.shape[0]
        R, t = m[:, :9].view(A, 3, 3).clone(), m[:, 9:].clone()
        X, x, valid, thr2 = self.X[pairs], self.x[pairs], self.valid[pairs], self.thr2[pairs, None]
        eye = torch.eye(3, device=m.device).expand(A, X.shape[1], 3, 3)
        lam = torch.full((A,), 1e-3, device=m.device)

        def terms(R: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            c = X @ R.transpose(-1, -2) + t[:, None]  # [A, N, 3]
            z = c[..., 2]
            iz = 1 / z.clamp_min(1e-12)
            res = c[..., :2] * iz[..., None] - x  # [A, N, 2]
            err = (res**2).sum(-1)
            w = valid & (z > 0) & (err < thr2)
            cost = (torch.where(w, err, thr2) * valid).sum(1)
            jp = torch.stack(
                (iz, torch.zeros_like(iz), -c[..., 0] * iz**2, torch.zeros_like(iz), iz, -c[..., 1] * iz**2), -1
            )
            dc = torch.cat((-(R[:, None] @ rp._cross(X)), eye), -1)  # noqa: SLF001 - [A, N, 3, 6]
            jac = (jp.view(A, -1, 2, 3) @ dc) * w[..., None, None]  # [A, N, 2, 6]
            jtj = torch.einsum("anrk,anrl->akl", jac, jac)
            jtr = torch.einsum("anrk,anr->ak", jac, res * w[..., None])
            return jtj, jtr, cost

        jtj, jtr, cost = terms(R, t)
        for _ in range(iterations):
            damped = jtj + lam[:, None, None] * torch.diag_embed(jtj.diagonal(dim1=1, dim2=2).clamp_min(1e-12))
            step = -torch.linalg.solve(damped, jtr)
            step = torch.where(torch.isfinite(step).all(1, keepdim=True), step, torch.zeros_like(step))
            Rn = R @ rp._rodrigues(step[:, :3])  # noqa: SLF001
            tn = t + step[:, 3:]
            jtj_n, jtr_n, cost_n = terms(Rn, tn)
            better = cost_n < cost
            R = torch.where(better[:, None, None], Rn, R)
            t = torch.where(better[:, None], tn, t)
            jtj = torch.where(better[:, None, None], jtj_n, jtj)
            jtr = torch.where(better[:, None], jtr_n, jtr)
            cost = torch.where(better, cost_n, cost)
            lam = torch.where(better, (lam * 0.1).clamp_min(1e-6), (lam * 10).clamp_max(1e6))
        return torch.cat((R.reshape(A, 9), t), 1)

    def finish(
        self, best: torch.Tensor, best_score: torch.Tensor, best_count: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The final stage of the RANSAC poses `[P, 12]`: refinement, inlier mask, `ok`."""
        P = best.shape[0]
        pairs = torch.arange(P, device=best.device)
        if rp._LM_FINAL:  # noqa: SLF001
            refined = self.refine(pairs, best, rp._LM_FINAL)  # noqa: SLF001
            score, count = (v[:, 0] for v in self.score(pairs, refined[:, None], self.valid.shape[1]))
            better = rp._key(score, count) > rp._key(best_score, best_count)  # noqa: SLF001
            best = torch.where(better[:, None], refined, best)
        inliers = self.inliers(best)
        return best, inliers, inliers.sum(1) >= MIN_MATCHES


class _KernelScorer(_TorchScorer):
    """The CUDA kernels: P3P hypotheses, fused reprojection scoring and the LM polish (`fastforks.kernels.ransac`)."""

    def __init__(
        self, k: CubinModule, x: torch.Tensor, X: torch.Tensor, valid: torch.Tensor, thr2: torch.Tensor
    ) -> None:
        super().__init__(x, X, valid, thr2)
        from fastforks.kernels import ransac  # noqa: PLC0415 - optional at import time

        self.k, self.lib = k, ransac
        pad = torch.zeros_like(X)
        self.pts = torch.cat((x, X, pad), -1).contiguous()  # [P, N, 8]: u, v, X, Y, Z, 0, 0, 0
        self.counts = valid.sum(1, dtype=torch.int32)
        self.thr2 = thr2.float().contiguous()

    def hypotheses(self, pairs: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lib.hypotheses(self.k, self.pts, pairs.int(), idx)

    def score(self, pairs: torch.Tensor, m: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lib.score(self.k, self.pts, self.counts, self.thr2, pairs.int(), m, n)

    def inliers(self, m: torch.Tensor, pairs: torch.Tensor | None = None) -> torch.Tensor:
        if pairs is None:
            return self.lib.inlier_masks(self.k, self.pts, self.counts, self.thr2, m)
        return super().inliers(m, pairs)

    def finish(
        self,
        best: torch.Tensor,
        best_score: torch.Tensor,  # noqa: ARG002 - the kernel rescores
        best_count: torch.Tensor,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m, _, ok, inliers = self.lib.polish_abs(
            self.k,
            self.pts,
            self.counts,
            self.thr2,
            best,
            rp._LM_FINAL,  # noqa: SLF001 - shared RANSAC settings
            MIN_MATCHES,
        )
        return m, inliers, ok


# ---------------------------------------------------------------------------------------------------------------------
# Lambda Twist P3P, batched (float64)


def _sym(  # noqa: PLR0913, PLR0917 - the six entries
    xx: torch.Tensor, xy: torch.Tensor, xz: torch.Tensor, yy: torch.Tensor, yz: torch.Tensor, zz: torch.Tensor
) -> torch.Tensor:
    """A symmetric 3 x 3 matrix as its six upper elements `[..., 6]`."""
    return torch.stack((xx, xy, xz, yy, yz, zz), -1)


def _quad(s: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """`u^T S v` of symmetric `s` `[..., 6]`."""
    su = torch.stack(
        (
            s[..., 0] * v[..., 0] + s[..., 1] * v[..., 1] + s[..., 2] * v[..., 2],
            s[..., 1] * v[..., 0] + s[..., 3] * v[..., 1] + s[..., 4] * v[..., 2],
            s[..., 2] * v[..., 0] + s[..., 4] * v[..., 1] + s[..., 5] * v[..., 2],
        ),
        -1,
    )
    return (u * su).sum(-1)


def _det_sym(s: torch.Tensor) -> torch.Tensor:
    a, b, c, d, e, f = s.unbind(-1)
    return a * (d * f - e * e) - b * (b * f - e * c) + c * (b * e - d * c)


def _tr_adj(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """`tr(adj(S) T)` of symmetric `s`, `t` `[..., 6]`."""
    a, b, c, d, e, f = s.unbind(-1)
    a00, a11, a22 = d * f - e * e, a * f - c * c, a * d - b * b
    a01, a02, a12 = c * e - b * f, b * e - d * c, b * c - a * e
    return (
        a00 * t[..., 0] + a11 * t[..., 3] + a22 * t[..., 5] + 2 * (a01 * t[..., 1] + a02 * t[..., 2] + a12 * t[..., 4])
    )


def _cubic_roots(
    c3: torch.Tensor, c2: torch.Tensor, c1: torch.Tensor, c0: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real roots `[B, 3]` of `c3 g^3 + c2 g^2 + c1 g + c0` (Cardano / trigonometric, Newton polish) and which exist."""
    scale = torch.stack((c3, c2, c1, c0), -1).abs().amax(-1).clamp_min(_TINY)
    cubic = c3.abs() >= 1e-12 * scale
    c3s = torch.where(cubic, c3, torch.ones_like(c3))
    a, b, c = c2 / c3s, c1 / c3s, c0 / c3s
    p, q, sh = b - a * a / 3, 2 * a**3 / 27 - a * b / 3 + c, -a / 3
    disc = q * q / 4 + p**3 / 27
    sq = disc.clamp_min(0).sqrt()
    cbrt = lambda v: v.sign() * v.abs() ** (1 / 3)  # noqa: E731
    one = cbrt(-q / 2 + sq) + cbrt(-q / 2 - sq) + sh
    mag = 2 * (-p / 3).clamp_min(0).sqrt()
    th = torch.acos((3 * q / (p * mag).where(mag > 0, torch.ones_like(mag))).clamp(-1, 1)) / 3
    three = torch.stack([mag * torch.cos(th - 2 * math.pi / 3 * k) + sh for k in range(3)], -1)
    roots = torch.where((disc >= 0)[:, None], one[:, None].expand(-1, 3), three)
    have = torch.ones_like(roots, dtype=torch.bool)
    have[:, 1:] &= (disc < 0)[:, None]
    # Quadratic where the cubic coefficient vanishes.
    c2s = torch.where(c2.abs() >= 1e-12 * scale, c2, torch.ones_like(c2))
    dq = (c1 * c1 - 4 * c2 * c0).clamp_min(0).sqrt()
    quad = torch.stack(((-c1 + dq) / (2 * c2s), (-c1 - dq) / (2 * c2s), torch.zeros_like(c1)), -1)
    quad_ok = ((c2.abs() >= 1e-12 * scale) & (c1 * c1 - 4 * c2 * c0 >= 0))[:, None] & torch.tensor(
        [True, True, False], device=c1.device
    )[None]
    roots = torch.where(cubic[:, None], roots, quad)
    have = torch.where(cubic[:, None], have, quad_ok)
    for _ in range(3):
        f = ((c3[:, None] * roots + c2[:, None]) * roots + c1[:, None]) * roots + c0[:, None]
        df = (3 * c3[:, None] * roots + 2 * c2[:, None]) * roots + c1[:, None]
        roots = torch.where(df.abs() > _TINY, roots - f / df.where(df.abs() > _TINY, torch.ones_like(df)), roots)
    return roots, have


def _eigvec(d0: torch.Tensor, ev: torch.Tensor) -> torch.Tensor:
    """Unit eigenvector `[..., 3]` of symmetric `d0` `[..., 6]` for the eigenvalue `ev` (cross product of two rows)."""
    r0 = torch.stack((d0[..., 0] - ev, d0[..., 1], d0[..., 2]), -1)
    r1 = torch.stack((d0[..., 1], d0[..., 3] - ev, d0[..., 4]), -1)
    r2 = torch.stack((d0[..., 2], d0[..., 4], d0[..., 5] - ev), -1)
    c = torch.stack((torch.cross(r0, r1, dim=-1), torch.cross(r0, r2, dim=-1), torch.cross(r1, r2, dim=-1)), -2)
    n = c.norm(dim=-1)
    v = torch.gather(c, -2, n.argmax(-1)[..., None, None].expand(*n.shape[:-1], 1, 3))[..., 0, :]
    return v / v.norm(dim=-1, keepdim=True).clamp_min(_TINY)


def _inv3(m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of `[..., 3, 3]` by the adjugate, and the determinant."""
    det = torch.linalg.det(m)
    adj = torch.stack(
        [
            torch.stack(
                [
                    m[..., (j + 1) % 3, (i + 1) % 3] * m[..., (j + 2) % 3, (i + 2) % 3]
                    - m[..., (j + 1) % 3, (i + 2) % 3] * m[..., (j + 2) % 3, (i + 1) % 3]
                    for j in range(3)
                ],
                -1,
            )
            for i in range(3)
        ],
        -2,
    )
    return adj / det.where(det.abs() > _TINY, torch.ones_like(det))[..., None, None], det


def _p3p(y: torch.Tensor, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: PLR0915 - one solver
    """Lambda Twist P3P of unit bearings `y` `[B, 3, 3]` and world points `X` `[B, 3, 3]`: poses `[B, 4, 12]`, ok.

    The distances `l_i` (`x_cam_i = l_i y_i`) satisfy three ellipses `l^T M_k l = a_k`; two homogeneous differences
    `D1`, `D2` are combined into a rank-2 cone `D1 + g D2` (`det = 0`, a cubic in `g`) that factors into two planes,
    each cutting the cone in two rays; two Gauss-Newton steps on the ellipses polish the distances. `R` maps the
    frame of the three world points onto the frame of the camera points.
    """
    B = y.shape[0]
    d01, d02, d12 = X[:, 0] - X[:, 1], X[:, 0] - X[:, 2], X[:, 1] - X[:, 2]
    a12, a13, a23 = (d01**2).sum(1), (d02**2).sum(1), (d12**2).sum(1)
    b12, b13, b23 = (y[:, 0] * y[:, 1]).sum(1), (y[:, 0] * y[:, 2]).sum(1), (y[:, 1] * y[:, 2]).sum(1)
    z, o = torch.zeros_like(a12), torch.ones_like(a12)
    m12, m13, m23 = _sym(o, -b12, z, o, z, z), _sym(o, z, -b13, z, z, o), _sym(z, z, z, o, -b23, o)
    d1 = a23[:, None] * m12 - a12[:, None] * m23
    d2 = a23[:, None] * m13 - a13[:, None] * m23
    roots, have = _cubic_roots(_det_sym(d2), _tr_adj(d2, d1), _tr_adj(d1, d2), _det_sym(d1))
    d0 = d1[:, None] + roots[..., None] * d2[:, None]  # [B, 3, 6]
    tr = d0[..., 0] + d0[..., 3] + d0[..., 5]
    mn = (
        d0[..., 0] * d0[..., 3]
        - d0[..., 1] ** 2
        + d0[..., 0] * d0[..., 5]
        - d0[..., 2] ** 2
        + d0[..., 3] * d0[..., 5]
        - d0[..., 4] ** 2
    )
    disc = tr * tr - 4 * mn
    sq = disc.clamp_min(0).sqrt()
    e1, e2 = (tr + sq) / 2, (tr - sq) / 2
    root_ok = have & (disc > 0) & (e1 > 0) & (e2 < 0)
    pick = root_ok.float().argmax(1)
    ar = torch.arange(B, device=y.device)
    ok = root_ok[ar, pick]
    d0, e1, e2, g = d0[ar, pick], e1[ar, pick], e2[ar, pick], roots[ar, pick]
    v1, v2 = _eigvec(d0, e1), _eigvec(d0, e2)
    s1, s2 = e1.clamp_min(0).sqrt()[:, None], (-e2).clamp_min(0).sqrt()[:, None]
    q = torch.where((g.abs() >= 1)[:, None], d1, d2)  # the better-conditioned cone to cut with the planes
    w = torch.stack((s1 * v1 + s2 * v2, s1 * v1 - s2 * v2), 1)  # [B, 2, 3]
    kp = w.abs().argmax(-1)  # [B, 2]
    i1, i2 = (kp + 1) % 3, (kp + 2) % 3
    wk = torch.gather(w, -1, kp[..., None])
    eye = torch.eye(3, device=y.device, dtype=y.dtype)
    r1 = eye[i1] - torch.gather(w, -1, i1[..., None]) / wk * eye[kp]
    r2 = eye[i2] - torch.gather(w, -1, i2[..., None]) / wk * eye[kp]
    qq = q[:, None]
    al, be, de = _quad(qq, r1, r1), _quad(qq, r1, r2), _quad(qq, r2, r2)
    dq = be * be - al * de
    plane_ok = dq >= -1e-9 * (be * be + (al * de).abs())
    dq = dq.clamp_min(0).sqrt()
    use_al = al.abs() >= de.abs()
    als, des = al.where(use_al, torch.ones_like(al)), de.where(~use_al, torch.ones_like(de))
    ls = []
    for sign in (1, -1):
        l1 = torch.where(use_al, (-be + sign * dq) / als, torch.ones_like(al))
        l2 = torch.where(use_al, torch.ones_like(al), (-be + sign * dq) / des)
        ls.append(l1[..., None] * r1 + l2[..., None] * r2)
    lam = torch.stack(ls, 2).reshape(B, SOLUTIONS, 3)  # (plane, sign) -> 4
    sol_ok = ok[:, None] & plane_ok[:, :, None].expand(B, 2, 2).reshape(B, SOLUTIONS)
    q12 = _quad(m12[:, None], lam, lam)
    sol_ok &= q12 > 0
    scale = (a12[:, None] / q12.where(q12 > 0, torch.ones_like(q12))).sqrt()
    scale = torch.where(lam.sum(-1) < 0, -scale, scale)
    lam = lam * scale[..., None]
    # Gauss-Newton on the three ellipses.
    bs = torch.stack((b12, b13, b23), 1)[:, None]  # [B, 1, 3]
    a_s = torch.stack((a12, a13, a23), 1)[:, None]
    for _ in range(2):
        f = (
            torch.stack(
                (_quad(m12[:, None], lam, lam), _quad(m13[:, None], lam, lam), _quad(m23[:, None], lam, lam)), -1
            )
            - a_s
        )
        l0, l1, l2 = lam.unbind(-1)
        zz = torch.zeros_like(l0)
        jac = 2 * torch.stack(
            (
                torch.stack((l0 - bs[..., 0] * l1, l1 - bs[..., 0] * l0, zz), -1),
                torch.stack((l0 - bs[..., 1] * l2, zz, l2 - bs[..., 1] * l0), -1),
                torch.stack((zz, l1 - bs[..., 2] * l2, l2 - bs[..., 2] * l1), -1),
            ),
            -2,
        )
        inv, det = _inv3(jac)
        step = (inv @ f[..., None])[..., 0]
        lam = torch.where((det.abs() > _TINY)[..., None], lam - step, lam)
    sol_ok &= (lam > 0).all(-1)
    # Camera points and the frames.
    c = lam[..., None] * y[:, None]  # [B, 4, 3, 3]
    e01, e02 = c[:, :, 0] - c[:, :, 1], c[:, :, 0] - c[:, :, 2]
    fc = torch.stack((e01, e02, torch.cross(e01, e02, dim=-1)), -1)
    fw = torch.stack((d01, d02, torch.cross(d01, d02, dim=-1)), -1)
    fwi, det = _inv3(fw)
    sol_ok &= (det.abs() > _TINY)[:, None]
    R = fc @ fwi[:, None]
    t = c[:, :, 0] - (R @ X[:, None, 0, :, None])[..., 0]
    m = torch.cat((R.reshape(B, SOLUTIONS, 9), t), -1)
    sol_ok &= torch.isfinite(m).all(-1)
    return torch.where(sol_ok[..., None], m, torch.zeros_like(m)), sol_ok
