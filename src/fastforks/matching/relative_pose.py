"""Relative pose from matches: batched RANSAC of the essential matrix on the GPU (or CPU), `R`, `t` and inliers.

```python
from fastforks.matching import estimate_pose

pose = estimate_pose(points1, points2, K1, K2)  # pixels [N, 2] x 2 (or [P, N, 2] for P pairs), intrinsics [3, 3]
pose.R, pose.t, pose.inliers  # [3, 3], [3] (unit, camera 1 -> camera 2), [N] bool
```

The protocol is OpenCV's `findEssentialMat` (RANSAC, Nister's 5-point solver, Sampson error below `threshold` px
divided by the mean focal length, confidence 0.99999, at most 1000 iterations) and `recoverPose` (cheirality over the
inliers), with a local optimisation (inlier refits) that OpenCV's RANSAC has not. All pairs of a batch run in
parallel: a round solves and scores `_ROUND` hypotheses of every unfinished pair at once (no loop over hypotheses),
every candidate is scored on a subset of the matches, the best of them on all. The 5-point solver is batched plain
PyTorch (`_five_point`); `mode="kernels"` runs it and the scoring in one CUDA kernel library
(`fastforks.kernels.ransac`).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from fastforks._core.errors import InputError, KernelUnavailableError

if TYPE_CHECKING:
    from fastforks._core.kernels import CubinModule

__all__ = ["RelativePose", "estimate_pose"]

log = logging.getLogger(__name__)

# OpenCV findEssentialMat defaults: RANSAC confidence and iteration cap, 5-point hypotheses (`# fixed:` protocol).
CONFIDENCE = 0.99999
MAX_ITERATIONS = 1000
MIN_MATCHES = 5
# Per round (all unfinished pairs at once): hypotheses, matches each candidate is pre-scored on, candidates per pair
# scored on all matches (the best of the pre-score), inlier refits of the local optimisation.
_ROUND = 128
_SUBSET = 256
_TOP = 8
_REFITS = 3
_SEED = 0


@dataclass(frozen=True)
class RelativePose:
    """Relative pose of camera 2 with respect to camera 1 (`x2 ~ R x1 + t`), per pair for a batch.

    Attributes:
        R: Rotation `[3, 3]` (`[P, 3, 3]` for a batch).
        t: Unit translation `[3]` (`[P, 3]`); its scale is not observable from two views.
        E: The essential matrix `[3, 3]` (`[P, 3, 3]`), unit Frobenius norm, `x2^T E x1 = 0` in normalised coordinates.
        inliers: `[N]` bool (`[P, N]`): matches within `threshold` of the final model (false where `mask` was).
        ok: `[]` bool (`[P]`): a model with at least 5 inliers and a valid cheirality was found; `R`, `t` are
            the identity and `(0, 0, 1)` otherwise.
    """

    R: torch.Tensor
    t: torch.Tensor
    E: torch.Tensor
    inliers: torch.Tensor
    ok: torch.Tensor


def estimate_pose(  # noqa: PLR0913 - the matches, the two intrinsics and the options
    points1: torch.Tensor,
    points2: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    *,
    threshold: float = 0.5,
    mask: torch.Tensor | None = None,
    mode: Literal["auto", "portable", "kernels"] = "auto",
) -> RelativePose:
    """Essential-matrix RANSAC + `recoverPose` of pixel matches, one pair or a batch of pairs at once.

    Args:
        points1: Pixel keypoints in image 1, `[N, 2]` or `[P, N, 2]` (float; any device, the pose is computed there).
        points2: The matching keypoints in image 2, same shape.
        k1: Intrinsics of camera 1, `[3, 3]` or `[P, 3, 3]`.
        k2: Intrinsics of camera 2, same.
        threshold: Inlier threshold in pixels (Sampson distance); divided by the mean focal length of the pair, as
            upstream matching benchmarks do.
        mask: `[N]` / `[P, N]` bool, the matches to use (pairs of a batch with fewer matches pad with false).
        mode: `"portable"` is plain PyTorch (any device); `"kernels"` the CUDA kernels (raise
            `KernelUnavailableError` where they cannot run); `"auto"` the kernels on a CUDA device that has them.

    Returns:
        The `RelativePose`, unbatched if the inputs were.

    Raises:
        InputError: Shapes do not match, fewer than 5 matches, or a non-positive threshold.
        KernelUnavailableError: `mode="kernels"` without a usable kernel binary / CUDA device.
    """
    batched = points1.dim() == 3  # noqa: PLR2004 - [P, N, 2]
    p1, p2 = (p.reshape(-1, *p.shape[-2:]).to(torch.float32) for p in (points1, points2))
    k1, k2 = (k.reshape(-1, 3, 3).to(torch.float32).expand(p1.shape[0], 3, 3) for k in (k1, k2))
    if points1.shape != points2.shape or p1.shape[-1] != 2 or p1.dim() != 3:  # noqa: PLR2004
        got = f"{tuple(points1.shape)} {tuple(points2.shape)}"
        msg = f"points1 and points2 must be [N, 2] or [P, N, 2] of one shape, got {got}"
        raise InputError(msg)
    if threshold <= 0:
        msg = f"threshold must be positive, got {threshold}"
        raise InputError(msg)
    n = p1.shape[1]
    valid = torch.ones(p1.shape[:2], dtype=torch.bool, device=p1.device) if mask is None else mask.reshape(-1, n)
    if valid.shape != p1.shape[:2] or valid.dtype != torch.bool:
        msg = f"mask must be bool {tuple(p1.shape[:2])}, got {valid.dtype} {tuple(valid.shape)}"
        raise InputError(msg)
    counts = valid.sum(1)
    if n < MIN_MATCHES or int(counts.min()) < MIN_MATCHES:
        msg = f"need at least {MIN_MATCHES} matches per pair, got {int(counts.min())}"
        raise InputError(msg)
    solver = _solver(mode, p1.device)
    pose = _estimate(p1, p2, k1, k2, valid, threshold, solver)
    if batched:
        return pose
    return RelativePose(pose.R[0], pose.t[0], pose.E[0], pose.inliers[0], pose.ok[0])


def _solver(mode: str, device: torch.device) -> CubinModule | None:
    """The kernel module for `mode` on `device`, `None` for the portable path."""
    if mode == "portable":
        return None
    if mode not in {"auto", "kernels"}:
        msg = f"mode must be 'auto', 'portable' or 'kernels', got {mode!r}"
        raise InputError(msg)
    if device.type != "cuda":
        if mode == "kernels":
            lib, why = "ransac", f"needs a CUDA device, got {device}"
            raise KernelUnavailableError(lib, why)
        return None
    from fastforks.kernels import ransac  # noqa: PLC0415 - optional at import time

    try:
        return ransac.load(device)
    except KernelUnavailableError:
        if mode == "kernels":
            raise
        log.info("ransac kernels unavailable on %s, portable path", device)
        return None


# ---------------------------------------------------------------------------------------------------------------------
# RANSAC


def _estimate(  # noqa: PLR0913, PLR0917 - the RANSAC state
    p1: torch.Tensor,
    p2: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    valid: torch.Tensor,
    threshold: float,
    solver: CubinModule | None,
) -> RelativePose:
    """RANSAC + local optimisation + cheirality of normalised `[P, N, 2]` matches, batched over the pairs."""
    dev = p1.device
    P, N = p1.shape[:2]
    gen = torch.Generator(dev).manual_seed(_SEED)
    # Normalised coordinates; a random order per pair, valid matches first (the first _SUBSET pre-score candidates).
    order = torch.argsort(torch.rand(P, N, device=dev, generator=gen) + (~valid).float(), dim=1)
    x1 = _normalise(torch.gather(p1, 1, order[..., None].expand(P, N, 2)), k1)
    x2 = _normalise(torch.gather(p2, 1, order[..., None].expand(P, N, 2)), k2)
    counts = valid.sum(1)
    valid = torch.arange(N, device=dev)[None] < counts[:, None]
    focal = (k1[:, 0, 0] + k1[:, 1, 1] + k2[:, 0, 0] + k2[:, 1, 1]) / 4
    thr2 = (threshold / focal) ** 2  # [P]
    scorer = _KernelScorer(solver, x1, x2, valid, thr2) if solver is not None else _TorchScorer(x1, x2, valid, thr2)

    best_count = torch.zeros(P, dtype=torch.int32, device=dev)
    best_e = torch.zeros(P, 9, dtype=torch.float32, device=dev)
    used = 0
    active = torch.arange(P, device=dev)
    while active.numel():
        idx = (torch.rand(active.numel(), _ROUND, 5, device=dev, generator=gen) * counts[active, None, None]).int()
        cand, cand_valid = scorer.hypotheses(active, idx)  # [A, C, 9], [A, C]
        sub = torch.where(cand_valid, scorer.score(active, cand, _SUBSET), -1)
        top = sub.topk(_TOP, dim=1).indices
        cand = torch.gather(cand, 1, top[..., None].expand(-1, -1, 9))
        full = scorer.score(active, cand, N)
        count, which = full.max(1)
        better = count > best_count[active]
        best_count[active] = torch.where(better, count, best_count[active])
        best_e[active] = torch.where(
            better[:, None], cand[torch.arange(active.numel(), device=dev), which], best_e[active]
        )
        used += _ROUND
        need = _iterations(best_count[active].float() / counts[active].float())
        active = active[need > used]
    # Local optimisation: refit on the inliers, keep a refit while it has more inliers.
    for _ in range(_REFITS):
        inl = scorer.inliers(best_e)
        refit = _refit(scorer.kron, inl.float(), best_e)
        count = scorer.score(torch.arange(P, device=dev), refit[:, None], N)[:, 0]
        better = count > best_count
        best_count = torch.where(better, count, best_count)
        best_e = torch.where(better[:, None], refit, best_e)
    inliers = scorer.inliers(best_e)
    R, t, ok = _recover_pose(best_e.view(P, 3, 3), x1, x2, inliers)
    ok &= best_count >= MIN_MATCHES
    eye = torch.eye(3, device=dev).expand(P, 3, 3)
    R = torch.where(ok[:, None, None], R, eye)
    t = torch.where(ok[:, None], t, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(P, 3))
    # Back to the caller's match order.
    inliers_out = torch.zeros(P, N, dtype=torch.bool, device=dev).scatter_(1, order, inliers)
    return RelativePose(R, t, best_e.view(P, 3, 3), inliers_out, ok)


def _normalise(p: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Pixels `[P, N, 2]` -> homogeneous normalised `[P, N, 3]` float32 through `K^-1`."""
    f = torch.stack((k[:, 0, 0], k[:, 1, 1]), 1)[:, None]
    c = k[:, :2, 2][:, None]
    return torch.cat(((p - c) / f, torch.ones_like(p[..., :1])), -1)


def _iterations(ratio: torch.Tensor) -> torch.Tensor:
    """OpenCV `RANSACUpdateNumIters`: hypotheses for `CONFIDENCE` at inlier ratio `ratio`, capped."""
    denom = torch.log1p(-(ratio.clamp(0, 1 - 1e-7) ** MIN_MATCHES))
    num = math.log(1 - CONFIDENCE)
    need = (num / denom).where(denom < 0, float(MAX_ITERATIONS))
    return need.clamp(max=MAX_ITERATIONS).ceil()


def _kron(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """`x2 (x) x1` `[P, N, 9]`: `x2^T E x1 = kron . vec(E)` (row-major `E`)."""
    return (x2[..., :, None] * x1[..., None, :]).reshape(*x1.shape[:-1], 9)


class _TorchScorer:
    """Hypotheses and Sampson inlier counts in plain PyTorch."""

    def __init__(self, x1: torch.Tensor, x2: torch.Tensor, valid: torch.Tensor, thr2: torch.Tensor) -> None:
        self.x1, self.x2, self.valid, self.thr2 = x1, x2, valid, thr2
        self.kron = _kron(x1, x2) * valid[..., None]
        # Point features for the five Sampson terms: x2^T E x1, (E x1)_0, (E x1)_1, (E^T x2)_0, (E^T x2)_1.
        self.feat = torch.cat((self.kron, x1, x2), -1).transpose(1, 2).contiguous()  # [P, 15, N]

    def hypotheses(self, pairs: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Essential matrices `[A, H * 10, 9]` of 5-point samples `idx` `[A, H, 5]` of `pairs`, and which are real."""
        A, H = idx.shape[:2]
        x1 = self.x1[pairs[:, None, None], idx]  # [A, H, 5, 3]
        x2 = self.x2[pairs[:, None, None], idx]
        e, ok = _five_point(x1.reshape(-1, 5, 3).double(), x2.reshape(-1, 5, 3).double())
        return e.float().reshape(A, H * 10, 9), ok.reshape(A, H * 10)

    def score(self, pairs: torch.Tensor, e: torch.Tensor, n: int) -> torch.Tensor:
        """Inlier counts `[A, C]` of `e` `[A, C, 9]` on the first `n` matches of `pairs`."""
        A, C = e.shape[:2]
        n = min(n, self.feat.shape[-1])
        v = torch.matmul(_coef(e).reshape(A, C * 5, 15), self.feat[pairs, :, :n]).view(A, C, 5, n)
        err = v[:, :, 0] ** 2 / (v[:, :, 1:] ** 2).sum(2)
        inl = (err < self.thr2[pairs, None, None]) & self.valid[pairs, None, :n]
        return inl.sum(2, dtype=torch.int32)

    def inliers(self, e: torch.Tensor) -> torch.Tensor:
        """Inlier mask `[P, N]` of one model per pair `[P, 9]`."""
        P, N = self.valid.shape
        v = torch.matmul(_coef(e[:, None])[:, 0], self.feat).view(P, 5, N)
        err = v[:, 0] ** 2 / (v[:, 1:] ** 2).sum(1)
        return (err < self.thr2[:, None]) & self.valid


def _coef(e: torch.Tensor) -> torch.Tensor:
    """`[A, C, 5, 15]` Sampson coefficient rows of `e` `[A, C, 9]` against the `[kron, x1, x2]` features."""
    A, C = e.shape[:2]
    E = e.view(A, C, 3, 3)
    z3, z9 = torch.zeros_like(e[..., :3]), torch.zeros_like(e)
    return torch.stack(
        (
            torch.cat((e, z3, z3), -1),
            torch.cat((z9, E[:, :, 0, :], z3), -1),
            torch.cat((z9, E[:, :, 1, :], z3), -1),
            torch.cat((z9, z3, E[:, :, :, 0]), -1),
            torch.cat((z9, z3, E[:, :, :, 1]), -1),
        ),
        2,
    )


class _KernelScorer(_TorchScorer):
    """The CUDA kernels: 5-point hypotheses and fused Sampson scoring (`fastforks.kernels.ransac`)."""

    def __init__(
        self, k: CubinModule, x1: torch.Tensor, x2: torch.Tensor, valid: torch.Tensor, thr2: torch.Tensor
    ) -> None:
        super().__init__(x1, x2, valid, thr2)
        from fastforks.kernels import ransac  # noqa: PLC0415 - optional at import time

        self.k, self.lib = k, ransac
        self.pts = torch.cat((x1[..., :2], x2[..., :2]), -1).contiguous()  # [P, N, 4]
        self.counts = valid.sum(1, dtype=torch.int32)
        self.thr2 = thr2.float().contiguous()

    def hypotheses(self, pairs: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lib.hypotheses(self.k, self.pts, pairs.int(), idx)

    def score(self, pairs: torch.Tensor, e: torch.Tensor, n: int) -> torch.Tensor:
        return self.lib.score(self.k, self.pts, self.counts, self.thr2, pairs.int(), e, n)

    def inliers(self, e: torch.Tensor) -> torch.Tensor:
        return self.lib.inlier_masks(self.k, self.pts, self.counts, self.thr2, e)


# ---------------------------------------------------------------------------------------------------------------------
# Nister's 5-point solver, batched (float64 recommended)


def _monomials() -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """Exponents (x, y, z) of the linear, quadratic and cubic monomial bases (cubic in Nister's order)."""
    lin = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)]
    quad = sorted({(a[0] + b[0], a[1] + b[1], a[2] + b[2]) for a in lin for b in lin}, reverse=True)
    cubic = [
        (3, 0, 0), (0, 3, 0), (2, 1, 0), (1, 2, 0), (2, 0, 1), (2, 0, 0), (0, 2, 1), (0, 2, 0), (1, 1, 1), (1, 1, 0),
        (1, 0, 2), (1, 0, 1), (1, 0, 0), (0, 1, 2), (0, 1, 1), (0, 1, 0), (0, 0, 3), (0, 0, 2), (0, 0, 1), (0, 0, 0),
    ]  # fmt: skip
    return lin, quad, cubic


def _product_table(a: list, b: list, out: list, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """`T[i, j, k] = 1` where monomial `a[i] * b[j] == out[k]`."""
    t = torch.zeros(len(a), len(b), len(out), device=device, dtype=dtype)
    where = {m: k for k, m in enumerate(out)}
    for i, ma in enumerate(a):
        for j, mb in enumerate(b):
            t[i, j, where[ma[0] + mb[0], ma[1] + mb[1], ma[2] + mb[2]]] = 1
    return t


def _tables(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    lin, quad, cubic = _monomials()
    return _product_table(lin, lin, quad, device, dtype), _product_table(quad, lin, cubic, device, dtype)


def _pmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Product of two polynomials, coefficients `[..., da + 1]` and `[..., db + 1]` (highest power first)."""
    out = torch.zeros(*a.shape[:-1], a.shape[-1] + b.shape[-1] - 1, dtype=a.dtype, device=a.device)
    for i in range(a.shape[-1]):
        out[..., i : i + b.shape[-1]] += a[..., i : i + 1] * b
    return out


def _peval(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Horner evaluation of `p` `[..., d + 1]` (highest power first) at `z` `[...]`."""
    v = torch.zeros_like(z) + p[..., 0]
    for i in range(1, p.shape[-1]):
        v = v * z + p[..., i]
    return v


def _roots(p: torch.Tensor, iterations: int = 60) -> torch.Tensor:
    """All complex roots of real polynomials `p` `[B, d + 1]` (Aberth-Ehrlich, fixed iterations), `[B, d]`."""
    d = p.shape[-1] - 1
    p = p / p.abs().amax(-1, keepdim=True).clamp_min(1e-300)
    lead = p[:, :1].abs().clamp_min(1e-30)
    k = torch.arange(1, d + 1, device=p.device, dtype=p.dtype)
    radius = 2 * (p[:, 1:].abs() / lead).pow(1 / k).amax(-1).clamp(1e-3, 1e6)  # Fujiwara bound
    angle = 2 * math.pi * (torch.arange(d, device=p.device, dtype=p.dtype) + 0.37) / d
    z = radius[:, None] * torch.polar(torch.ones_like(angle), angle)[None]
    pc = p.to(z.dtype)
    dp = pc[:, :-1] * torch.arange(d, 0, -1, device=p.device, dtype=p.dtype)
    for _ in range(iterations):
        ratio = _peval(pc[:, None], z) / _peval(dp[:, None], z)
        diff = z[:, :, None] - z[:, None, :]
        diff = diff + torch.eye(d, device=p.device, dtype=z.dtype)[None]  # no self term below
        s = (1 / diff).sum(-1) - 1
        z = z - ratio / (1 - ratio * s)
    return z


def _five_point(x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Nister's 5-point solver.

    Essential matrices `[B, 10, 9]` (unit norm) of five normalised correspondences `[B, 5, 3]`, and `[B, 10]` which
    of the ten slots hold a real solution.

    Null space of the epipolar constraints (SVD), the ten cubic constraints as a `[10, 20]` matrix over the cubic
    monomials, Gauss-Jordan elimination of the first ten, the `3 x 3` polynomial matrix `B(z)`, the degree-10
    polynomial `det B`, its roots (Aberth, polished by Newton) and `(x, y)` from the null vector of `B(z)`.
    """
    B, dt, dev = x1.shape[0], x1.dtype, x1.device
    tlq, tql = _tables(dev, dt)
    a = _kron(x1, x2)  # [B, 5, 9]
    null = torch.linalg.svd(a, full_matrices=True).Vh[:, 5:, :]  # [B, 4, 9]: E = x E1 + y E2 + z E3 + E4
    E = null.permute(0, 2, 1).reshape(B, 3, 3, 4)  # linear polynomial per entry
    eet = torch.einsum("bika,bjkc,acd->bijd", E, E, tlq)  # [B, 3, 3, 10]
    trace = eet[:, 0, 0] + eet[:, 1, 1] + eet[:, 2, 2]
    m = 2 * eet - trace[:, None, None] * torch.eye(3, device=dev, dtype=dt)[None, :, :, None]
    me = torch.einsum("bika,bkjc,acd->bijd", m, E, tql).reshape(B, 9, 20)
    cof = torch.stack(
        (
            torch.einsum("ba,bc,acd->bd", E[:, 1, 1], E[:, 2, 2], tlq)
            - torch.einsum("ba,bc,acd->bd", E[:, 1, 2], E[:, 2, 1], tlq),
            torch.einsum("ba,bc,acd->bd", E[:, 1, 2], E[:, 2, 0], tlq)
            - torch.einsum("ba,bc,acd->bd", E[:, 1, 0], E[:, 2, 2], tlq),
            torch.einsum("ba,bc,acd->bd", E[:, 1, 0], E[:, 2, 1], tlq)
            - torch.einsum("ba,bc,acd->bd", E[:, 1, 1], E[:, 2, 0], tlq),
        ),
        1,
    )  # [B, 3, 10]
    det = torch.einsum("bja,bjc,acd->bd", cof, E[:, 0], tql)[:, None]  # [B, 1, 20]
    rows = torch.cat((det, me), 1)  # [B, 10, 20]
    lu, piv, info = torch.linalg.lu_factor_ex(rows[:, :, :10])
    x = torch.linalg.lu_solve(lu, piv, rows[:, :, 10:])  # [B, 10, 10]: rows of [I | X]

    # Rows e..j (x^2 z, x^2, y^2 z, y^2, xyz, xy) over the tail monomials xz^2 xz x yz^2 yz y z^3 z^2 z 1:
    # k = e - z f, l = g - z h, m = i - z j are x px(z) + y py(z) + pc(z) with degrees 3, 3, 4.
    def combine(e: torch.Tensor, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        px = torch.stack((-f[:, 0], e[:, 0] - f[:, 1], e[:, 1] - f[:, 2], e[:, 2]), -1)
        py = torch.stack((-f[:, 3], e[:, 3] - f[:, 4], e[:, 4] - f[:, 5], e[:, 5]), -1)
        pc = torch.stack((-f[:, 6], e[:, 6] - f[:, 7], e[:, 7] - f[:, 8], e[:, 8] - f[:, 9], e[:, 9]), -1)
        return px, py, pc

    kx, ky, kc = combine(x[:, 4], x[:, 5])
    lx, ly, lc = combine(x[:, 6], x[:, 7])
    mx, my, mc = combine(x[:, 8], x[:, 9])
    poly = _pmul(kx, _pmul(ly, mc) - _pmul(lc, my)) - _pmul(ky, _pmul(lx, mc) - _pmul(lc, mx))
    poly = poly + _pmul(kc, _pmul(lx, my) - _pmul(ly, mx))  # [B, 11]
    z = _roots(poly)  # [B, 10] complex
    real = z.imag.abs() < 1e-6 * (1 + z.real.abs()) if dt == torch.float64 else z.imag.abs() < 1e-3 * (1 + z.real.abs())
    zr = z.real
    dpoly = poly[:, :-1] * torch.arange(10, 0, -1, device=dev, dtype=dt)
    for _ in range(3):  # Newton polish on the real axis
        zr = zr - _peval(poly[:, None], zr) / _peval(dpoly[:, None], zr)
    # (x, y, 1) is the null vector of B(z): the cross product of the two best-conditioned rows.
    ev = lambda p: _peval(p[:, None], zr)  # noqa: E731 - local shorthand
    rk, rl, rm = (ev(kx), ev(ky), ev(kc)), (ev(lx), ev(ly), ev(lc)), (ev(mx), ev(my), ev(mc))
    xs, ys, ws = [], [], []
    for (ax, ay, ac), (bx, by, bc) in ((rk, rl), (rk, rm), (rl, rm)):
        xs.append(ay * bc - ac * by)
        ys.append(ac * bx - ax * bc)
        ws.append(ax * by - ay * bx)
    xs, ys, ws = torch.stack(xs), torch.stack(ys), torch.stack(ws)  # [3, B, 10]
    scale = (xs**2 + ys**2 + ws**2).sqrt().clamp_min(1e-300)
    pick = (ws.abs() / scale).argmax(0, keepdim=True)
    w = torch.gather(ws, 0, pick)[0]
    xv = torch.gather(xs, 0, pick)[0] / w
    yv = torch.gather(ys, 0, pick)[0] / w
    e = xv[..., None] * null[:, None, 0] + yv[..., None] * null[:, None, 1] + zr[..., None] * null[:, None, 2]
    e = e + null[:, None, 3]
    e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-300)
    ok = real & torch.isfinite(e).all(-1) & (info == 0)[:, None] & (w.abs() / torch.gather(scale, 0, pick)[0] > 1e-8)  # noqa: PLR2004
    return torch.where(ok[..., None], e, 0.0), ok


# ---------------------------------------------------------------------------------------------------------------------
# Local optimisation and recoverPose


def _refit(kron: torch.Tensor, w: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """Weighted 8-point refit of `e` `[P, 9]` on `kron` `[P, N, 9]` with weights `[P, N]`.

    Projected to an essential matrix; `e` where fewer than 8 points weigh.
    """
    ata = torch.einsum("pni,pn,pnj->pij", kron.double(), w.double(), kron.double())
    vec = torch.linalg.eigh(ata).eigenvectors[..., 0]  # smallest eigenvalue
    u, _, vh = torch.linalg.svd(vec.view(-1, 3, 3))
    proj = (u * torch.tensor([1.0, 1.0, 0.0], device=e.device, dtype=u.dtype)[None, None]) @ vh
    proj = (proj / proj.flatten(1).norm(dim=1)[:, None, None]).reshape(-1, 9).float()
    enough = (w.sum(1) >= 8)[:, None]  # noqa: PLR2004 - the linear solver's minimum
    return torch.where(enough & torch.isfinite(proj).all(1, keepdim=True), proj, e)


def _recover_pose(
    e: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, inliers: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """OpenCV `recoverPose`: the decomposition of `E` `[P, 3, 3]` with most inliers in front of both cameras."""
    P = e.shape[0]
    u, _, vh = torch.linalg.svd(e.double())
    u = u * torch.where(torch.linalg.det(u) < 0, -1.0, 1.0)[:, None, None]
    vh = vh * torch.where(torch.linalg.det(vh) < 0, -1.0, 1.0)[:, None, None]
    w = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=e.device, dtype=u.dtype)
    r1, r2, t = u @ w @ vh, u @ w.T @ vh, u[:, :, 2]
    rs = torch.stack((r1, r1, r2, r2), 1).float()  # [P, 4, 3, 3]
    ts = torch.stack((t, -t, t, -t), 1).float()  # [P, 4, 3]
    # Depths of the midpoint triangulation: minimise |l2 x2 - l1 R x1 - t|^2 over (l1, l2).
    y = torch.einsum("pkij,pnj->pkni", rs, x1)  # [P, 4, N, 3]
    yy, xx = (y * y).sum(-1), (x2 * x2).sum(-1)[:, None]
    yx = (y * x2[:, None]).sum(-1)
    yt, xt = (y * ts[:, :, None]).sum(-1), (x2[:, None] * ts[:, :, None]).sum(-1)
    det = yy * xx - yx * yx
    l1 = (-yt * xx + yx * xt) / det
    l2 = (yy * xt - yx * yt) / det
    front = ((l1 > 0) & (l2 > 0) & inliers[:, None]).sum(-1)  # [P, 4]
    best = front.argmax(1)
    ar = torch.arange(P, device=e.device)
    return rs[ar, best], ts[ar, best], front[ar, best] > 0
