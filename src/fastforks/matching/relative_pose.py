"""Relative pose from matches: batched RANSAC of the essential matrix on the GPU (or CPU), `R`, `t` and inliers.

```python
from fastforks.matching import estimate_pose

pose = estimate_pose(points1, points2, K1, K2)  # pixels [N, 2] x 2 (or [P, N, 2] for P pairs), intrinsics [3, 3]
pose.R, pose.t, pose.inliers  # [3, 3], [3] (unit, camera 1 -> camera 2), [N] bool

fund = estimate_fundamental(points1, points2)  # unknown intrinsics: the fundamental matrix, 7-point RANSAC
fund.F, fund.inliers  # [3, 3] (x2^T F x1 = 0 in pixels), [N] bool
```

The protocol is OpenCV's `findEssentialMat` (RANSAC, Nister's 5-point solver, Sampson error below `threshold` px
divided by the mean focal length, confidence 0.99999, at most 1000 iterations) and `recoverPose` (cheirality over the
inliers), with a local optimisation (inlier refits) that OpenCV's RANSAC has not. All pairs of a batch run in
parallel: a round solves and scores a few hundred hypotheses of every unfinished pair at once (no loop over hypotheses),
every candidate is scored on a subset of the matches, the best of them on all. The 5-point solver is batched plain
PyTorch (`_five_point`); `mode="kernels"` runs it and the scoring in one CUDA kernel library
(`fastforks.kernels.ransac`). `estimate_fundamental` is the same RANSAC with the 7-point solver on Hartley-normalised
pixels (`findFundamentalMat`), rank-2 refits instead of the pose refinement, no cheirality.
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

__all__ = ["FundamentalMatrix", "RelativePose", "estimate_fundamental", "estimate_pose"]

log = logging.getLogger(__name__)

# OpenCV findEssentialMat defaults: RANSAC confidence and iteration cap, 5-point hypotheses (`# fixed:` protocol).
CONFIDENCE = 0.99999
MAX_ITERATIONS = 1000
MIN_MATCHES = 5
MIN_MATCHES_F = 7
# Per round (all unfinished pairs at once): hypotheses (kernels; portable: fewer, its solver is slower and needs
# more memory), matches each candidate is pre-scored on, candidates per pair scored on all matches (the best of the
# pre-score), inlier refits of the local optimisation (in `_LO_DTYPE`: float32 matched float64 on MegaDepth-1500 at
# half the time). Tuned on MegaDepth-1500 (results/2026-11-12_gpu_ransac.md).
_ROUND = 512
_ROUND_PORTABLE = 128
_SUBSET = 256
_TOP = 8
_REFITS = 1
_SEED = 0
_LO_DTYPE = torch.float32
# Non-linear refinement (Levenberg-Marquardt of R, t on the Sampson error of the inliers, truncated loss): iterations
# of the final refinement, and of the local optimisation of every round's best model (0: off).
_LM_FINAL = 10
_MSAC = True
_LM_ROUND = 0
_SCORE_ELEMENTS = 2**26  # portable scoring chunk: elements of the [A, C, 5, n] Sampson intermediate (256 MB fp32)


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


@dataclass(frozen=True)
class FundamentalMatrix:
    """Fundamental matrix of a pair (`x2^T F x1 = 0` in pixels), per pair for a batch.

    Attributes:
        F: `[3, 3]` (`[P, 3, 3]` for a batch), unit Frobenius norm.
        inliers: `[N]` bool (`[P, N]`): matches within `threshold` of `F` (false where `mask` was).
        ok: `[]` bool (`[P]`): a model with at least 7 inliers was found; `F` is zero otherwise.
    """

    F: torch.Tensor
    inliers: torch.Tensor
    ok: torch.Tensor


def estimate_fundamental(
    points1: torch.Tensor,
    points2: torch.Tensor,
    *,
    threshold: float = 1.0,
    mask: torch.Tensor | None = None,
    mode: Literal["auto", "portable", "kernels"] = "auto",
) -> FundamentalMatrix:
    """Fundamental-matrix RANSAC (7-point solver) of pixel matches with unknown intrinsics, one pair or a batch.

    Args:
        points1: Pixel keypoints in image 1, `[N, 2]` or `[P, N, 2]` (float; any device).
        points2: The matching keypoints in image 2, same shape.
        threshold: Inlier threshold in pixels (Sampson distance).
        mask: `[N]` / `[P, N]` bool, the matches to use.
        mode: As for `estimate_pose`.

    Returns:
        The `FundamentalMatrix`, unbatched if the inputs were.

    Raises:
        InputError: Shapes do not match, fewer than 7 matches, or a non-positive threshold.
        KernelUnavailableError: `mode="kernels"` without a usable kernel binary / CUDA device.
    """
    batched, p1, p2, valid = _check(points1, points2, threshold, mask, MIN_MATCHES_F)
    # Hartley normalisation as "intrinsics": centroid per image, one scale per pair (the mean distance from the
    # centroids over both images), so the Sampson error of the normalised problem is exactly scale^2 times the
    # pixel one and F = K2^-T F_n K1^-1.
    w = valid.float()[..., None]
    c1, c2 = ((p * w).sum(1) / w.sum(1) for p in (p1, p2))
    dist = ((p1 - c1[:, None]).norm(dim=-1) + (p2 - c2[:, None]).norm(dim=-1)) * valid
    f = (dist.sum(1) / (2 * valid.sum(1))).clamp_min(1e-6)  # 1 / scale
    k1, k2 = _hartley(f, c1), _hartley(f, c2)
    solver = _solver(mode, p1.device)
    est = _estimate(p1, p2, k1, k2, valid, threshold, solver, MIN_MATCHES_F)
    k1_inv, k2_inv = (torch.linalg.inv(k) for k in (k1, k2))
    fm = k2_inv.transpose(1, 2) @ est.E @ k1_inv
    fm = fm / fm.flatten(1).norm(dim=1).clamp_min(1e-30)[:, None, None]
    fm = torch.where(est.ok[:, None, None], fm, torch.zeros_like(fm))
    if batched:
        return FundamentalMatrix(fm, est.inliers, est.ok)
    return FundamentalMatrix(fm[0], est.inliers[0], est.ok[0])


def _hartley(f: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Intrinsics-like `[P, 3, 3]` of a Hartley normalisation: focal `f` `[P]` (1 / scale), centre `c` `[P, 2]`."""
    k = torch.zeros(f.shape[0], 3, 3, device=f.device, dtype=f.dtype)
    k[:, 0, 0] = k[:, 1, 1] = f
    k[:, :2, 2] = c
    k[:, 2, 2] = 1.0
    return k


def _check(
    points1: torch.Tensor, points2: torch.Tensor, threshold: float, mask: torch.Tensor | None, minimum: int
) -> tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shape checks shared by the estimators: `(batched, p1, p2, valid)`, the points `[P, N, 2]` float32."""
    batched = points1.dim() == 3  # noqa: PLR2004 - [P, N, 2]
    p1, p2 = (p.reshape(-1, *p.shape[-2:]).to(torch.float32) for p in (points1, points2))
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
    if n < minimum or int(counts.min()) < minimum:
        msg = f"need at least {minimum} matches per pair, got {int(counts.min())}"
        raise InputError(msg)
    return batched, p1, p2, valid


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
    batched, p1, p2, valid = _check(points1, points2, threshold, mask, MIN_MATCHES)
    k1, k2 = (k.reshape(-1, 3, 3).to(torch.float32).expand(p1.shape[0], 3, 3) for k in (k1, k2))
    solver = _solver(mode, p1.device)
    pose = _estimate(p1, p2, k1, k2, valid, threshold, solver, MIN_MATCHES)
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
    sample: int,
) -> RelativePose:
    """RANSAC + local optimisation + cheirality of normalised `[P, N, 2]` matches, batched over the pairs.

    `sample` 5: essential matrices (5-point solver, pose refinement, `recoverPose`); 7: fundamental matrices
    (7-point solver, rank-2 refits; `R`, `t` are the identity and `(0, 0, 1)`).
    """
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

    best_score = torch.zeros(P, dtype=torch.float32, device=dev)
    best_count = torch.zeros(P, dtype=torch.int32, device=dev)
    best_e = torch.zeros(P, 9, dtype=torch.float32, device=dev)
    rounds = _ROUND if solver is not None else _ROUND_PORTABLE
    used = 0
    active = torch.arange(P, device=dev)
    while active.numel():
        idx = (torch.rand(active.numel(), rounds, sample, device=dev, generator=gen) * counts[active, None, None]).int()
        cand, cand_valid = scorer.hypotheses(active, idx)  # [A, C, 9], [A, C]
        sub = torch.where(cand_valid, scorer.score(active, cand, _SUBSET)[0], -1.0)
        top = sub.topk(_TOP, dim=1).indices
        cand = torch.gather(cand, 1, top[..., None].expand(-1, -1, 9))
        score, count = scorer.score(active, cand, N)
        which = (score if _MSAC else count.float()).argmax(1)
        cand = cand[torch.arange(active.numel(), device=dev), which]
        score, count = (torch.gather(v, 1, which[:, None])[:, 0] for v in (score, count))
        if _LM_ROUND:  # local optimisation of the round's best model
            cand, score, count = _improve(scorer, active, cand, score, count, _LM_ROUND)
        better = _key(score, count) > _key(best_score[active], best_count[active])
        best_score[active] = torch.where(better, score, best_score[active])
        best_count[active] = torch.where(better, count, best_count[active])
        best_e[active] = torch.where(better[:, None], cand, best_e[active])
        used += rounds
        need = _iterations(best_count[active].float() / counts[active].float(), sample)
        active = active[need > used]
    R, t, best_e, inliers, ok = scorer.finish(best_e, best_score, best_count, sample)
    # Back to the caller's match order.
    inliers_out = torch.zeros(P, N, dtype=torch.bool, device=dev).scatter_(1, order, inliers)
    return RelativePose(R, t, best_e.view(P, 3, 3), inliers_out, ok)


def _normalise(p: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Pixels `[P, N, 2]` -> homogeneous normalised `[P, N, 3]` float32 through `K^-1`."""
    f = torch.stack((k[:, 0, 0], k[:, 1, 1]), 1)[:, None]
    c = k[:, :2, 2][:, None]
    return torch.cat(((p - c) / f, torch.ones_like(p[..., :1])), -1)


def _iterations(ratio: torch.Tensor, sample: int) -> torch.Tensor:
    """OpenCV `RANSACUpdateNumIters`: `sample`-point hypotheses for `CONFIDENCE` at inlier ratio `ratio`, capped."""
    denom = torch.log1p(-(ratio.clamp(0, 1 - 1e-7) ** sample))
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
        """Models `[A, H * S, 9]` of the samples `idx` `[A, H, 5 | 7]` of `pairs`, and which are real.

        Five-point samples give up to `S = 10` essential matrices each, seven-point samples up to `S = 3`
        fundamental matrices.
        """
        A, _, size = idx.shape
        x1 = self.x1[pairs[:, None, None], idx].reshape(-1, size, 3).double()  # [A * H, size, 3]
        x2 = self.x2[pairs[:, None, None], idx].reshape(-1, size, 3).double()
        e, ok = _five_point(x1, x2) if size == MIN_MATCHES else _seven_point(x1, x2)
        return e.float().reshape(A, -1, 9), ok.reshape(A, -1)

    def score(self, pairs: torch.Tensor, e: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """MSAC scores and inlier counts `[A, C]` of `e` `[A, C, 9]` on the first `n` matches of `pairs`.

        The score of a match is `1 - error / threshold^2` (truncated quadratic loss; 0 for outliers), so the score
        of a model is at most its inlier count.
        """
        A, C = e.shape[:2]
        n = min(n, self.feat.shape[-1])
        step = max(1, _SCORE_ELEMENTS // (C * 5 * n))  # pairs per chunk: bounds the [A, C, 5, n] intermediate
        coef = _coef(e).reshape(A, C * 5, 15)
        scores, counts = [], []
        for a in range(0, A, step):
            v = torch.matmul(coef[a : a + step], self.feat[pairs[a : a + step], :, :n]).view(-1, C, 5, n)
            den = (v[:, :, 1:] ** 2).sum(2)
            err = v[:, :, 0] ** 2 / den.clamp_min(1e-30)
            inlier = self.valid[pairs[a : a + step], None, :n] & (den > 0)
            gain = (1 - err / self.thr2[pairs[a : a + step], None, None]).clamp_min(0) * inlier
            scores.append(gain.sum(2))
            counts.append((gain > 0).sum(2, dtype=torch.int32))
        return torch.cat(scores), torch.cat(counts)

    def cost(self, pairs: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Truncated Sampson cost `[A]` of one model per pair `[A, 9]`: `sum(min(error, threshold^2))`."""
        score = self.score(pairs, e[:, None], self.valid.shape[1])[0][:, 0]
        return self.thr2[pairs] * (self.valid[pairs].sum(1) - score)

    def refine(self, pairs: torch.Tensor, e: torch.Tensor, iterations: int) -> torch.Tensor:
        """Levenberg-Marquardt of `R`, `t` (`E = [t]_x R`) on the Sampson error of the inliers of `e` `[A, 9]`.

        Truncated quadratic loss (PoseLib's LO-RANSAC refinement): the cost is the MSAC cost, inliers are refreshed
        every iteration, a step counts when it lowers the cost. Returns the refined `E` `[A, 9]` (unit norm).
        """
        return _refine(self, pairs, e, iterations)

    def lm_terms(
        self, pairs: torch.Tensor, e: torch.Tensor, de: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`J^T J` `[A, 5, 5]`, `J^T r` `[A, 5]` and the truncated cost `[A]` of the models `e` `[A, 9]`.

        `de` `[A, 9, 5]` are the derivatives of `e` by the pose parameters; the Jacobian runs over the inliers.
        """
        res, dres = _sampson(e.view(-1, 3, 3), self.x1[pairs], self.x2[pairs])
        thr2 = self.thr2[pairs, None]
        w = (self.valid[pairs] & (res**2 < thr2)).float()
        jac = (dres * w[..., None]) @ de  # [A, N, 5]
        cost = (torch.where(w > 0, res**2, thr2) * self.valid[pairs]).sum(1)
        return jac.transpose(1, 2) @ jac, (jac * res[..., None]).sum(1), cost

    def finish(
        self, best_e: torch.Tensor, best_score: torch.Tensor, best_count: torch.Tensor, sample: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The final stage of the RANSAC models `[P, 9]`: polish, `recoverPose`, inlier mask.

        Returns `R` `[P, 3, 3]`, `t` `[P, 3]`, `E` `[P, 9]`, inliers `[P, N]` and `ok` `[P]` (identity and
        `(0, 0, 1)` where not ok, and for fundamental matrices, `sample` 7).
        """
        P, dev = best_e.shape[0], best_e.device
        fundamental = sample == MIN_MATCHES_F
        best_e, best_count = _polish(self, best_e, best_score, best_count, fundamental=fundamental)
        inliers = self.inliers(best_e)
        eye = torch.eye(3, device=dev).expand(P, 3, 3)
        unit = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(P, 3)
        if fundamental:
            return eye, unit, best_e, inliers, best_count >= sample
        R, t, ok = _recover_pose(best_e.view(P, 3, 3), self.x1, self.x2, inliers)
        ok &= best_count >= sample
        R = torch.where(ok[:, None, None], R, eye)
        t = torch.where(ok[:, None], t, unit)
        return R, t, best_e, inliers, ok

    def inliers(self, e: torch.Tensor, pairs: torch.Tensor | None = None) -> torch.Tensor:
        """Inlier mask `[A, N]` of one model per pair `[A, 9]` (all pairs when `pairs` is None)."""
        feat, valid, thr2 = self.feat, self.valid, self.thr2
        if pairs is not None:
            feat, valid, thr2 = feat[pairs], valid[pairs], thr2[pairs]
        A, N = valid.shape
        v = torch.matmul(_coef(e[:, None])[:, 0], feat).view(A, 5, N)
        den = (v[:, 1:] ** 2).sum(1)
        err = v[:, 0] ** 2 / den.clamp_min(1e-30)
        return (err < thr2[:, None]) & (den > 0) & valid


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

    def score(self, pairs: torch.Tensor, e: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lib.score(self.k, self.pts, self.counts, self.thr2, pairs.int(), e, n)

    def inliers(self, e: torch.Tensor, pairs: torch.Tensor | None = None) -> torch.Tensor:
        if pairs is None:
            return self.lib.inlier_masks(self.k, self.pts, self.counts, self.thr2, e)
        return super().inliers(e, pairs)

    def finish(
        self,
        best_e: torch.Tensor,
        best_score: torch.Tensor,  # noqa: ARG002 - the kernel rescores
        best_count: torch.Tensor,  # noqa: ARG002
        sample: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fundamental = sample == MIN_MATCHES_F
        R, t, e, _, ok, inliers = self.lib.polish(
            self.k,
            self.pts,
            self.counts,
            self.thr2,
            best_e,
            _REFITS,
            0 if fundamental else _LM_FINAL,
            fundamental=fundamental,
            minimum=sample,
        )
        return R, t, e, inliers, ok

    def refine(self, pairs: torch.Tensor, e: torch.Tensor, iterations: int) -> torch.Tensor:
        A = e.shape[0]
        r, t, _ = _recover_pose(e.view(A, 3, 3), self.x1[pairs], self.x2[pairs], self.inliers(e, pairs))
        rt = torch.cat((r.reshape(A, 9), t), 1).float()
        rt = self.lib.lm_refine(self.k, self.pts, self.counts, self.thr2, pairs.int(), rt, iterations)
        out = (_cross(rt[:, 9:]) @ rt[:, :9].view(A, 3, 3)).reshape(A, 9)
        return out / out.norm(dim=1, keepdim=True).clamp_min(1e-12)


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
# The 7-point fundamental-matrix solver, batched


def _seven_point(x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fundamental matrices `[B, 3, 9]` (unit norm) of seven correspondences `[B, 7, 3]`, and `[B, 3]` which are real.

    `F = l F1 + (1 - l) F2` over the two null vectors of the epipolar constraints; `det F = 0` is a cubic in `l`
    (interpolated from four determinants), solved as the eigenvalues of its companion matrix.
    """
    B, dt, dev = x1.shape[0], x1.dtype, x1.device
    null = torch.linalg.svd(_kron(x1, x2), full_matrices=True).Vh[:, 7:, :]  # [B, 2, 9]
    f1, f2 = null[:, 0].view(B, 3, 3), null[:, 1].view(B, 3, 3)
    d0, d1 = torch.linalg.det(f2), torch.linalg.det(f1)
    dm, d2 = torch.linalg.det(2 * f2 - f1), torch.linalg.det(2 * f1 - f2)
    a2 = 0.5 * (d1 + dm) - d0
    a3 = (d2 - d0 - 4 * a2 - (d1 - dm)) / 6
    a1 = 0.5 * (d1 - dm) - a3
    scale = torch.stack((a3, a2, a1, d0), 1).abs().amax(1).clamp_min(1e-300)
    cubic = a3.abs() > 1e-10 * scale  # else lower degree: the companion matrix is not defined
    a3s = torch.where(cubic, a3, torch.ones_like(a3))
    comp = torch.zeros(B, 3, 3, device=dev, dtype=dt)
    comp[:, 1, 0] = comp[:, 2, 1] = 1.0
    comp[:, 0, 0], comp[:, 0, 1], comp[:, 0, 2] = -a2 / a3s, -a1 / a3s, -d0 / a3s
    z = torch.linalg.eigvals(comp)  # [B, 3] complex
    real = cubic[:, None] & (z.imag.abs() < 1e-8 * (1 + z.real.abs()))
    lam = z.real
    for _ in range(2):  # Newton polish
        v = ((a3[:, None] * lam + a2[:, None]) * lam + a1[:, None]) * lam + d0[:, None]
        dv = (3 * a3[:, None] * lam + 2 * a2[:, None]) * lam + a1[:, None]
        lam = torch.where(dv != 0, lam - v / dv, lam)
    f = lam[..., None] * null[:, None, 0] + (1 - lam[..., None]) * null[:, None, 1]  # [B, 3, 9]
    f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-300)
    ok = real & torch.isfinite(f).all(-1)
    return torch.where(ok[..., None], f, 0.0), ok


# ---------------------------------------------------------------------------------------------------------------------
# Local optimisation and recoverPose


def _polish(
    scorer: _TorchScorer,
    best_e: torch.Tensor,
    best_score: torch.Tensor,
    best_count: torch.Tensor,
    *,
    fundamental: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local optimisation of the final models `[P, 9]`: refits on the inliers, then non-linear refinement.

    A step is kept while the score rises. Returns the models and their inlier counts. Fundamental matrices get
    rank-2 refits and no pose refinement.
    """
    everyone = torch.arange(best_e.shape[0], device=best_e.device)
    n = scorer.valid.shape[1]
    for _ in range(_REFITS):
        refit = _refit(scorer.kron, scorer.inliers(best_e).float(), best_e, fundamental=fundamental)
        score, count = (v[:, 0] for v in scorer.score(everyone, refit[:, None], n))
        better = _key(score, count) > _key(best_score, best_count)
        best_score, best_count = torch.where(better, score, best_score), torch.where(better, count, best_count)
        best_e = torch.where(better[:, None], refit, best_e)
    if _LM_FINAL and not fundamental:
        best_e, _, best_count = _improve(scorer, everyone, best_e, best_score, best_count, _LM_FINAL)
    return best_e, best_count


def _key(score: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    """The model selection key: the MSAC score, or the inlier count."""
    return score if _MSAC else count.float()


def _improve(  # noqa: PLR0913, PLR0917 - the RANSAC state
    scorer: _TorchScorer,
    pairs: torch.Tensor,
    e: torch.Tensor,
    score: torch.Tensor,
    count: torch.Tensor,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Non-linear refinement of the models `e` `[A, 9]` of `pairs`, kept where the MSAC score rises."""
    refined = scorer.refine(pairs, e, iterations)
    new_score, new_count = (v[:, 0] for v in scorer.score(pairs, refined[:, None], scorer.valid.shape[1]))
    better = _key(new_score, new_count) > _key(score, count)
    return (
        torch.where(better[:, None], refined, e),
        torch.where(better, new_score, score),
        torch.where(better, new_count, count),
    )


def _cross(t: torch.Tensor) -> torch.Tensor:
    """`[t]_x` `[..., 3, 3]` of `t` `[..., 3]`."""
    z = torch.zeros_like(t[..., 0])
    rows = (z, -t[..., 2], t[..., 1], t[..., 2], z, -t[..., 0], -t[..., 1], t[..., 0], z)
    return torch.stack(rows, -1).view(*t.shape[:-1], 3, 3)


def _sampson(e: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed Sampson error `[P, N]` of `e` `[P, 3, 3]` and its derivative `[P, N, 9]` with respect to `e`."""
    lhs, rhs = torch.einsum("pij,pnj->pni", e, x1), torch.einsum("pji,pnj->pni", e, x2)  # E x1, E^T x2
    num = (x2 * lhs).sum(-1)
    den = lhs[..., 0] ** 2 + lhs[..., 1] ** 2 + rhs[..., 0] ** 2 + rhs[..., 1] ** 2
    inv = den.clamp_min(1e-20).rsqrt()
    kron = (x2[..., :, None] * x1[..., None, :]).reshape(*x1.shape[:-1], 9)
    dden = torch.cat((lhs[..., :1] * x1, lhs[..., 1:2] * x1, torch.zeros_like(x1)), -1)  # rows: E x1 terms
    rhs = torch.stack((rhs[..., 0], rhs[..., 1], torch.zeros_like(rhs[..., 0])), -1)  # columns: E^T x2 terms
    dden = dden + (x2[..., :, None] * rhs[..., None, :]).reshape(dden.shape)
    return num * inv, kron * inv[..., None] - (num * inv**3)[..., None] * dden


def _refine(scorer: _TorchScorer, pairs: torch.Tensor, e: torch.Tensor, iterations: int) -> torch.Tensor:
    """The portable Levenberg-Marquardt (`_TorchScorer.refine`); the kernels run it inside `lm_refine`."""
    A = e.shape[0]
    x1, x2 = scorer.x1[pairs], scorer.x2[pairs]
    r, t, _ = _recover_pose(e.view(A, 3, 3), x1, x2, scorer.inliers(e, pairs))
    basis = torch.eye(3, device=e.device)
    lam = torch.full((A,), 1e-3, device=e.device)
    best = scorer.cost(pairs, (_cross(t) @ r).reshape(A, 9))
    for _ in range(iterations):
        # Tangent basis of t (two unit vectors orthogonal to t); derivatives of E = [t]_x R by the 5 parameters.
        b1 = torch.linalg.cross(t, basis[t.abs().argmin(1)])
        b1 = b1 / b1.norm(dim=1, keepdim=True).clamp_min(1e-12)
        b2 = torch.linalg.cross(t, b1)
        tx = _cross(t)
        d_omega = torch.stack([(tx @ r @ _cross(basis[k])).reshape(A, 9) for k in range(3)], -1)  # [A, 9, 3]
        d_alpha = torch.stack([(_cross(b) @ r).reshape(A, 9) for b in (b1, b2)], -1)  # [A, 9, 2]
        jtj, jtr, _ = scorer.lm_terms(pairs, (tx @ r).reshape(A, 9), torch.cat((d_omega, d_alpha), -1))
        damped = jtj + lam[:, None, None] * torch.diagonal(jtj, dim1=1, dim2=2).clamp_min(1e-12).diag_embed()
        step = -torch.linalg.solve(damped, jtr[..., None])[..., 0]
        step = torch.where(torch.isfinite(step).all(1, keepdim=True), step, torch.zeros_like(step))
        r_new = r @ _rodrigues(step[:, :3])
        t_new = t + step[:, 3:4] * b1 + step[:, 4:5] * b2
        t_new = t_new / t_new.norm(dim=1, keepdim=True).clamp_min(1e-12)
        new = scorer.cost(pairs, (_cross(t_new) @ r_new).reshape(A, 9))
        accept = new < best
        r = torch.where(accept[:, None, None], r_new, r)
        t = torch.where(accept[:, None], t_new, t)
        best = torch.where(accept, new, best)
        lam = torch.where(accept, lam * 0.1, lam * 10).clamp(1e-6, 1e6)
    out = (_cross(t) @ r).reshape(A, 9)
    return out / out.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _rodrigues(w: torch.Tensor) -> torch.Tensor:
    """Rotation matrices `[A, 3, 3]` of the axis-angle vectors `w` `[A, 3]`."""
    theta = w.norm(dim=1, keepdim=True).clamp_min(1e-12)
    k = _cross(w / theta)
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    return torch.eye(3, device=w.device, dtype=w.dtype) + s * k + (1 - c) * (k @ k)


def _refit(kron: torch.Tensor, w: torch.Tensor, e: torch.Tensor, *, fundamental: bool = False) -> torch.Tensor:
    """Weighted 8-point refit of `e` `[P, 9]` on `kron` `[P, N, 9]` with weights `[P, N]`.

    Projected to an essential matrix (equal singular values) or, `fundamental`, to rank 2; `e` where fewer than
    8 points weigh.
    """
    kron, w = kron.to(_LO_DTYPE), w.to(_LO_DTYPE)
    ata = torch.einsum("pni,pn,pnj->pij", kron, w, kron)
    vec = torch.linalg.eigh(ata).eigenvectors[..., 0]  # smallest eigenvalue
    u, sig, vh = torch.linalg.svd(vec.view(-1, 3, 3))
    rank2 = torch.tensor([1.0, 1.0, 0.0], device=e.device, dtype=u.dtype)[None]
    proj = (u * (sig * rank2 if fundamental else rank2)[:, None]) @ vh
    proj = (proj / proj.flatten(1).norm(dim=1)[:, None, None]).reshape(-1, 9).float()
    enough = (w.sum(1) >= 8)[:, None]  # noqa: PLR2004 - the linear solver's minimum
    return torch.where(enough & torch.isfinite(proj).all(1, keepdim=True), proj, e)


def _recover_pose(
    e: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, inliers: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """OpenCV `recoverPose`: the decomposition of `E` `[P, 3, 3]` with most inliers in front of both cameras."""
    P = e.shape[0]
    u, _, vh = torch.linalg.svd(e.to(_LO_DTYPE))
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
