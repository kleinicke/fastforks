"""`mode="mlx"` of the pose estimators, without torch.

The RANSAC of `relative_pose` / `absolute_pose` on the Metal kernels of `fastforks.kernels.ransac_mlx`. The same
protocol as `mode="kernels"` (CUDA): rounds of `_ROUND` samples per unfinished pair, every hypothesis pre-scored
on `_SUBSET` matches, the `_TOP` best on all, OpenCV's iteration count; the final refits, `recoverPose` and
Levenberg-Marquardt in one kernel. Inputs and bookkeeping are NumPy (one host sync per round), the random draws
NumPy's (seeded: the same inputs give the same result, though not the samples of the torch modes). The results are
torch tensors on the device of a tensor input, else the container of `fastforks.get_outputs()`.
"""
# ruff: noqa: N806 - P, N, R, X follow the pose notation

from __future__ import annotations

import math
from typing import Any

import numpy as np

from fastforks._core.arrays import Outputs, convert, get_outputs, is_tensor, to_numpy
from fastforks._core.errors import InputError
from fastforks.matching import relative_pose as rp

SAMPLE_ABS, MIN_MATCHES_ABS = 3, 4


def _array(x: object, name: str) -> np.ndarray:
    try:
        return to_numpy(x) if is_tensor(x) else np.asarray(x)
    except (TypeError, ValueError) as e:
        msg = f"{name} must be a tensor, numpy array or nested list of numbers, got {type(x).__name__}"
        raise InputError(msg) from e


def _output(inputs: tuple[object, ...]) -> tuple[Outputs, Any]:
    """The result container and torch device: that of the first tensor input, else `get_outputs()` on the CPU."""
    for x in inputs:
        if is_tensor(x):
            return "torch", x.device
    return get_outputs(), None


def _check(  # noqa: PLR0913, PLR0917 - as `relative_pose._check`, plus the point widths
    points1: object, points2: object, threshold: float, mask: object, minimum: int, widths: tuple[int, int] = (2, 2)
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """`(batched, p1, p2, valid)`: the points `[P, N, w]` float32, `valid` `[P, N]` bool."""
    a1, a2 = _array(points1, "points1"), _array(points2, "points2")
    batched = a1.ndim == 3  # noqa: PLR2004 - [P, N, w]
    ok = a1.ndim in (2, 3) and a1.ndim == a2.ndim and a1.shape[:-1] == a2.shape[:-1]
    if not ok or (a1.shape[-1], a2.shape[-1]) != widths:
        w1, w2 = widths
        msg = f"the points must be [N, {w1}] and [N, {w2}] (or [P, N, ...]) of one N, got {a1.shape} {a2.shape}"
        raise InputError(msg)
    if threshold <= 0:
        msg = f"threshold must be positive, got {threshold}"
        raise InputError(msg)
    p1 = a1.reshape(-1, *a1.shape[-2:]).astype(np.float32)
    p2 = a2.reshape(-1, *a2.shape[-2:]).astype(np.float32)
    n = p1.shape[1]
    valid = np.ones(p1.shape[:2], dtype=bool) if mask is None else _array(mask, "mask").reshape(-1, n)
    if valid.shape != p1.shape[:2] or valid.dtype != np.bool_:
        msg = f"mask must be bool {p1.shape[:2]}, got {valid.dtype} {valid.shape}"
        raise InputError(msg)
    counts = valid.sum(1)
    if n < minimum or int(counts.min()) < minimum:
        msg = f"need at least {minimum} matches per pair, got {int(counts.min())}"
        raise InputError(msg)
    return batched, p1, p2, valid


def _intrinsics(k: object, name: str, pairs: int) -> np.ndarray:
    a = _array(k, name)
    if a.shape[-2:] != (3, 3) or a.ndim not in (2, 3) or (a.ndim == 3 and a.shape[0] not in (1, pairs)):  # noqa: PLR2004
        msg = f"{name} must be [3, 3] or [P, 3, 3] with P = {pairs} pairs, got {a.shape}"
        raise InputError(msg)
    return np.broadcast_to(a.reshape(-1, 3, 3).astype(np.float64), (pairs, 3, 3))


def _order(valid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random order per pair, the valid matches first."""
    return np.argsort(rng.random(valid.shape) + ~valid, axis=1, kind="stable")


def _normalise(p: np.ndarray, k: np.ndarray) -> np.ndarray:
    f = np.stack((k[:, 0, 0], k[:, 1, 1]), 1)[:, None]
    return (p - k[:, None, :2, 2]) / f


def _iterations(ratio: np.ndarray, sample: int) -> np.ndarray:
    """`relative_pose._iterations` in NumPy."""
    with np.errstate(divide="ignore"):
        denom = np.log1p(-(np.clip(ratio, 0, 1 - 1e-7) ** sample))
        need = np.where(denom < 0, math.log(1 - rp.CONFIDENCE) / np.where(denom < 0, denom, -1.0), rp.MAX_ITERATIONS)
    return np.ceil(np.minimum(need, rp.MAX_ITERATIONS))


def _ransac(  # noqa: PLR0913, PLR0917 - the loop state
    ops: Any,  # noqa: ANN401 - the kernel module
    pts: Any,  # noqa: ANN401 - mx.array [P, N, 4 | 8]
    counts: np.ndarray,
    thr2: Any,  # noqa: ANN401 - mx.array [P]
    sample: int,
    rounds: int,
    top: int,
    width: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`relative_pose._ransac` on the kernels: the best model `[P, width]`, its score and inlier count `[P]`."""
    mx = ops.mx
    P, N = pts.shape[:2]
    best_e = np.zeros((P, width), np.float32)
    best_score = np.zeros(P, np.float32)
    best_count = np.zeros(P, np.int32)
    counts_mx = mx.array(counts.astype(np.int32))
    used = 0
    active = np.arange(P)
    while active.size:
        draw = rng.random((active.size, rounds, sample)) * counts[active, None, None]
        idx = np.minimum(draw.astype(np.int32), counts[active, None, None] - 1)
        pairs = mx.array(active.astype(np.int32))
        cand, ok = ops.hypotheses(pts, pairs, mx.array(idx))
        sub = mx.where(ok, ops.score(pts, counts_mx, thr2, pairs, cand, rp._SUBSET)[0], -1.0)  # noqa: SLF001
        keep = mx.argpartition(-sub, top - 1, axis=1)[:, :top]
        cand = mx.take_along_axis(cand, keep[..., None], axis=1)
        score, count = ops.score(pts, counts_mx, thr2, pairs, cand, N)
        which = mx.argmax(score, axis=1)[:, None]
        cand = mx.take_along_axis(cand, which[..., None], axis=1)[:, 0]
        score, count = (mx.take_along_axis(v, which, axis=1)[:, 0] for v in (score, count))
        cand, score, count = np.array(cand), np.array(score), np.array(count)
        better = score > best_score[active]
        best_score[active] = np.where(better, score, best_score[active])
        best_count[active] = np.where(better, count, best_count[active])
        best_e[active] = np.where(better[:, None], cand, best_e[active])
        used += rounds
        active = active[_iterations(best_count[active] / counts[active], sample) > used]
    return best_e, best_score, best_count


def _kernels() -> Any:  # noqa: ANN401 - the module
    from fastforks.kernels.ransac_mlx import ops  # noqa: PLC0415 - needs the mlx extra

    return ops


def _relative(  # noqa: PLR0913, PLR0917 - the matches and the options
    p1: np.ndarray, p2: np.ndarray, k1: np.ndarray, k2: np.ndarray, valid: np.ndarray, threshold: float, sample: int
) -> dict[str, np.ndarray]:
    """`relative_pose._estimate` on the kernels: `R`, `t`, `E` (or `F_n`), `inliers`, `ok` arrays."""
    ops = _kernels()
    mx = ops.mx
    P, N = p1.shape[:2]
    rng = np.random.default_rng(rp._SEED)  # noqa: SLF001
    order = _order(valid, rng)
    x1 = _normalise(np.take_along_axis(p1, order[..., None], 1), k1)
    x2 = _normalise(np.take_along_axis(p2, order[..., None], 1), k2)
    counts = valid.sum(1)
    pts = mx.array(np.concatenate((x1, x2), -1).astype(np.float32))
    focal = (k1[:, 0, 0] + k1[:, 1, 1] + k2[:, 0, 0] + k2[:, 1, 1]) / 4
    thr2 = (threshold / focal) ** 2
    rounds, top = rp._ROUND, rp._TOP  # noqa: SLF001
    fundamental = sample == rp.MIN_MATCHES_F
    if fundamental:
        rounds, top = rounds * rp._F_ROUNDS, top * rp._F_ROUNDS  # noqa: SLF001
    wide = thr2 * (rp._F_WIDE**2 if fundamental else 1.0)  # noqa: SLF001
    best_e, _, _ = _ransac(ops, pts, counts, mx.array(wide.astype(np.float32)), sample, rounds, top, 9, rng)
    counts_mx, thr2_mx = mx.array(counts.astype(np.int32)), mx.array(thr2.astype(np.float32))
    refits, lm = rp._REFITS, rp._LM_FINAL  # noqa: SLF001
    r, t, e, _, ok, mask = ops.polish(
        pts, counts_mx, thr2_mx, mx.array(best_e), refits, lm, fundamental=fundamental, minimum=sample
    )
    inliers = np.zeros((P, N), dtype=bool)
    np.put_along_axis(inliers, order, np.array(mask), 1)
    e = np.array(e).reshape(P, 3, 3)
    return {"R": np.array(r), "t": np.array(t), "E": e, "inliers": inliers, "ok": np.array(ok)}


def estimate_pose(  # noqa: PLR0913 - as `relative_pose.estimate_pose`
    points1: object, points2: object, k1: object, k2: object, *, threshold: float, mask: object
) -> rp.RelativePose:
    """`relative_pose.estimate_pose(..., mode="mlx")`."""
    kind, device = _output((points1, points2, k1, k2, mask))
    batched, p1, p2, valid = _check(points1, points2, threshold, mask, rp.MIN_MATCHES)
    P = p1.shape[0]
    k1, k2 = _intrinsics(k1, "k1", P), _intrinsics(k2, "k2", P)
    out = _relative(p1, p2, k1, k2, valid, threshold, rp.MIN_MATCHES)
    fields = [out[name] if batched else out[name][0] for name in ("R", "t", "E", "inliers", "ok")]
    return rp.RelativePose(*convert(fields, kind, device))


def estimate_fundamental(points1: object, points2: object, *, threshold: float, mask: object) -> rp.FundamentalMatrix:
    """`relative_pose.estimate_fundamental(..., mode="mlx")`."""
    kind, device = _output((points1, points2, mask))
    batched, p1, p2, valid = _check(points1, points2, threshold, mask, rp.MIN_MATCHES_F)
    w = valid[..., None].astype(np.float64)
    c1, c2 = ((p * w).sum(1) / w.sum(1) for p in (p1, p2))
    dist = (np.linalg.norm(p1 - c1[:, None], axis=-1) + np.linalg.norm(p2 - c2[:, None], axis=-1)) * valid
    f = np.maximum(dist.sum(1) / (2 * valid.sum(1)), 1e-6)
    k1, k2 = _hartley(f, c1), _hartley(f, c2)
    out = _relative(p1, p2, k1, k2, valid, threshold, rp.MIN_MATCHES_F)
    fm = np.linalg.inv(k2).transpose(0, 2, 1) @ out["E"].astype(np.float64) @ np.linalg.inv(k1)
    fm /= np.maximum(np.linalg.norm(fm.reshape(-1, 9), axis=1), 1e-30)[:, None, None]
    fm = np.where(out["ok"][:, None, None], fm, 0.0).astype(np.float32)
    fields = [fm, out["inliers"], out["ok"]]
    return rp.FundamentalMatrix(*convert(fields if batched else [v[0] for v in fields], kind, device))


def _hartley(f: np.ndarray, c: np.ndarray) -> np.ndarray:
    k = np.zeros((f.shape[0], 3, 3))
    k[:, 0, 0] = k[:, 1, 1] = f
    k[:, :2, 2] = c
    k[:, 2, 2] = 1.0
    return k


def estimate_absolute_pose(points2d: object, points3d: object, k: object, *, threshold: float, mask: object) -> Any:  # noqa: ANN401 - absolute_pose.AbsolutePose (imported lazily: it imports this module)
    """`absolute_pose.estimate_absolute_pose(..., mode="mlx")`."""
    from fastforks.matching.absolute_pose import AbsolutePose  # noqa: PLC0415 - circular

    ops = _kernels()
    mx = ops.mx
    kind, device = _output((points2d, points3d, k, mask))
    batched, p2, p3, valid = _check(points2d, points3d, threshold, mask, MIN_MATCHES_ABS, (2, 3))
    P, N = p2.shape[:2]
    k = _intrinsics(k, "k", P)
    rng = np.random.default_rng(rp._SEED)  # noqa: SLF001
    order = _order(valid, rng)
    x = _normalise(np.take_along_axis(p2, order[..., None], 1), k)
    X = np.take_along_axis(p3, order[..., None], 1)
    counts = valid.sum(1)
    pts = mx.array(np.concatenate((x, X, np.zeros_like(X)), -1).astype(np.float32))
    thr2 = mx.array(((threshold / ((k[:, 0, 0] + k[:, 1, 1]) / 2)) ** 2).astype(np.float32))
    best, _, _ = _ransac(ops, pts, counts, thr2, SAMPLE_ABS, rp._ROUND, rp._TOP, 12, rng)  # noqa: SLF001
    lm = rp._LM_FINAL  # noqa: SLF001
    m, _, ok, mask_out = ops.polish_abs(
        pts, mx.array(counts.astype(np.int32)), thr2, mx.array(best), lm, MIN_MATCHES_ABS
    )
    m, ok = np.array(m), np.array(ok)
    inliers = np.zeros((P, N), dtype=bool)
    np.put_along_axis(inliers, order, np.array(mask_out), 1)
    R = np.where(ok[:, None, None], m[:, :9].reshape(P, 3, 3), np.eye(3, dtype=np.float32))
    t = np.where(ok[:, None], m[:, 9:], 0.0).astype(np.float32)
    fields = [R, t, inliers, ok]
    return AbsolutePose(*convert(fields if batched else [v[0] for v in fields], kind, device))
